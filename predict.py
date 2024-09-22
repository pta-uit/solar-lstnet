import argparse
import torch
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from models.LSTNet import Model
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using LSTNet model')
    parser.add_argument('--model_path', type=str, default='s3://trambk/solar-energy/model/modelv2.pt', help='S3 path to the model file')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data pickle file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save the predictions')
    parser.add_argument('--strategy', type=str, default='weighted_average', choices=['single', 'average', 'most_recent', 'weighted_average'], help='Strategy for processing predictions')
    parser.add_argument('--lambda_param', type=float, default=0.1, help='Lambda parameter for weighted average strategy')
    return parser.parse_args()

def load_model_from_s3(s3_path):
    s3 = S3FileSystem()
    
    def load_pickle(file_path):
        with s3.open(file_path, 'rb') as f:
            return pickle.load(f)

    try:
        state_dict = load_pickle(s3_path)
        metadata_path = s3_path.replace('.pt', '_metadata.pt')
        metadata = load_pickle(metadata_path)

        args = argparse.Namespace(**metadata['args'])
        data = type('Data', (), {'m': len(metadata['features'])})()
        
        model = Model(args, data)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def create_scaler(data):
    scaler = MinMaxScaler()
    reshaped_data = data.reshape(-1, data.shape[-1])
    scaler.fit(reshaped_data)
    return scaler

def preprocess_input(data, scaler):
    # Reshape the input data to 2D for scaling
    original_shape = data.shape
    reshaped_data = data.reshape(-1, data.shape[-1])
    
    # Scale the data
    scaled_data = scaler.transform(reshaped_data)
    
    # Reshape back to original shape
    scaled_data = scaled_data.reshape(original_shape)
    
    # Reshape to match the model's expected input
    # Assuming the model expects (batch_size, 100, 6, 24)
    batch_size, time_steps, features = scaled_data.shape
    reshaped_data = scaled_data.reshape(batch_size, 100, 6, 24)
    
    return torch.FloatTensor(reshaped_data)

def make_predictions(model, input_sequence, horizon, strategy='weighted_average'):
    with torch.no_grad():
        predictions = model(input_sequence)
    return predictions.numpy()

def process_predictions(predictions, timestamps, strategy='weighted_average', lambda_param=0.1):
    if strategy == 'single':
        return pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
    
    min_length = min(len(predictions), len(timestamps))
    predictions = predictions[:min_length]
    timestamps = timestamps[:min_length]
    
    predictions_df = pd.DataFrame(predictions, columns=[f'h{i+1}' for i in range(predictions.shape[1])])
    predictions_df['timestamp'] = timestamps
    
    melted = predictions_df.melt(id_vars=['timestamp'], var_name='horizon', value_name='prediction')
    melted['hour_offset'] = melted['horizon'].str.extract('(\d+)').astype(int)
    melted['pred_timestamp'] = melted['timestamp'] + pd.to_timedelta(melted['hour_offset'], unit='h')
    
    if strategy == 'average':
        return melted.groupby('pred_timestamp')['prediction'].mean().reset_index()
    elif strategy == 'most_recent':
        return melted.sort_values('timestamp').groupby('pred_timestamp').last().reset_index()
    elif strategy == 'weighted_average':
        melted['weight'] = np.exp(-lambda_param * melted['hour_offset'])
        weighted_avg = melted.groupby('pred_timestamp').apply(
            lambda x: np.average(x['prediction'], weights=x['weight'])
        ).reset_index(name='prediction')
        return weighted_avg
    else:
        raise ValueError("Invalid strategy.")

def inverse_transform_predictions(predictions_df, scaler, features, target_feature):
    dummy = pd.DataFrame(0, index=predictions_df.index, columns=features)
    dummy[target_feature] = predictions_df['prediction']
    unscaled = scaler.inverse_transform(dummy)
    return pd.DataFrame({
        'timestamp': predictions_df['pred_timestamp'],
        'prediction': unscaled[:, list(features).index(target_feature)]
    })

def main():
    args = parse_args()
    
    try:
        model, metadata = load_model_from_s3(args.model_path)
        if model is None or metadata is None:
            raise ValueError("Failed to load model or metadata")
        
        with open(args.input_data, 'rb') as f:
            input_data = pickle.load(f)
        
        features = input_data['features']
        X = input_data['X']
        
        scaler = create_scaler(X)
        input_sequence = preprocess_input(X, scaler)
        
        horizon = metadata['args']['horizon']
        raw_predictions = make_predictions(model, input_sequence, horizon, args.strategy)
        
        start_datetime = datetime.fromisoformat(input_data['start_datetime'])
        prediction_timestamps = [start_datetime + timedelta(hours=i) for i in range(X.shape[0])]
        
        processed_predictions = process_predictions(raw_predictions, prediction_timestamps, args.strategy, args.lambda_param)
        target_feature = input_data['target']
        final_predictions = inverse_transform_predictions(processed_predictions, scaler, features, target_feature)
        final_predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()