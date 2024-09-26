import argparse
import torch
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from models.LSTNet import Model
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using LSTNet model')
    parser.add_argument('--model_path', type=str, default='s3://trambk/solar-energy/model/modelv2.pt', help='S3 path to the model file')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the prepared input data pickle file')
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

def load_input_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data.reshape(-1, data.shape[-1]))
    return scaler

def preprocess_input(data, model_features, input_features, window_size, highway_window, scaler, h):
    print(f"Original data shape: {data.shape}")
    
    # Reorder and select features to match model features
    feature_indices = [input_features.index(feat) for feat in model_features if feat in input_features]
    data_reordered = data[:, :, feature_indices]
    
    print(f"Reordered data shape: {data_reordered.shape}")
    
    # Reshape the data for scaling
    data_reshaped = data_reordered.reshape(-1, data_reordered.shape[-1])
    
    # Check if the scaler needs to be refitted
    if scaler.n_features_in_ != data_reshaped.shape[1]:
        print(f"Refitting scaler to match {data_reshaped.shape[1]} features")
        scaler.fit(data_reshaped)
    
    data_scaled = scaler.transform(data_reshaped).reshape(data_reordered.shape)
    
    # Ensure the input sequence matches the expected window size
    if data_scaled.shape[1] > window_size:
        print(f"Truncating input sequence to match window size {window_size}")
        input_sequence = data_scaled[:, -window_size:, :]
    elif data_scaled.shape[1] < window_size:
        print(f"Padding input sequence to match window size {window_size}")
        pad_length = window_size - data_scaled.shape[1]
        padding = np.zeros((data_scaled.shape[0], pad_length, data_scaled.shape[2]))
        input_sequence = np.concatenate([padding, data_scaled], axis=1)
    else:
        input_sequence = data_scaled
    
    print(f"Preprocessed input shape: {input_sequence.shape}")
    
    # Adjust h if necessary
    h = min(h, window_size - highway_window)
    print(f"Adjusted h: {h}")
    
    return torch.FloatTensor(input_sequence), h

def make_predictions(model, input_sequence, h, horizon, strategy='weighted_average'):
    print(f"Input sequence shape: {input_sequence.shape}")
    print(f"Historical data length (h): {h}")
    print(f"Horizon: {horizon}")
    
    with torch.no_grad():
        predictions = []
        for i in range(input_sequence.shape[0]):
            window = input_sequence[i]
            window = window.unsqueeze(0)  # Add batch dimension
            print(f"Window shape: {window.shape}")
            pred = model(window)
            predictions.append(pred.squeeze().numpy())
        predictions = np.array(predictions)
    
    print(f"Raw predictions shape: {predictions.shape}")
    
    if strategy == 'single':
        return predictions[:, 0] if predictions.ndim > 1 else predictions
    else:
        return predictions

def process_predictions(predictions, timestamps, h, strategy='weighted_average', lambda_param=0.1):
    print(f"Processing predictions. Shape: {predictions.shape}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Historical data length (h): {h}")
    
    forecast_timestamps = timestamps[h:]
    print(f"Number of forecast timestamps: {len(forecast_timestamps)}")
    
    if len(forecast_timestamps) == 0:
        print("Warning: forecast_timestamps is empty. Adjusting h.")
        h = max(0, len(timestamps) - 1)
        forecast_timestamps = timestamps[h:]
    
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    predictions_df = pd.DataFrame(predictions, columns=[f'h{i+1}' for i in range(predictions.shape[1])])
    
    if len(forecast_timestamps) > 0:
        predictions_df['timestamp'] = forecast_timestamps[:len(predictions_df)]
    else:
        print("Error: Unable to assign timestamps. No forecast timestamps available.")
        return None
    
    print(f"Predictions DataFrame shape: {predictions_df.shape}")
    
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
    print(f"Inverse transforming predictions. Shape: {predictions_df.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Target feature: {target_feature}")
    print(f"Scaler n_features_in_: {scaler.n_features_in_}")

    # Find the index of the target feature in the original feature list
    target_index = features.index(target_feature)
    
    # Create a list of features used by the scaler (excluding the target feature)
    scaler_features = [f for f in features if f != target_feature]
    print(f"Scaler features: {scaler_features}")
    
    # Create a dummy array with the correct number of features for the scaler
    dummy = np.zeros((len(predictions_df), scaler.n_features_in_))
    
    # Assign the predictions to a temporary column in the dummy array
    dummy[:, 0] = predictions_df['prediction'].values
    
    print(f"Dummy array shape before inverse transform: {dummy.shape}")
    
    # Perform inverse transform
    unscaled = scaler.inverse_transform(dummy)
    
    print(f"Unscaled array shape: {unscaled.shape}")
    
    # Extract the predictions from the first column (where we assigned them)
    unscaled_predictions = unscaled[:, 0]
    
    return pd.DataFrame({
        'timestamp': predictions_df['pred_timestamp'],
        'prediction': unscaled_predictions
    })

def main():
    args = parse_args()
    
    try:
        model, metadata = load_model_from_s3(args.model_path)
        if model is None or metadata is None:
            raise ValueError("Failed to load model or metadata")
        
        print("Model metadata:")
        print(json.dumps(metadata, indent=2))
        
        input_data = load_input_data(args.input_data)
        print(f"Input data shape: {input_data['X'].shape}")
        print(f"Input data features: {input_data['features']}")
        
        model_features = metadata['features']
        window_size = metadata['args']['window']
        highway_window = metadata['args']['highway_window']
        horizon = metadata['args']['horizon']
        h = input_data['h']  # Use the h value from the input data
        
        print(f"Original h: {h}")
        print(f"Window size: {window_size}")
        print(f"Highway window: {highway_window}")
        print(f"Horizon: {horizon}")
        
        scaler = create_scaler(input_data['X'])
        input_sequence, h = preprocess_input(input_data['X'], model_features, input_data['features'], 
                                             window_size, highway_window, scaler, h)
        
        print(f"Model input shape: {input_sequence.shape}")
        print(f"Adjusted historical data length (h): {h}")
        
        raw_predictions = make_predictions(model, input_sequence, h, horizon, args.strategy)
        print(f"Raw predictions shape: {raw_predictions.shape}")

        start_datetime = pd.to_datetime(input_data['start_datetime'])
        prediction_timestamps = [start_datetime + pd.Timedelta(hours=i) for i in range(len(input_sequence))]
        print(f"Number of prediction timestamps: {len(prediction_timestamps)}")
        print(f"First timestamp: {prediction_timestamps[0]}")
        print(f"Last timestamp: {prediction_timestamps[-1]}")
        
        processed_predictions = process_predictions(raw_predictions, prediction_timestamps, h, args.strategy, args.lambda_param)
        
        if processed_predictions is None:
            raise ValueError("Failed to process predictions")
        
        print(f"Processed predictions shape: {processed_predictions.shape}")
        print(f"Processed predictions columns: {processed_predictions.columns}")
        print(f"First few rows of processed predictions:")
        print(processed_predictions.head())
        print(f"Scaler feature names: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'Not available'}")

        target_feature = input_data['target']
        final_predictions = inverse_transform_predictions(processed_predictions, scaler, input_data['features'], target_feature)
        
        final_predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        print(f"Final predictions shape: {final_predictions.shape}")
        print(f"First 5 predictions:")
        print(final_predictions.head())

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()