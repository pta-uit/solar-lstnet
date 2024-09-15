import argparse
import torch
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from models.LSTNet import Model
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using LSTNet model')
    parser.add_argument('--model_path', type=str, required=True, help='S3 path to the model file')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save the predictions')
    parser.add_argument('--preprocessed', action='store_true', help='Flag to indicate if input data is already preprocessed')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for more verbose output')
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

def preprocess_input(df, features, window_size, scaler=None):
    if scaler:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    else:
        df_scaled = df  # If data is already preprocessed, use it as is
    
    input_data = df_scaled[features].values
    input_sequence = np.array([input_data[i:i+window_size] for i in range(len(input_data)-window_size+1)])
    return torch.FloatTensor(input_sequence)

def make_predictions(model, input_sequence, horizon):
    with torch.no_grad():
        predictions = model(input_sequence)
    return predictions.numpy()

def inverse_transform_predictions(predictions, scaler, target_column_index):
    if scaler:
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, target_column_index] = predictions
        return scaler.inverse_transform(dummy)[:, target_column_index]
    else:
        return predictions  # If data was already preprocessed, return as is

def main():
    args = parse_args()
    
    try:
        model, metadata = load_model_from_s3(args.model_path)
        if model is None or metadata is None:
            raise ValueError("Failed to load model or metadata")
        print("Model loaded successfully.")
        
        if args.debug:
            print("Metadata keys:", metadata.keys())
            print("Metadata content:", metadata)

        # Load input data
        input_df = pd.read_csv(args.input_data, parse_dates=['timestamp'], index_col='timestamp')
        
        if args.debug:
            print("Input data shape:", input_df.shape)
            print("Input data columns:", input_df.columns)

        # Preprocess input data
        features = metadata['features']
        window_size = metadata['args']['window']
        scaler = metadata['scaler'] if not args.preprocessed else None
        input_sequence = preprocess_input(input_df, features, window_size, scaler)
        
        if args.debug:
            print("Input sequence shape:", input_sequence.shape)

        # Make predictions
        horizon = metadata['args']['horizon']
        raw_predictions = make_predictions(model, input_sequence, horizon)
        
        if args.debug:
            print("Raw predictions shape:", raw_predictions.shape)

        # Inverse transform predictions if necessary
        target = metadata.get('target', 'Solar Generation [W/kW]')  # Default target column name
        target_column_index = input_df.columns.get_loc(target)
        final_predictions = inverse_transform_predictions(raw_predictions, scaler, target_column_index)

        if args.debug:
            print("Final predictions shape:", final_predictions.shape)

        # Create a DataFrame with the predictions
        prediction_dates = input_df.index[window_size:] + pd.Timedelta(hours=1)
        
        # Ensure prediction_dates and final_predictions have the same length
        min_length = min(len(prediction_dates), final_predictions.shape[0])
        prediction_dates = prediction_dates[:min_length]
        final_predictions = final_predictions[:min_length]

        # Create a list of DataFrames, one for each prediction horizon
        prediction_dfs = []
        for i in range(final_predictions.shape[1]):
            df = pd.DataFrame({
                'timestamp': prediction_dates,
                f'predicted_solar_generation_h{i+1}': final_predictions[:, i]
            })
            prediction_dfs.append(df)

        # Merge all prediction DataFrames
        predictions_df = pd.concat(prediction_dfs, axis=1)
        predictions_df = predictions_df.loc[:, ~predictions_df.columns.duplicated()]  # Remove duplicate timestamp columns

        if args.debug:
            print("Predictions DataFrame shape:", predictions_df.shape)
            print("Predictions DataFrame columns:", predictions_df.columns)
            
        # Save predictions
        predictions_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        if args.debug:
            import traceback
            print("Traceback:")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()