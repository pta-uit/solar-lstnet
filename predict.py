import argparse
import torch
import numpy as np
import pandas as pd
from models.LSTNet import Model
import json
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='LSTNet Solar Generation Prediction')
    parser.add_argument('--model_path', type=str, default='s3://trambk/solar-energy/model/model.pt', help='S3 path to the saved model')
    parser.add_argument('--params_path', type=str, default='s3://trambk/solar-energy/model/best_params.json', help='S3 path to the model parameters JSON file')
    parser.add_argument('--historical_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/hdata.csv', help='S3 path to the historical data file')
    parser.add_argument('--forecast_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/fdata.csv', help='S3 path to the weather forecast data file')
    parser.add_argument('--n_hours', type=int, required=True, help='Number of hours to predict')
    parser.add_argument('--output_path', type=str, default='s3://trambk/solar-energy/model/predicted.csv', help='S3 path to save the prediction results')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    return parser.parse_args()

def load_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    try:
        print(f"Attempting to download file from S3: {s3_path}")
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        print(f"Successfully downloaded file from S3: {s3_path}")
        buffer.seek(0)
        return buffer
    except ClientError as e:
        print(f"Error downloading from S3: {str(e)}")
        sys.exit(1)

def load_model(model_path, params, data, device):
    with load_from_s3(model_path) as f:
        try:
            model_state = torch.load(f, map_location=device)
            print("Model state loaded successfully using torch.load")
        except RuntimeError as e:
            print(f"Error loading with torch.load: {str(e)}")
            print("Attempting to load as a pickle file...")
            f.seek(0)
            try:
                model_state = pickle.load(f)
                print("Model state loaded successfully using pickle")
            except Exception as pickle_error:
                print(f"Error loading as pickle: {str(pickle_error)}")
                raise RuntimeError("Unable to load model state using both torch.load and pickle.")

    print("Analyzing state dict...")
    if isinstance(model_state, dict):
        loaded_horizon = model_state['linear1.weight'].size(0)
        print(f"Loaded model horizon: {loaded_horizon}")
        print(f"Current configuration horizon: {params['horizon']}")
        
        if loaded_horizon != params['horizon']:
            print(f"Mismatch in horizon. Adjusting model configuration to match loaded model...")
            params['horizon'] = loaded_horizon
        
        print("Initializing model with adjusted configuration...")
        model = Model(argparse.Namespace(**params), data).to(device)
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in model_state.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(pretrained_dict) != len(model_dict):
            print(f"Warning: {len(model_dict) - len(pretrained_dict)} layers from the model were not loaded due to size mismatch.")
        model.load_state_dict(pretrained_dict, strict=False)
    elif isinstance(model_state, torch.nn.Module):
        print("Loaded state is a full model. Using it directly.")
        model = model_state
    else:
        raise TypeError("Loaded object is neither a state dict nor a torch.nn.Module")

    return model

def load_json_s3(s3_path):
    with load_from_s3(s3_path) as f:
        return json.load(f)

def load_csv_s3(s3_path):
    with load_from_s3(s3_path) as f:
        return pd.read_csv(f, parse_dates=True, index_col=0)

def validate_forecast_data(forecast_data, n_hours):
    forecast_range = (forecast_data.index[-1] - forecast_data.index[0]).total_seconds() / 3600
    if forecast_range < n_hours:
        raise ValueError(f"Forecast data only covers {forecast_range:.2f} hours, but {n_hours} hours of prediction were requested.")

def preprocess_data(historical_data, forecast_data, window_size, features):
    combined_data = pd.concat([historical_data, forecast_data])
    combined_data = combined_data.sort_index()
    combined_data = combined_data[features]
    combined_data = (combined_data - combined_data.mean()) / combined_data.std()
    X = combined_data.values[-window_size:].reshape(1, window_size, -1)
    return X, combined_data

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    params = load_json_s3(args.params_path)
    
    # Add default values for missing parameters
    default_params = {
        "window": 24,
        "horizon": 1,  # Set to 1 for single-step prediction, 24 for multi-step
        "highway_window": 24,
        "skip": 24,
        "hidCNN": 100,
        "hidRNN": 100,
        "hidSkip": 5,
        "CNN_kernel": 6,
        "dropout": 0.2,
        "output_fun": "sigmoid",
        "cuda": True if args.gpu >= 0 and torch.cuda.is_available() else False,
    }
    
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    
    features = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
                'trend', 'seasonal', 'residual']
    
    class Data:
        def __init__(self, m):
            self.m = m
    
    data = Data(len(features))
    
    try:
        model = load_model(args.model_path, params, data, device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Model parameters:", params)
        sys.exit(1)

    model.to(device)
    model.eval()

    historical_data = load_csv_s3(args.historical_data)
    forecast_data = load_csv_s3(args.forecast_data)

    try:
        validate_forecast_data(forecast_data, args.n_hours)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    try:
        X, combined_data = preprocess_data(historical_data, forecast_data, params['window'], features)
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        sys.exit(1)

    predictions = []
    timestamps = []
    current_time = combined_data.index[-1]
    
    try:
        with torch.no_grad():
            for i in range(args.n_hours):
                X_tensor = torch.FloatTensor(X).to(device)
                output = model(X_tensor)
                prediction = output.cpu().numpy()[0]
                predictions.append(prediction[0])  # For single-step prediction
                # For multi-step prediction, you might use: predictions.extend(prediction)
                
                current_time += pd.Timedelta(hours=1)
                timestamps.append(current_time)
                
                next_step_data = forecast_data.loc[current_time][features].values
                X = np.roll(X, -1, axis=1)
                X[0, -1, :] = next_step_data
                
                # For multi-step prediction, you might only update X every 24 iterations
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

    prediction_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Predicted Solar Generation [W/kW]': predictions
    })
    prediction_df.set_index('Timestamp', inplace=True)
    
    try:
        s3 = boto3.client('s3')
        bucket, key = args.output_path.replace("s3://", "").split("/", 1)
        csv_buffer = io.StringIO()
        prediction_df.to_csv(csv_buffer)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        print(f"Predictions for the next {args.n_hours} hours have been saved to {args.output_path}")
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()