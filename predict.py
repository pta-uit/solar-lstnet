import argparse
import torch
import numpy as np
import pandas as pd
import json
import boto3
import io
import pickle
from models.LSTNet import Model
from sklearn.preprocessing import StandardScaler
from botocore.exceptions import NoCredentialsError, ClientError

# Define default features globally
DEFAULT_FEATURES = [
    'Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
    'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
    'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
    'trend', 'seasonal', 'residual'
]

class DataUtil:
    def __init__(self):
        self.scaler = None
        self.features = None

    def fit_scaler(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        self.features = data.columns.tolist()

    def transform(self, data):
        if self.features is None:
            raise ValueError("Features are not set. Call fit_scaler first.")
        
        # Ensure data has all required features
        for feature in self.features:
            if feature not in data.columns:
                data[feature] = 0  # or another appropriate default value
        
        # Reorder columns to match training data
        data = data[self.features]
        
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def parse_args():
    parser = argparse.ArgumentParser(description='LSTNet Solar Generation Prediction')
    parser.add_argument('--model_path', type=str, default='s3://trambk/solar-energy/model/model.pt', help='S3 path to the saved model')
    parser.add_argument('--params_path', type=str, default='s3://trambk/solar-energy/model/best_params.json', help='S3 path to the model parameters JSON file')
    parser.add_argument('--preprocessed_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/preprocessed_data.pkl', help='S3 path to the preprocessed data file')
    parser.add_argument('--historical_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/hdata.csv', help='S3 path to the historical data file (CSV)')
    parser.add_argument('--forecast_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/fdata.csv', help='S3 path to the weather forecast data file')
    parser.add_argument('--n_hours', type=int, required=True, help='Number of hours to predict')
    parser.add_argument('--output_path', type=str, default='s3://trambk/solar-energy/model/predicted.csv', help='S3 path to save the prediction results')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecast horizon')
    parser.add_argument('--window', type=int, default=24, help='Window size for LSTNet model')
    parser.add_argument('--output_fun', type=str, default='linear', help='Output function (linear, sigmoid, or tanh)')
    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def check_aws_credentials():
    try:
        boto3.client('sts').get_caller_identity()
        print("AWS credentials are valid.")
    except NoCredentialsError:
        raise Exception("No AWS credentials found. Please ensure AWS credentials are properly configured.")
    except ClientError as e:
        raise Exception(f"Error checking AWS credentials: {str(e)}")

def load_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj['Body'].read()
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise Exception(f"The specified file does not exist in the S3 bucket. Path: {s3_path}")
        elif e.response['Error']['Code'] == 'AccessDenied':
            raise Exception(f"Access denied to S3 bucket. Please check your AWS credentials and bucket permissions. Path: {s3_path}")
        else:
            raise Exception(f"An error occurred while accessing S3: {str(e)}")

def load_model(args, device):
    print(f"Loading hyperparameters from {args.params_path}...")
    with io.BytesIO(load_from_s3(args.params_path)) as f:
        hyperparams = json.load(f)
    print(f"Loaded hyperparameters: {hyperparams}")

    for key, value in hyperparams.items():
        setattr(args, key, value)
    print(f"Updated arguments: {args}")

    print(f"Loading preprocessed data from {args.preprocessed_data}...")
    with io.BytesIO(load_from_s3(args.preprocessed_data)) as f:
        preprocessed_data = pickle.load(f)
    print("Preprocessed data loaded successfully.")

    data_util = DataUtil()
    if 'scaler' in preprocessed_data:
        data_util.scaler = preprocessed_data['scaler']
        data_util.features = preprocessed_data.get('features', DEFAULT_FEATURES)
    else:
        print("Warning: No scaler found in preprocessed data. Creating a new scaler.")
        data_util.scaler = StandardScaler()
        data_util.features = DEFAULT_FEATURES

    print(f"Using features: {data_util.features}")

    num_features = len(data_util.features)

    class Data:
        def __init__(self, m):
            self.m = m

    data = Data(num_features)
    
    print(f"Loading model state from {args.model_path}...")
    with io.BytesIO(load_from_s3(args.model_path)) as f:
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
        print(f"Current configuration horizon: {args.horizon}")
        
        if loaded_horizon != args.horizon:
            print(f"Mismatch in horizon. Adjusting model configuration to match loaded model...")
            args.horizon = loaded_horizon
        
        print("Initializing model with adjusted configuration...")
        model = Model(args, data).to(device)
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in model_state.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(pretrained_dict) != len(model_dict):
            print(f"Warning: {len(model_dict) - len(pretrained_dict)} layers from the model were not loaded due to size mismatch.")
        model.load_state_dict(pretrained_dict, strict=False)
    elif isinstance(model_state, torch.nn.Module):
        print("Loaded state is a full model. Using it directly.")
        model = model_state.to(device)
    else:
        raise TypeError("Loaded object is neither a state dict nor a torch.nn.Module")

    model.eval()
    print("Model loaded successfully")
    return model, data_util, args, preprocessed_data

def load_csv_s3(s3_path):
    return pd.read_csv(io.BytesIO(load_from_s3(s3_path)), parse_dates=True, index_col=0)

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"Using device: {device}")

    try:
        check_aws_credentials()
        model, data_util, updated_args, preprocessed_data = load_model(args, device)
        
        # Load historical data
        if 'historical_data' in preprocessed_data:
            print("Using historical data from preprocessed file.")
            historical_data = preprocessed_data['historical_data']
        else:
            print(f"Loading historical data from {args.historical_data}...")
            historical_data = load_csv_s3(args.historical_data)
        
        print(f"Loading forecast data from {args.forecast_data}...")
        forecast_data = load_csv_s3(args.forecast_data)

        # Ensure all required features are present
        missing_features = set(data_util.features) - set(forecast_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in forecast data: {missing_features}")

        # Add missing columns with default values if necessary
        for feature in data_util.features:
            if feature not in historical_data.columns:
                historical_data[feature] = 0  # or another appropriate default value
            if feature not in forecast_data.columns:
                forecast_data[feature] = 0  # or another appropriate default value

        # Reorder columns to match training data
        historical_data = historical_data[data_util.features]
        forecast_data = forecast_data[data_util.features]

        combined_data = pd.concat([historical_data, forecast_data]).sort_index()
        
        if not data_util.scaler.n_features_in_:
            print("Fitting scaler on combined data...")
            data_util.fit_scaler(combined_data)
        
        scaled_data = data_util.transform(combined_data)

        window_size = updated_args.window
        X = scaled_data[-window_size:]

        predictions = []
        timestamps = []
        current_time = combined_data.index[-1]

        available_forecast_hours = (forecast_data.index[-1] - current_time).total_seconds() / 3600
        prediction_hours = min(args.n_hours, int(available_forecast_hours))

        if prediction_hours < args.n_hours:
            print(f"Warning: Forecast data only covers {prediction_hours:.2f} hours, but {args.n_hours} hours of prediction were requested.")
            print(f"Proceeding with prediction for {prediction_hours} hours.")

        with torch.no_grad():
            for _ in range(prediction_hours):
                X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)
                output = model(X_tensor)
                prediction = output.cpu().numpy()[0][0]
                predictions.append(prediction)

                current_time += pd.Timedelta(hours=1)
                timestamps.append(current_time)

                if current_time in forecast_data.index:
                    next_step_data = forecast_data.loc[current_time].values.reshape(1, -1)
                    next_step_scaled = data_util.transform(pd.DataFrame(next_step_data, columns=data_util.features))
                    X = np.roll(X, -1, axis=0)
                    X[-1] = next_step_scaled
                else:
                    print(f"Warning: No forecast data available for {current_time}. Using last available data.")

        predictions = data_util.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        prediction_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted Solar Generation [W/kW]': predictions
        })
        prediction_df.set_index('Timestamp', inplace=True)

        s3 = boto3.client('s3')
        bucket, key = args.output_path.replace("s3://", "").split("/", 1)
        csv_buffer = io.StringIO()
        prediction_df.to_csv(csv_buffer)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        print(f"Predictions for the next {prediction_hours} hours have been saved to {args.output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()