import os
import warnings
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.LSTNet import Model
import torch
import numpy as np
from utils.data_util import DataUtil
import pickle
import boto3
import json
import io
import sys
from botocore.exceptions import NoCredentialsError, ClientError
import argparse
from statsmodels.tsa.seasonal import STL
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

model = None
data_util = None
args = None
device = None

FEATURE_NAMES = [
    'Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
    'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
    'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
    'trend', 'seasonal', 'residual'
]

class ProgressPercentage(object):
    def __init__(self, filename, file_size):
        self._filename = filename
        self._size = file_size
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(f"\r{self._filename}  {self._seen_so_far} / {self._size}  ({percentage:.2f}%)")
            sys.stdout.flush()

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
        print(f"Attempting to download file from S3: {s3_path}")
        file_size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']
        print(f"File size: {file_size} bytes")
        
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, key, buffer, 
                            Callback=ProgressPercentage(key, file_size))
        print(f"\nSuccessfully downloaded file from S3: {s3_path}")
        buffer.seek(0)
        return buffer
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            raise Exception(f"Access denied to S3 bucket. Please check your AWS credentials and bucket permissions. Bucket: {bucket}, Key: {key}")
        elif e.response['Error']['Code'] == 'NoSuchKey':
            raise Exception(f"The specified file does not exist in the S3 bucket. Bucket: {bucket}, Key: {key}")
        else:
            raise Exception(f"An error occurred while accessing S3: {str(e)}")

def load_model():
    global model, data_util, args, device

    try:
        print("Checking AWS credentials...")
        check_aws_credentials()
        
        parser = argparse.ArgumentParser(description='LSTNet for Solar Generation Forecasting')
        parser.add_argument('--preprocessed_data', type=str, default='s3://trambk/solar-energy/preprocessed_data/preprocessed_data.pkl', help='S3 path to the preprocessed data file')
        parser.add_argument('--model_path', type=str, default='s3://trambk/solar-energy/model/model.pt', help='S3 path to the saved model')
        parser.add_argument('--hyperparams_path', type=str, default='s3://trambk/solar-energy/model/best_params.json', help='S3 path to the hyperparameters JSON file')
        parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
        parser.add_argument('--window', type=int, default=24, help='Window size for LSTNet model')
        parser.add_argument('--horizon', type=int, default=12, help='Forecast horizon')
        parser.add_argument('--output_fun', type=str, default='linear', help='Output function (linear, sigmoid, or tanh)')
        
        args = parser.parse_args()
        print(f"Initial arguments: {args}")

        print(f"Loading hyperparameters from {args.hyperparams_path}...")
        with load_from_s3(args.hyperparams_path) as f:
            hyperparams = json.load(f)
        print(f"Loaded hyperparameters: {hyperparams}")

        for key, value in hyperparams.items():
            setattr(args, key, value)
        print(f"Updated arguments: {args}")

        args.cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
        print(f"Using device: {device}")

        print(f"Loading preprocessed data from {args.preprocessed_data}...")
        with load_from_s3(args.preprocessed_data) as f:
            preprocessed_data = pickle.load(f)
        print("Preprocessed data loaded successfully.")

        data_util = DataUtil(None, None)
        data_util.scaler = preprocessed_data['scaler']

        class Data:
            def __init__(self, m):
                self.m = m

        data = Data(len(preprocessed_data['features']))
        
        print(f"Loading model state from {args.model_path}...")
        with load_from_s3(args.model_path) as f:
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
            model = model_state
        else:
            raise TypeError("Loaded object is neither a state dict nor a torch.nn.Module")

        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)

    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month

    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_of_year_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 365))
    df['day_of_year_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 365))

    df['trend'] = 0
    df['seasonal'] = 0
    df['residual'] = 0

    return df[FEATURE_NAMES]

@app.route('/predict', methods=['POST'])
def predict():
    global model, data_util, args, device

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        if not all(feature in data for feature in FEATURE_NAMES + ['DATE_TIME']):
            missing_features = [f for f in FEATURE_NAMES + ['DATE_TIME'] if f not in data]
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        df = preprocess_input(data)
        
        scaled_data = data_util.scaler.transform(df)

        input_data = np.tile(scaled_data, (1, args.window, 1))
        input_tensor = torch.FloatTensor(input_data).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        prediction = prediction.cpu().numpy()
        prediction = data_util.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return jsonify({'prediction': prediction.tolist(), 'target': 'Solar Generation [W/kW]'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting the application...")
    load_model()
    print("Model loaded, starting Flask server...")
    app.run(host='0.0.0.0', port=5000)