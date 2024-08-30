import os
import warnings
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
from botocore.exceptions import NoCredentialsError, ClientError
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

# Initialize global variables
model = None
data_util = None
args = None
device = None

def check_aws_credentials():
    try:
        boto3.client('sts').get_caller_identity()
    except NoCredentialsError:
        raise Exception("No AWS credentials found. Please ensure AWS credentials are properly configured.")

def load_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(response['Body'].read())
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

        # Load hyperparameters from S3
        print(f"Loading hyperparameters from {args.hyperparams_path}...")
        with load_from_s3(args.hyperparams_path) as f:
            hyperparams = json.load(f)
        print(f"Loaded hyperparameters: {hyperparams}")

        # Update args with loaded hyperparameters
        for key, value in hyperparams.items():
            setattr(args, key, value)
        print(f"Updated arguments: {args}")

        args.cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')

        print(f"Loading preprocessed data from {args.preprocessed_data}...")
        with load_from_s3(args.preprocessed_data) as f:
            preprocessed_data = pickle.load(f)

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
            except RuntimeError as e:
                print(f"Error loading with torch.load: {str(e)}")
                print("Attempting to load as a pickle file...")
                f.seek(0)
                try:
                    model_state = pickle.load(f)
                except Exception as pickle_error:
                    print(f"Error loading as pickle: {str(pickle_error)}")
                    raise RuntimeError("Unable to load model state using both torch.load and pickle.")

        print("Model state loaded. Analyzing state dict...")
        if isinstance(model_state, dict):
            # Determine the horizon from the loaded model
            loaded_horizon = model_state['linear1.weight'].size(0)
            print(f"Loaded model horizon: {loaded_horizon}")
            print(f"Current configuration horizon: {args.horizon}")
            
            if loaded_horizon != args.horizon:
                print(f"Mismatch in horizon. Adjusting model configuration to match loaded model...")
                args.horizon = loaded_horizon
            
            print("Initializing model with adjusted configuration...")
            model = Model(args, data).to(device)
            
            # Load the state dict
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

@app.route('/predict', methods=['POST'])
def predict():
    global model, data_util, device

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        features = data['features']

        input_data = np.array(features).reshape(1, -1, len(features))
        input_data = data_util.scaler.transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)
        input_tensor = torch.FloatTensor(input_data).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        prediction = prediction.cpu().numpy()
        prediction = data_util.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)