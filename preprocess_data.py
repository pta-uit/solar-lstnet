import argparse
import numpy as np
import pandas as pd
from utils.data_util import DataUtil
import os
import pickle
from pickle import dump
from s3fs.core import S3FileSystem
import boto3
from botocore.exceptions import NoCredentialsError

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for LSTNet Solar Generation Forecasting')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='Path to the building CSV file')
    parser.add_argument('--window', type=int, default=168, help='Window size (default: 168)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon (default: 24)')
    parser.add_argument('--output', type=str, default='preprocessed_data.pkl', help='Path to save the preprocessed data')
    return parser.parse_args()

def load_s3(s3_path,arr):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        f.write(pickle.dumps(arr))

def main():
    args = parse_args()

    # Load and preprocess data
    data_util = DataUtil(args.weather_data, args.building_data)
    df = data_util.load_and_preprocess_data()
    df = data_util.perform_feature_engineering()

    # Define features and target
    features = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
                'trend', 'seasonal', 'residual']
    target = 'Solar Generation [W/kW]'

    # Prepare sequences
    X, y = data_util.prepare_sequences(args.window, args.horizon, features, target)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Save preprocessed data
    preprocessed_data = {
        'X': X,
        'y': y,
        'features': features,
        'target': target,
        'scaler': data_util.scaler,
        'window': args.window,
        'horizon': args.horizon
    }
    
    #os.makedirs(os.path.dirname(args.output), exist_ok=True)
    #with open(args.output, 'wb') as f:
     #   pickle.dump(preprocessed_data, f)

    load_s3(os.path.join(args.output,"preprocessed_data.pkl"),preprocessed_data)
    print(f"Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()