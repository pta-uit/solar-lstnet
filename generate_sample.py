import argparse
import pickle
import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import random
from datetime import datetime, timedelta

def get_sample_dataframe(X, features, window, start_date, periods):
    # Randomly select a sample from X
    sample_index = random.randint(0, len(X) - 1)
    sample = X[sample_index]
    
    # Create a DataFrame with the correct number of rows and columns
    df = pd.DataFrame(sample, columns=features)
    
    # Adjust the sample to match the requested number of periods
    if periods > window:
        # Repeat the sample data to cover the requested periods
        repetitions = periods // window + 1
        df = pd.concat([df] * repetitions, ignore_index=True)
    
    # Trim or extend the DataFrame to match the exact number of periods
    df = df.iloc[:periods]
    
    # Add a datetime index
    df.index = pd.date_range(start=start_date, periods=periods, freq='H')
    
    return df

def load_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pickle.load(BytesIO(obj['Body'].read()))

def save_to_s3(df, bucket, key):
    s3 = boto3.client('s3')
    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

def parse_s3_path(s3_path):
    parts = s3_path.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

def main():
    parser = argparse.ArgumentParser(description='Generate a sample DataFrame for prediction')
    parser.add_argument('--preprocessed_data', type=str, required=True, help='S3 path to the preprocessed data file')
    parser.add_argument('--output', type=str, required=True, help='S3 path to the output folder')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD HH:MM:SS format')
    parser.add_argument('--hdata', type=int, help='Number of hours for historical data')
    parser.add_argument('--fdata', type=int, help='Number of hours for future data')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Parse S3 paths
    input_bucket, input_key = parse_s3_path(args.preprocessed_data)
    output_bucket, output_folder = parse_s3_path(args.output)

    # Load preprocessed data from S3
    preprocessed_data = load_from_s3(input_bucket, input_key)

    X = preprocessed_data['X']
    features = preprocessed_data['features']
    window = preprocessed_data['window']

    # Parse start date
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d %H:%M:%S')

    if args.hdata:
        # Generate historical data
        periods = args.hdata
        sample_df = get_sample_dataframe(X, features, window, start_date - timedelta(hours=periods), periods)
        output_key = f"{output_folder.rstrip('/')}/hdata.csv"
    elif args.fdata:
        # Generate future data
        periods = args.fdata
        sample_df = get_sample_dataframe(X, features, window, start_date, periods)
        output_key = f"{output_folder.rstrip('/')}/fdata.csv"
    else:
        raise ValueError("Either --hdata or --fdata must be provided")

    # Save the sample DataFrame to S3
    save_to_s3(sample_df, output_bucket, output_key)
    print(f"Sample DataFrame saved to s3://{output_bucket}/{output_key}")
    
    print("\nSample DataFrame:")
    print(sample_df)

if __name__ == "__main__":
    main()