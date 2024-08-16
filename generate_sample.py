import argparse
import pickle
import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import random

def get_sample_dataframe(X, features, window):
    # Randomly select a sample from X
    sample_index = random.randint(0, len(X) - 1)
    sample = X[sample_index]
    
    # Create a DataFrame with the correct number of rows and columns
    df = pd.DataFrame(sample, columns=features)
    
    # Add a datetime index
    df.index = pd.date_range(end='2023-08-16', periods=window, freq='H')
    
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
    parser.add_argument('--output', type=str, default='s3://your-bucket/sample_data.csv', help='S3 path to save the sample DataFrame')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Parse S3 paths
    input_bucket, input_key = parse_s3_path(args.preprocessed_data)
    output_bucket, output_key = parse_s3_path(args.output)

    # Load preprocessed data from S3
    preprocessed_data = load_from_s3(input_bucket, input_key)

    X = preprocessed_data['X']
    features = preprocessed_data['features']
    window = preprocessed_data['window']

    # Get a sample DataFrame
    sample_df = get_sample_dataframe(X, features, window)
    
    # Save the sample DataFrame to S3
    save_to_s3(sample_df, output_bucket, output_key)
    print(f"Sample DataFrame saved to s3://{output_bucket}/{output_key}")
    
    print("\nSample DataFrame:")
    print(sample_df)

if __name__ == "__main__":
    main()