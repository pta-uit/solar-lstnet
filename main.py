import argparse
import torch
import numpy as np
import pandas as pd
from models.LSTNet import Model
from sklearn.model_selection import train_test_split
import os
import json
import pickle
from s3fs.core import S3FileSystem
import boto3
from botocore.exceptions import NoCredentialsError

def parse_args():
    parser = argparse.ArgumentParser(description='LSTNet for Solar Generation Forecasting')
    
    # Required arguments
    parser.add_argument('--preprocessed_data', type=str, required=True, help='Path to the preprocessed data file')
    
    # Optional arguments with default values
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    parser.add_argument('--hidRNN', type=int, default=100, help='Number of RNN hidden units (default: 100)')
    parser.add_argument('--hidCNN', type=int, default=100, help='Number of CNN hidden units (default: 100)')
    parser.add_argument('--hidSkip', type=int, default=5, help='Number of skip RNN hidden units (default: 5)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size (default: 6)')
    parser.add_argument('--skip', type=int, default=24, help='Skip length (default: 24)')
    parser.add_argument('--highway_window', type=int, default=24, help='Highway window size (default: 24)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--output_fun', type=str, default='sigmoid', help='Output function: sigmoid, tanh or None (default: sigmoid)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--loss_history', type=str, default=None, help='Path to save the loss history (default: None, i.e., do not save)')

    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def load_s3(s3_path,arr):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        f.write(pickle.dumps(arr))

def get_s3(s3_path):
    s3 = S3FileSystem()
    return np.load(s3.open(s3_path), allow_pickle=True)

def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
   # with open(args.preprocessed_data, 'rb') as f:
    #    preprocessed_data = pickle.load(f)

    #s3client = boto3.client('s3')
    #response = s3client.get_object(Bucket='trambk', Key=args.preprocessed_data)
    #body = response['Body'].read()
    #preprocessed_data = pickle.loads(body)

    preprocessed_data = get_s3(args.preprocessed_data)
    
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    features = preprocessed_data['features']
    
    # Add window and horizon to args
    args.window = preprocessed_data['window']
    args.horizon = preprocessed_data['horizon']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # Create a Data object to pass to LSTNet
    class Data:
        def __init__(self, m):
            self.m = m

    data = Data(len(features))

    # Initialize the model
    model = Model(args, data).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) / args.batch_size)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                batch_X = X_val[i:i+args.batch_size]
                batch_y = y_val[i:i+args.batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(X_val) / args.batch_size)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the loss history if specified
    if args.loss_history:
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        os.makedirs(os.path.dirname(args.loss_history), exist_ok=True)
        with open(args.loss_history, 'w') as f:
            json.dump(history, f)
        print(f'Loss history saved to {args.loss_history}')

    # Save the model
    torch.save(model.state_dict(), "model.pt")
    #print(f'Model saved to {args.save}')

    load_s3(os.path.join(args.save,"model.pt"),model.state_dict())
    print(f'Saved model to {args.save}')

if __name__ == "__main__":
    main()