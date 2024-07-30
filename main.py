import argparse
import torch
import numpy as np
import pandas as pd
from models.LSTNet import Model
from utils.data_util import DataUtil
from sklearn.model_selection import train_test_split

class Args:
    def __init__(self):
        self.cuda = False
        self.window = 24 * 7
        self.horizon = 24
        self.hidRNN = 100
        self.hidCNN = 100
        self.hidSkip = 5
        self.CNN_kernel = 6
        self.skip = 24
        self.highway_window = 24
        self.dropout = 0.2
        self.output_fun = 'sigmoid'

def parse_args():
    parser = argparse.ArgumentParser(description='LSTNet for Solar Generation Forecasting')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='Path to the building CSV file')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    parser.add_argument('--window', type=int, default=24*7, help='Window size')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon')
    parser.add_argument('--hidRNN', type=int, default=100, help='Number of RNN hidden units')
    parser.add_argument('--hidCNN', type=int, default=100, help='Number of CNN hidden units')
    parser.add_argument('--hidSkip', type=int, default=5, help='Number of skip RNN hidden units')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size')
    parser.add_argument('--skip', type=int, default=24, help='Skip length')
    parser.add_argument('--highway_window', type=int, default=24, help='Highway window size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--output_fun', type=str, default='sigmoid', help='Output function: sigmoid, tanh or None')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()

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

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

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

    # Training loop
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
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), args.save)
    print(f'Model saved to {args.save}')

if __name__ == "__main__":
    main()