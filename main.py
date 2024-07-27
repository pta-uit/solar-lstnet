import argparse
import torch
import numpy as np
import pandas as pd
from models.LSTNet import LSTNet
from utils.data_util import DataUtil
from utils.evaluator import Evaluator
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='LSTNet for Solar Generation Forecasting')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='Path to the building CSV file')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    parser.add_argument('--hidSkip', type=int, default=5, help='Number of skipped hidden states')
    parser.add_argument('--output_fun', type=str, default='Linear', help='Output function: Linear or Sigmoid')
    parser.add_argument('--seq_length', type=int, default=24, help='Input sequence length')
    parser.add_argument('--pred_length', type=int, default=24, help='Prediction sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')

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
    X, y = data_util.prepare_sequences(args.seq_length, args.pred_length, features, target)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # Initialize the model
    model = LSTNet(
        input_size=len(features),
        seq_length=args.seq_length,
        pred_length=args.pred_length,
        skip_steps=args.hidSkip,
        output_fun=args.output_fun
    ).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), args.save)
    print(f'Model saved to {args.save}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert back to numpy for evaluation
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Initialize the evaluator
    evaluator = Evaluator()

    # Evaluate the model
    r2, mse, mae = evaluator.evaluate(y_test, y_pred)

    # Print evaluation metrics
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Plot results
    evaluator.plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()