import matplotlib
matplotlib.use('Agg')

import argparse
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.LSTNet import Model
from utils.data_util import DataUtil
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LSTNet for Solar Generation Forecasting')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='Path to the building CSV file')
    
    # Optional arguments
    parser.add_argument('--output_folder', type=str, default='evaluation_results', help='Folder to save output figures and results')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--window', type=int, default=168, help='Window size (default: 168)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon (default: 24)')
    parser.add_argument('--hidRNN', type=int, default=100, help='Number of RNN hidden units (default: 100)')
    parser.add_argument('--hidCNN', type=int, default=100, help='Number of CNN hidden units (default: 100)')
    parser.add_argument('--hidSkip', type=int, default=5, help='Number of skip RNN hidden units (default: 5)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size (default: 6)')
    parser.add_argument('--skip', type=int, default=24, help='Skip length (default: 24)')
    parser.add_argument('--highway_window', type=int, default=24, help='Highway window size (default: 24)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--output_fun', type=str, default='sigmoid', help='Output function: sigmoid, tanh or None (default: sigmoid)')
    parser.add_argument('--loss_history', type=str, default=None, help='Path to the loss history JSON file (default: None, i.e., do not plot loss history)')
    
    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def load_model(model_path, args, device):
    data = type('Data', (), {'m': 12})()  # Dummy Data object with m attribute
    model = Model(args, data).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def plot_loss_history(loss_history_path, output_folder):
    if not loss_history_path or not os.path.exists(loss_history_path):
        print(f"Loss history file not found at {loss_history_path}. Skipping loss history plot.")
        return

    with open(loss_history_path, 'r') as f:
        history = json.load(f)

    train_losses = history['train_loss']
    val_losses = history['val_loss']

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'loss_history.png'))
    plt.close()

    print(f"Loss history plot saved in folder: '{output_folder}'")

def evaluate_model(model, X_test, y_test, data_util, target_idx):
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    y_test = y_test.cpu().numpy()
    X_test = X_test.cpu().numpy()
    X_test_last = X_test[:, -1, :]

    y_pred_inv = np.zeros_like(y_pred)
    y_test_inv = np.zeros_like(y_test)

    for i in range(y_pred.shape[1]):
        combined_pred = np.concatenate((X_test_last, y_pred[:, i].reshape(-1, 1)), axis=1)
        combined_test = np.concatenate((X_test_last, y_test[:, i].reshape(-1, 1)), axis=1)
        y_pred_inv[:, i] = data_util.inverse_transform(combined_pred)[:, target_idx]
        y_test_inv[:, i] = data_util.inverse_transform(combined_test)[:, target_idx]

    y_pred_flat = y_pred_inv.flatten()
    y_test_flat = y_test_inv.flatten()

    r2 = r2_score(y_test_flat, y_pred_flat)
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)

    # Calculate metrics for scaled data
    y_pred_scaled = y_pred.flatten()
    y_test_scaled = y_test.flatten()
    r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)

    return r2, mse, mae, r2_scaled, mse_scaled, mae_scaled, y_pred_inv, y_test_inv

def plot_results(y_test, y_pred, output_folder, horizon):
    os.makedirs(output_folder, exist_ok=True)

    # Time Series Plot
    plt.figure(figsize=(15, 6))
    subset_size = min(500, len(y_test))  # Show at most 500 data points
    plt.plot(y_test[-subset_size:, 0], label='Actual', alpha=0.7)
    plt.plot(y_pred[-subset_size:, 0], label='Predicted (1-hour horizon)', alpha=0.7)
    plt.title(f'Actual vs Predicted Solar Generation (Last {subset_size} points, 1-hour horizon)')
    plt.xlabel('Time Steps')
    plt.ylabel('Solar Generation [W/kW]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'time_series_plot_1h.png'))
    plt.close()

    # Forecast plot (plot first sample)
    plt.figure(figsize=(12, 6))
    hours = range(1, horizon + 1)
    plt.plot(hours, y_test[0], label='Actual', marker='o')
    plt.plot(hours, y_pred[0], label='Predicted', marker='x')
    plt.xlabel('Forecast Horizon (hours)')
    plt.ylabel('Solar Generation [W/kW]')
    plt.title(f'Solar Generation Forecast (Horizon: {horizon} hours)')
    plt.legend()
    plt.xticks(hours)
    plt.grid(True)
    plt.tight_layout()
    forecast_path = os.path.join(output_folder, 'forecast_plot.png')
    plt.savefig(forecast_path)
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Values (1-hour horizon)')
    plt.xlabel('Actual Solar Generation [W/kW]')
    plt.ylabel('Predicted Solar Generation [W/kW]')
    plt.savefig(os.path.join(output_folder, 'scatter_plot_1h.png'))
    plt.close()

    # Error Distribution Plot
    errors = y_pred[:, 0] - y_test[:, 0]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.title('Distribution of Prediction Errors (1-hour horizon)')
    plt.xlabel('Prediction Error [W/kW]')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_distribution_1h.png'))
    plt.close()

    # Horizon Performance Plot
    mae_per_horizon = []
    rmse_per_horizon = []
    
    for i in range(horizon):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae_per_horizon.append(mae)
        rmse_per_horizon.append(rmse)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, horizon + 1), mae_per_horizon, label='MAE', marker='o')
    plt.plot(range(1, horizon + 1), rmse_per_horizon, label='RMSE', marker='s')
    plt.title('Model Performance Across Forecast Horizon')
    plt.xlabel('Forecast Horizon (hours ahead)')
    plt.ylabel('Error [W/kW]')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, horizon + 1, 2))  # Show every other hour on x-axis
    plt.fill_between(range(1, horizon + 1), mae_per_horizon, rmse_per_horizon, alpha=0.2, label='MAE-RMSE gap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'horizon_performance.png'))
    plt.close()

    print(f"Plots saved in folder: '{output_folder}'")

def save_metrics(r2, mse, mae, r2_scaled, mse_scaled, mae_scaled, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    metrics_path = os.path.join(output_folder, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Metrics for original scale:\n')
        f.write(f'R-squared (R2) Score: {r2:.4f}\n')
        f.write(f'Mean Squared Error (MSE): {mse:.4f}\n')
        f.write(f'Mean Absolute Error (MAE): {mae:.4f}\n')
        f.write(f'\nMetrics for scaled data:\n')
        f.write(f'R-squared (R2) Score: {r2_scaled:.4f}\n')
        f.write(f'Mean Squared Error (MSE): {mse_scaled:.4f}\n')
        f.write(f'Mean Absolute Error (MAE): {mae_scaled:.4f}\n')
    print(f"Metrics saved in: '{metrics_path}'")

def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"Using device: {device}")

    data_util = DataUtil(args.weather_data, args.building_data)
    df = data_util.load_and_preprocess_data()
    df = data_util.perform_feature_engineering()

    features = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
                'trend', 'seasonal', 'residual']
    target = 'Solar Generation [W/kW]'

    all_columns = features + [target]
    target_idx = all_columns.index(target)

    X, y = data_util.prepare_sequences(args.window, args.horizon, features, target)
    
    split_point = int(0.8 * len(X))
    X_test = torch.FloatTensor(X[split_point:]).to(device)
    y_test = torch.FloatTensor(y[split_point:]).to(device)

    model = load_model(args.model, args, device)

    r2, mse, mae, r2_scaled, mse_scaled, mae_scaled, y_pred_inv, y_test_inv = evaluate_model(model, X_test, y_test, data_util, target_idx)

    print(f'R-squared (R2) Score: {r2:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    save_metrics(r2, mse, mae, r2_scaled, mse_scaled, mae_scaled, args.output_folder)

    plot_results(y_test_inv, y_pred_inv, args.output_folder, args.horizon)

    if args.loss_history:
        plot_loss_history(args.loss_history, args.output_folder)

if __name__ == "__main__":
    main()