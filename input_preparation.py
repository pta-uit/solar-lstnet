import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

def load_pickle(file_path):
    return pd.read_pickle(file_path)

def load_json_report(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_input_data(historical_data_path, weather_forecast_data_path, target_date, h, f):
    # Load data
    historical_data = load_pickle(historical_data_path)
    weather_forecast_data = load_pickle(weather_forecast_data_path)

    # Load JSON reports
    historical_report = load_json_report(historical_data_path.replace('.pkl', '.json'))
    weather_forecast_report = load_json_report(weather_forecast_data_path.replace('.pkl', '.json'))

    # Convert target_date to datetime
    target_datetime = pd.to_datetime(target_date)

    # Get features
    historical_features = historical_report['features']
    forecast_features = weather_forecast_report['features']

    # Calculate indices for slicing
    historical_start_time = pd.to_datetime(historical_report['start_datetime'])
    forecast_start_time = pd.to_datetime(weather_forecast_report['start_datetime'])
    
    historical_hours = int((target_datetime - historical_start_time).total_seconds() / 3600)
    forecast_hours = int((target_datetime - forecast_start_time).total_seconds() / 3600)
    
    historical_start_index = max(0, historical_hours - h)
    historical_end_index = historical_hours
    forecast_start_index = max(0, forecast_hours)
    forecast_end_index = forecast_start_index + f

    print(f"Historical start time: {historical_start_time + timedelta(hours=historical_start_index)}")
    print(f"Historical end time: {historical_start_time + timedelta(hours=historical_end_index)}")
    print(f"Forecast start time: {forecast_start_time + timedelta(hours=forecast_start_index)}")
    print(f"Forecast end time: {forecast_start_time + timedelta(hours=forecast_end_index)}")

    # Slice the data
    historical_slice = historical_data['X'][historical_start_index:historical_end_index]
    forecast_slice = weather_forecast_data['X'][forecast_start_index:forecast_end_index]

    # Create a mask for common features
    common_features = list(set(historical_features) & set(forecast_features))
    historical_feature_indices = [historical_features.index(feat) for feat in common_features]
    forecast_feature_indices = [forecast_features.index(feat) for feat in common_features]

    # Apply the mask to the data slices
    historical_slice = historical_slice[:, :, historical_feature_indices]
    forecast_slice = forecast_slice[:, :, forecast_feature_indices]

    # Combine datasets
    combined_data = np.concatenate([historical_slice, forecast_slice], axis=0)

    print(f"Original historical data shape: {historical_data['X'].shape}")
    print(f"Original historical target shape: {historical_data['y'].shape}")
    print(f"Sliced historical data shape: {historical_slice.shape}")
    print(f"Forecast slice shape: {forecast_slice.shape}")
    print(f"Combined data shape: {combined_data.shape}")

    # Add Solar Generation column (empty for forecast data)
    if 'y' in historical_data and historical_report['target'] is not None:
        historical_target = historical_data['y'][historical_start_index:historical_end_index]
        
        print(f"Sliced historical target shape: {historical_target.shape}")
        
        # Reshape historical_target to match the shape of combined_data
        solar_generation = np.zeros((combined_data.shape[0], combined_data.shape[1], 1))
        
        # Assuming the 24 values in historical_target represent daily data
        # and the 168 time steps in historical_slice represent hourly data for a week
        for i in range(len(historical_target)):
            day_index = i % 7
            solar_generation[i, day_index*24:(day_index+1)*24, 0] = historical_target[i]
        
        # Concatenate solar_generation to combined_data
        combined_data = np.concatenate([combined_data, solar_generation], axis=2)
        common_features.append('Solar Generation [W/kW]')

    # Ensure the combined data has the correct shape
    target_length = h + f
    current_length = combined_data.shape[0]
    if current_length < target_length:
        pad_length = target_length - current_length
        padding = np.zeros((pad_length, combined_data.shape[1], combined_data.shape[2]))
        combined_data = np.concatenate([combined_data, padding], axis=0)
    elif current_length > target_length:
        combined_data = combined_data[:target_length]

    # Ensure we have exactly 168 time steps (1 week) for each sample
    time_steps = 168
    num_samples = combined_data.shape[0] // time_steps
    combined_data = combined_data[:num_samples * time_steps].reshape(num_samples, time_steps, -1)

    print(f"Final combined data shape: {combined_data.shape}")

    # Create a dictionary to store the prepared data
    prepared_data = {
        'X': combined_data,
        'features': common_features,
        'start_datetime': (historical_start_time + timedelta(hours=historical_start_index)).isoformat(),
        'target': 'Solar Generation [W/kW]' if 'y' in historical_data else None,
        'h': h,
        'f': f
    }
    # Add debugging information
    print(f"Target datetime: {target_datetime}")
    print(f"Historical data range: {historical_start_index} to {historical_end_index}")
    print(f"Forecast data range: {forecast_start_index} to {forecast_end_index}")
    print(f"Common features: {common_features}")
    print(f"Prepared data time range: {prepared_data['start_datetime']} to {pd.to_datetime(prepared_data['start_datetime']) + timedelta(hours=combined_data.shape[0])}")
    
    return prepared_data

def main():
    parser = argparse.ArgumentParser(description="Prepare input data for prediction.")
    parser.add_argument("--historical_data", required=True, help="Path to historical data pickle file")
    parser.add_argument("--weather_forecast_data", required=True, help="Path to weather forecast data pickle file")
    parser.add_argument("--datetime", required=True, help="Target datetime (format: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--h", type=int, required=True, help="Number of historical hours to include")
    parser.add_argument("--f", type=int, required=True, help="Number of forecast hours to include")
    parser.add_argument("--output", default="prepared_data.pkl", help="Output pickle file name")

    args = parser.parse_args()

    try:
        prepared_data = prepare_input_data(args.historical_data, args.weather_forecast_data, 
                                           args.datetime, args.h, args.f)

        # Save to pickle file
        pd.to_pickle(prepared_data, args.output)
        print(f"Prepared data saved to {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()