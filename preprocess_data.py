import argparse
import pickle
from utils.data_util import DataUtil
from s3fs.core import S3FileSystem
import os
import logging
import time

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for LSTNet Solar Generation Forecasting')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, help='Path to the building CSV file')
    parser.add_argument('--window', type=int, default=168, help='Window size (default: 168)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon (default: 24)')
    parser.add_argument('--output', type=str, required=True, help='S3 path to save the preprocessed data')
    return parser.parse_args()

def save_to_s3(s3_path, data, logger):
    logger.info(f"Saving preprocessed data to S3: {s3_path}")
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info("Data successfully saved to S3")

def main():
    start_time = time.time()
    logger = setup_logging()
    logger.info("Starting preprocessing task")

    args = parse_args()
    logger.info(f"Arguments parsed: weather_data={args.weather_data}, building_data={args.building_data}, window={args.window}, horizon={args.horizon}")

    logger.info("Initializing DataUtil")
    data_util = DataUtil(args.weather_data, args.building_data)

    logger.info("Loading and preprocessing data")
    df = data_util.load_and_preprocess_data()
    logger.info(f"Data loaded and preprocessed. Shape: {df.shape}")

    logger.info("Performing feature engineering")
    df = data_util.perform_feature_engineering()
    logger.info(f"Feature engineering completed. Final shape: {df.shape}")

    features = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
                'trend', 'seasonal', 'residual']
    target = 'Solar Generation [W/kW]' if args.building_data else None

    logger.info(f"Preparing sequences with window={args.window}, horizon={args.horizon}")
    X, y = data_util.prepare_sequences(args.window, args.horizon, features, target)
    logger.info(f"Sequences prepared. X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}")

    preprocessed_data = {
        'X': X,
        'y': y,
        'features': features,
        'target': target,
        'scaler': data_util.scaler,
        'window': args.window,
        'horizon': args.horizon
    }
    
    filename = "preprocessed_data.pkl" if args.building_data else "preprocessed_weather_only.pkl"
    output_path = os.path.join(args.output, filename)
    
    save_to_s3(output_path, preprocessed_data, logger)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Preprocessing task completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()