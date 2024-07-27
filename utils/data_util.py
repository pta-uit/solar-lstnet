import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

class DataUtil:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.data_file)
        self.df['DATE_TIME'] = pd.to_datetime(self.df['DATE_TIME'])
        self.df.set_index('DATE_TIME', inplace=True)
        return self.df

    def perform_feature_engineering(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

        # Seasonal Decomposition
        stl = STL(self.df['Solar Generation [W/kW]'], period=24)
        result = stl.fit()
        self.df['trend'] = result.trend
        self.df['seasonal'] = result.seasonal
        self.df['residual'] = result.resid

        # Time-based features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['month'] = self.df.index.month

        # Periodic features
        self.df['hour_sin'] = np.sin(self.df['hour'] * (2 * np.pi / 24))
        self.df['hour_cos'] = np.cos(self.df['hour'] * (2 * np.pi / 24))
        self.df['day_of_year_sin'] = np.sin(self.df['day_of_year'] * (2 * np.pi / 365))
        self.df['day_of_year_cos'] = np.cos(self.df['day_of_year'] * (2 * np.pi / 365))

        return self.df

    def prepare_sequences(self, seq_length, pred_length, features, target):
        if self.df is None:
            raise ValueError("Data not loaded and preprocessed. Call load_and_preprocess_data() and perform_feature_engineering() first.")

        data = self.scaler.fit_transform(self.df[features + [target]])

        xs, ys = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            x = data[i:(i+seq_length), :-1]
            y = data[(i+seq_length):(i+seq_length+pred_length), -1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)