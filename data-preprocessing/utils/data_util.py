import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

class DataUtil:
    def __init__(self, weather_file, building_file=None):
        self.weather_file = weather_file
        self.building_file = building_file
        self.df = None
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        df_weather = pd.read_csv(self.weather_file, usecols=['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]', 'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]'])

        if self.building_file:
            df_building = pd.read_csv(self.building_file, usecols=['Month', 'Hour', 'Solar Generation [W/kW]'])
            self.df = pd.concat([df_building[['Month', 'Hour']], df_weather, df_building[['Solar Generation [W/kW]']]], axis=1)
            self.df['DATE_TIME'] = pd.to_datetime({'year': 2022, 'month': self.df['Month'], 'day': 1, 'hour': self.df['Hour']})
            self.df.set_index('DATE_TIME', inplace=True)
            self.df.drop(columns=['Month', 'Hour'], inplace=True)
        else:
            self.df = df_weather
            self.df['DATE_TIME'] = pd.date_range(start='2022-01-01', periods=len(self.df), freq='H')
            self.df.set_index('DATE_TIME', inplace=True)

        return self.df

    def perform_feature_engineering(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

        self.df['hour'] = self.df.index.hour
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['month'] = self.df.index.month

        self.df['hour_sin'] = np.sin(self.df['hour'] * (2 * np.pi / 24))
        self.df['hour_cos'] = np.cos(self.df['hour'] * (2 * np.pi / 24))
        self.df['day_of_year_sin'] = np.sin(self.df['day_of_year'] * (2 * np.pi / 365))
        self.df['day_of_year_cos'] = np.cos(self.df['day_of_year'] * (2 * np.pi / 365))

        stl_series = self.df['Solar Generation [W/kW]'] if 'Solar Generation [W/kW]' in self.df.columns else self.df['Direct Solar Radiation [W/m2]']
        result = STL(stl_series, period=24).fit()
        self.df['trend'], self.df['seasonal'], self.df['residual'] = result.trend, result.seasonal, result.resid

        return self.df

    def prepare_sequences(self, window_size, horizon, features, target=None):
        if self.df is None:
            raise ValueError("Data not loaded and preprocessed. Call load_and_preprocess_data() and perform_feature_engineering() first.")

        if target and target in self.df.columns:
            data = self.scaler.fit_transform(self.df[features + [target]])
            X = np.array([data[i:(i+window_size), :-1] for i in range(len(data) - window_size - horizon + 1)])
            y = np.array([data[(i+window_size):(i+window_size+horizon), -1] for i in range(len(data) - window_size - horizon + 1)])
            return X, y
        else:
            data = self.scaler.fit_transform(self.df[features])
            X = np.array([data[i:(i+window_size), :] for i in range(len(data) - window_size + 1)])
            return X, None

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)