"""
Feature Engineering Module
Creates features from preprocessed data for demand forecasting.
"""

import pandas as pd
import numpy as np


def run_feature_pipeline(df):
    """
    Create features from preprocessed data.
    
    Args:
        df: pandas.DataFrame, preprocessed data
    
    Returns:
        pandas.DataFrame: Data with engineered features
    """
    # Make a copy to avoid modifying original
    df_features = df.copy()
    
    # Ensure sorted by Product ID and Date
    df_features = df_features.sort_values(['Product ID', 'Date']).reset_index(drop=True)
    
    # Create lag features for Units Sold: 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30
    lag_periods = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]
    for lag in lag_periods:
        df_features[f'lag_{lag}'] = df_features.groupby('Product ID')['Units Sold'].shift(lag)
    
    # Create rolling mean features: windows 3, 7, 14, 21, 30
    rolling_windows = [3, 7, 14, 21, 30]
    for window in rolling_windows:
        df_features[f'rolling_mean_{window}'] = df_features.groupby('Product ID')['Units Sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    # Create rolling standard deviation features: windows 7, 14, 30
    rolling_std_windows = [7, 14, 30]
    for window in rolling_std_windows:
        df_features[f'rolling_std_{window}'] = df_features.groupby('Product ID')['Units Sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    # Create day-based features
    df_features['day_of_week'] = df_features['Date'].dt.dayofweek
    df_features['week_of_year'] = df_features['Date'].dt.isocalendar().week
    df_features['month'] = df_features['Date'].dt.month
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Create target columns (demand_t+1 to demand_t+7)
    for i in range(1, 8):
        df_features[f'demand_t+{i}'] = df_features.groupby('Product ID')['Units Sold'].shift(-i)
    
    # Drop rows without available future targets (last 7 rows per product)
    df_features = df_features.dropna(subset=[f'demand_t+{i}' for i in range(1, 8)])
    
    # Fill any remaining NaN values with 0 for numeric columns
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
    
    return df_features


