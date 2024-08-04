import pandas as pd
import numpy as np

def create_season_feature(df):
    """Create a season feature based on the month."""
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['season'] = df['month'].map(season_map)
    return df

def create_lagged_features(df, lag_periods=[1, 3, 6]):
    """Create lagged price features."""
    for lag in lag_periods:
        df[f'price_lag_{lag}'] = df.groupby(['stateDescription', 'sectorName'])['price'].shift(lag)
    return df

def engineer_features(df):
    """Apply all feature engineering steps."""
    df = create_season_feature(df)
    df = create_lagged_features(df)
    return df