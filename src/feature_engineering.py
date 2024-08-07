import pandas as pd
import numpy as np

def create_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a season feature based on the month.

    This function adds a 'season' column to the DataFrame, mapping months to seasons:
    0: Winter (Dec, Jan, Feb)
    1: Spring (Mar, Apr, May)
    2: Summer (Jun, Jul, Aug)
    3: Fall (Sep, Oct, Nov)

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'month' column.

    Returns:
        pd.DataFrame: DataFrame with the additional 'season' column.
    """
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['season'] = df['month'].map(season_map)
    return df

def create_lagged_features(df: pd.DataFrame, lag_periods: list = [1, 3, 6]) -> pd.DataFrame:
    """
    Create lagged price features.

    This function creates new columns with lagged price values for each combination
    of 'stateDescription' and 'sectorName'.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'stateDescription', 'sectorName', and 'price' columns.
        lag_periods (list, optional): List of lag periods to create. Defaults to [1, 3, 6].

    Returns:
        pd.DataFrame: DataFrame with additional lagged price columns.
    """
    for lag in lag_periods:
        df[f'price_lag_{lag}'] = df.groupby(['stateDescription', 'sectorName'])['price'].shift(lag)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    This function applies both the season feature creation and lagged feature creation
    to the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to engineer features for.

    Returns:
        pd.DataFrame: DataFrame with all engineered features added.
    """
    df = create_season_feature(df)
    df = create_lagged_features(df)
    return df