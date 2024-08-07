"""
This module contains functions for data processing in the electricity price prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from .utils.parallel_utils import parallel_apply

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the electricity price data.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with engineered features.

    Raises:
        FileNotFoundError: If the specified file is not found.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The CSV file is empty.")

    print(f"Initial data shape: {data.shape}")

    data = _handle_missing_values(data)
    data = _encode_categorical_variables(data)
    data = _engineer_features(data)

    # Now drop NaN values after all preprocessing steps
    initial_shape = data.shape
    data = data.dropna()
    final_shape = data.shape

    print(f"Shape after preprocessing: {initial_shape}")
    print(f"Shape after dropping NaN values: {final_shape}")

    if final_shape[0] == 0:
        raise ValueError("All data was dropped during preprocessing. Please check your data and preprocessing steps.")

    return data

def _handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    return data

def _encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables in the dataset."""
    le = LabelEncoder()
    categorical_columns = ['stateDescription', 'sectorName']
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))
    return data

def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features using parallel processing."""
    feature_functions = [
        _engineer_seasonal_features,
        _engineer_price_features,
        _engineer_lag_features,
        _engineer_rolling_features
    ]
    results = parallel_apply(_apply_feature_engineering, [(f, data.copy()) for f in feature_functions])
    
    for result in results:
        data = pd.concat([data, result], axis=1)
    
    return data

def _apply_feature_engineering(args: Tuple[Callable, pd.DataFrame]) -> pd.DataFrame:
    """Apply a feature engineering function to a DataFrame."""
    func, data = args
    return func(data)

def _engineer_seasonal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer seasonal features."""
    data['season'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-01').dt.month.map(
        {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
         7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    )
    data['season'] = LabelEncoder().fit_transform(data['season'])
    return data[['season']]

def _engineer_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer price-related features."""
    return pd.DataFrame({
        'price_per_customer': data['revenue'] / data['customers'].replace(0, np.nan),
        'sales_per_customer': data['sales'] / data['customers'].replace(0, np.nan)
    })

def _engineer_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer lag features."""
    lag_features = {}
    for lag in [1, 3, 6]:
        lag_features[f'price_lag_{lag}'] = data.groupby(['stateDescription', 'sectorName'])['price'].shift(lag)
    return pd.DataFrame(lag_features)

def _engineer_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer rolling mean features."""
    rolling_features = {}
    for window in [3, 6]:
        rolling_features[f'price_rolling_mean_{window}'] = (
            data.groupby(['stateDescription', 'sectorName'])['price']
            .rolling(window=window, min_periods=1)  # Use min_periods=1 to avoid NaN for the first rows
            .mean()
            .reset_index(0, drop=True)
        )
    return pd.DataFrame(rolling_features)

def split_data(data: pd.DataFrame, target_column: str = 'price') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): Preprocessed DataFrame.
        target_column (str, optional): Name of the target column. Defaults to 'price'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def get_feature_importance(model: object, feature_names: list) -> Dict[str, float]:
    """
    Get feature importance from the trained model.

    Args:
        model (object): Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.

    Returns:
        Dict[str, float]: Dictionary of feature names and their importance scores.
    """
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))