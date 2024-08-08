
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        ValueError: If all data is dropped during preprocessing.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        logging.error("The CSV file is empty.")
        raise pd.errors.EmptyDataError("The CSV file is empty.")

    logging.info(f"Initial data shape: {data.shape}")

    data = _handle_missing_values(data)
    data = _encode_categorical_variables(data)
    data = _engineer_features(data)

    initial_shape = data.shape
    logging.info(f"Shape after preprocessing: {initial_shape}")

    # Check for infinite values and replace with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Instead of dropping NaN values, we'll fill them
    data = data.fillna(method='ffill').fillna(method='bfill')
    final_shape = data.shape

    logging.info(f"Shape after handling NaN values: {final_shape}")

    if final_shape[0] == 0:
        logging.error("All data was dropped during preprocessing.")
        raise ValueError("All data was dropped during preprocessing. Please check your data and preprocessing steps.")

    # Log summary statistics
    logging.info(f"Summary statistics:\n{data.describe()}")

    # Log column information
    for column in data.columns:
        logging.info(f"Column '{column}': {data[column].dtype}, {data[column].nunique()} unique values")

    return data

def _handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        data (pd.DataFrame): Input DataFrame with potentially missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    initial_missing = data.isnull().sum()
    logging.info(f"Initial missing values:\n{initial_missing}")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    final_missing = data.isnull().sum()
    logging.info(f"Final missing values:\n{final_missing}")

    return data

def _encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables in the dataset.

    Args:
        data (pd.DataFrame): Input DataFrame with categorical variables.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical variables.
    """
    le = LabelEncoder()
    categorical_columns = ['stateDescription', 'sectorName']
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col].astype(str))
        logging.info(f"Encoded {col}: {len(le.classes_)} unique values")
    return data

def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features using parallel processing.

    Args:
        data (pd.DataFrame): Input DataFrame for feature engineering.

    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    feature_functions = [
        _engineer_seasonal_features,
        _engineer_price_features,
        _engineer_lag_features,
        _engineer_rolling_features
    ]
    results = parallel_apply(_apply_feature_engineering, [(f, data.copy()) for f in feature_functions])
    
    for result in results:
        data = pd.concat([data, result], axis=1)
    
    logging.info(f"Engineered features: {list(data.columns)}")
    return data

def _apply_feature_engineering(args: Tuple[Callable, pd.DataFrame]) -> pd.DataFrame:
    """
    Apply a feature engineering function to a DataFrame.

    Args:
        args (Tuple[Callable, pd.DataFrame]): Tuple containing the engineering function and the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the feature engineering function applied.
    """
    func, data = args
    return func(data)

def _engineer_seasonal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer seasonal features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'year' and 'month' columns.

    Returns:
        pd.DataFrame: DataFrame with engineered seasonal features.
    """
    data['season'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str) + '-01').dt.month.map(
        {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
         7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'}
    )
    data['season'] = LabelEncoder().fit_transform(data['season'])
    logging.info(f"Engineered seasonal features: {data['season'].unique()}")
    return data[['season']]

def _engineer_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer price-related features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'revenue', 'customers', and 'sales' columns.

    Returns:
        pd.DataFrame: DataFrame with engineered price features.
    """
    eps = 1e-8  # Small value to avoid division by zero
    price_features = pd.DataFrame({
        'price_per_customer': data['revenue'] / (data['customers'] + eps),
        'sales_per_customer': data['sales'] / (data['customers'] + eps)
    })
    logging.info(f"Engineered price features: {list(price_features.columns)}")
    return price_features

def _engineer_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer lag features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'price', 'stateDescription', and 'sectorName' columns.

    Returns:
        pd.DataFrame: DataFrame with engineered lag features.
    """
    lag_features = {}
    for lag in [1, 3, 6]:
        lag_features[f'price_lag_{lag}'] = data.groupby(['stateDescription', 'sectorName'])['price'].shift(lag)
    lag_df = pd.DataFrame(lag_features)
    logging.info(f"Engineered lag features: {list(lag_df.columns)}")
    return lag_df

def _engineer_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rolling mean features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'price', 'stateDescription', and 'sectorName' columns.

    Returns:
        pd.DataFrame: DataFrame with engineered rolling mean features.
    """
    rolling_features = {}
    for window in [3, 6]:
        rolling_features[f'price_rolling_mean_{window}'] = (
            data.groupby(['stateDescription', 'sectorName'])['price']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    rolling_df = pd.DataFrame(rolling_features)
    logging.info(f"Engineered rolling features: {list(rolling_df.columns)}")
    return rolling_df

def split_data(data: pd.DataFrame, target_column: str = 'price') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): Input DataFrame to be split.
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

    logging.info(f"Training data shape: {X_train_scaled.shape}")
    logging.info(f"Testing data shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def get_feature_importance(model: object, feature_names: list) -> Dict[str, float]:
    """
    Get feature importance from the trained model.

    Args:
        model (object): Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.

    Returns:
        Dict[str, float]: Dictionary of feature names and their importance scores, sorted in descending order.
    """
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    logging.info(f"Top 5 important features: {list(sorted_importance.items())[:5]}")
    return sorted_importance