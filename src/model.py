"""
This module contains functions for model training, evaluation, and hyperparameter tuning
in the electricity price prediction project.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from typing import Dict, Any
from .utils.parallel_utils import parallel_cross_validate

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any] = None) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        params (Dict[str, Any], optional): Model parameters. If None, default parameters are used.

    Returns:
        RandomForestRegressor: Trained model.
    """
    if params is None:
        params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestRegressor, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the model's performance.

    Args:
        model (RandomForestRegressor): Trained model.
        X (np.ndarray): Features to evaluate.
        y (np.ndarray): True target values.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics (MSE, RMSE, MAE, R2).
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 100, cv: int = 5) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for the Random Forest Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        n_iter (int, optional): Number of iterations for random search. Defaults to 100.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        Dict[str, Any]: Best hyperparameters and the best score.
    """
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return {
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_
    }