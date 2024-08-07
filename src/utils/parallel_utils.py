from sklearn.model_selection import cross_val_score
from typing import Any
import numpy as np

def parallel_cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
    """
    Perform cross-validation.

    Args:
        model (Any): The model to cross-validate.
        X (np.ndarray): Features.
        y (np.ndarray): Target values.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        np.ndarray: Cross-validation scores.
    """
    return cross_val_score(model, X, y, cv=cv, n_jobs=-1)