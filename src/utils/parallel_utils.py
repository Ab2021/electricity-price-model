"""
This module contains utility functions for parallel processing in the electricity price prediction project.
"""

from sklearn.model_selection import cross_val_score
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Any, Callable, List, Tuple

def parallel_cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, n_jobs: int = -1) -> np.ndarray:
    """
    Perform parallel cross-validation.

    Args:
        model (Any): The model to cross-validate.
        X (np.ndarray): Features.
        y (np.ndarray): Target values.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        n_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors. Defaults to -1.

    Returns:
        np.ndarray: Cross-validation scores.
    """
    def _cv_score(fold: np.ndarray) -> float:
        return cross_val_score(model, X, y, cv=[fold]).mean()

    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        cv_scores = list(executor.map(_cv_score, np.array_split(range(len(X)), cv)))

    return np.array(cv_scores)

def parallel_apply(func: Callable, iterable: List[Tuple[Callable, Any]], n_jobs: int = -1) -> List[Any]:
    """
    Apply a function to an iterable in parallel.

    Args:
        func (Callable): The function to apply.
        iterable (List[Tuple[Callable, Any]]): The iterable of (function, data) tuples to apply the function to.
        n_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors. Defaults to -1.

    Returns:
        List[Any]: Results of applying the function to the iterable.
    """
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(executor.map(func, iterable))
    return results