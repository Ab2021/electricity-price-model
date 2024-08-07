from sklearn.model_selection import cross_val_score
from typing import Any, Callable, List
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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

def parallel_apply(func: Callable, iterable: List[Any], n_jobs: int = -1) -> List[Any]:
    """
    Apply a function to an iterable in parallel.

    Args:
        func (Callable): The function to apply.
        iterable (List[Any]): The iterable to apply the function to.
        n_jobs (int, optional): Number of jobs to run in parallel. -1 means using all processors. Defaults to -1.

    Returns:
        List[Any]: Results of applying the function to the iterable.
    """
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(executor.map(func, iterable))
    return results