from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np

def create_model(model_type='rf', **kwargs):
    """Create a model based on the specified type."""
    if model_type == 'rf':
        return RandomForestRegressor(**kwargs)
    elif model_type == 'gb':
        return GradientBoostingRegressor(**kwargs)
    elif model_type == 'xgb':
        return xgb.XGBRegressor(**kwargs)
    elif model_type == 'lgb':
        return lgb.LGBMRegressor(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def tune_model(X, y, model_type='rf', param_distributions=None, n_iter=100, cv=5, n_jobs=-1):
    """Tune model hyperparameters using RandomizedSearchCV."""
    model = create_model(model_type)
    if param_distributions is None:
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                                       n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error',
                                       n_jobs=n_jobs, random_state=42)
    random_search.fit(X, y)
    return random_search.best_estimator_

def evaluate_model(model, X, y):
    """Evaluate the model using multiple metrics."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'R2': r2}

class EnsembleModel:
    """Ensemble of multiple models for improved predictions."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)