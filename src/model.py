from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X, y):
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse