import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.model import train_model, evaluate_model

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y = X['feature1'] * 2 + X['feature2'] * 3 + np.random.rand(100)
    return X, y

def test_train_model(sample_data):
    X, y = sample_data
    model = train_model(X,  y)
    
    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, 'predict')
    assert model.n_features_in_ == X.shape[1]

def test_evaluate_model(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    
    assert isinstance(metrics, dict)
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    assert 0 <= metrics['R2'] <= 1
    assert metrics['MSE'] >= 0
    assert metrics['RMSE'] >= 0

def test_model_prediction(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    
    # Test prediction on new data
    new_data = pd.DataFrame({
        'feature1': [0.5],
        'feature2': [0.5],
        'feature3': [0.5]
    })
    prediction = model.predict(new_data)
    
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])