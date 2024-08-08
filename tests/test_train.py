
"""
This module contains integration tests for the training pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.train import main
import os
import joblib

@pytest.fixture
def mock_data(tmp_path):
    """Create mock data for testing."""
    data = pd.DataFrame({
        'year': np.repeat(2021, 3),  # Only one year
        'month': range(1, 4),  # Only 3 months
        'stateDescription': np.random.choice(['Wyoming', 'Tennessee', 'Texas'], 3),
        'sectorName': np.random.choice(['residential', 'commercial', 'industrial'], 3),
        'price': np.random.rand(3) * 100,
        'revenue': np.random.rand(3) * 800,
        'sales': np.random.rand(3) * 1000,
        'customers': np.random.randint(100, 1000, 3)
    })
    
    file_path = tmp_path / "clean_data.csv"
    data.to_csv(file_path, index=False)
    return file_path

def test_main(mock_data, tmp_path):
    """Test the main training pipeline."""
    os.chdir(tmp_path)
    results = main(mock_data)
    
    assert os.path.exists('electricity_price_model.joblib')
    assert os.path.exists('feature_importance.png')
    
    model = joblib.load('electricity_price_model.joblib')
    assert hasattr(model, 'predict')
    
    assert 'train_metrics' in results
    assert 'test_metrics' in results
    assert 'cv_scores' in results
    assert 'feature_importance' in results
    assert 'best_hyperparameters' in results
    
    assert all(metric in results['train_metrics'] for metric in ['MSE', 'RMSE', 'MAE', 'R2'])
    assert all(metric in results['test_metrics'] for metric in ['MSE', 'RMSE', 'MAE', 'R2'])
    
    assert isinstance(results['cv_scores'], np.ndarray)
    assert len(results['cv_scores']) == 5  # Assuming 5-fold cross-validation
    
    assert isinstance(results['feature_importance'], dict)
    assert len(results['feature_importance']) > 0
    
    assert isinstance(results['best_hyperparameters'], dict)
    assert len(results['best_hyperparameters']) > 0