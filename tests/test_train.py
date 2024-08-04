import pytest
import pandas as pd
import numpy as np
from src.train import main
import os
import joblib

@pytest.fixture
def mock_data(tmp_path):
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
    
    # Introduce some NaN values to test robustness
    # data.loc[np.random.choice(data.index, 3), 'price'] = np.nan
    # data.loc[np.random.choice(data.index, 3), 'revenue'] = np.nan
    # data.loc[np.random.choice(data.index, 3), 'stateDescription'] = np.nan

    file_path = tmp_path / "clean_data.csv"
    data.to_csv(file_path, index=False)
    return file_path

def test_main(mock_data, tmp_path):
    os.chdir(tmp_path)
    main(mock_data)
    assert os.path.exists('best_model.joblib')
    model = joblib.load('best_model.joblib')
    assert hasattr(model, 'predict')