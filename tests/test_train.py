import pytest
import pandas as pd
import numpy as np
from src.train import main
import os

@pytest.fixture
def mock_data(tmp_path):
    data = pd.DataFrame({
        'year': np.repeat(range(2010, 2022), 12),
        'month': np.tile(range(1, 13), 12),
        'stateDescription': np.random.choice(['State1', 'State2', 'State3'], 144),
        'sectorName': np.random.choice(['Residential', 'Commercial', 'Industrial'], 144),
        'price': np.random.rand(144) * 100,
        'revenue': np.random.rand(144) * 1000,
        'sales': np.random.rand(144) * 500,
        'customers': np.random.randint(100, 1000, 144)
    })
    file_path = tmp_path / "clean_data.csv"
    data.to_csv(file_path, index=False)
    return file_path

def test_main(mock_data, monkeypatch):
    monkeypatch.chdir(os.path.dirname(mock_data))
    main()
    assert os.path.exists('best_model.joblib')