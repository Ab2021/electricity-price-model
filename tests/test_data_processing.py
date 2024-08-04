import unittest
import pandas as pd
from src.data_processing import load_and_preprocess_data, split_features_target

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'sectorName': ['Residential', 'Commercial', 'Industrial'],
            'stateDescription': ['State1', 'State2', 'State3'],
            'customers': [100, 200, 300],
            'revenue': [1000, 2000, 3000],
            'sales': [500, 1000, 1500],
            'price': [10, 20, 30],
            'year': [2020, 2021, 2022],
            'month': [1, 2, 3]
        })
        self.sample_data.to_csv('tests/sample_data.csv', index=False)

    def test_load_and_preprocess_data(self):
        processed_data = load_and_preprocess_data('tests/sample_data.csv')
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertNotIn('customers', processed_data.columns)
        self.assertNotIn('revenue', processed_data.columns)
        self.assertNotIn('sales', processed_data.columns)

    def test_split_features_target(self):
        processed_data = load_and_preprocess_data('tests/sample_data.csv')
        X, y = split_features_target(processed_data)
        self.assertEqual(X.shape[1], 5)
        self.assertEqual(y.shape[0], 3)

    def tearDown(self):
        import os
        if os.path.exists('tests/sample_data.csv'):
            os.remove('tests/sample_data.csv')

if __name__ == '__main__':
    unittest.main()