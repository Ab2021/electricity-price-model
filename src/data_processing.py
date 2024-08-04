import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    label_encoder = LabelEncoder()
    data['sectorName'] = label_encoder.fit_transform(data['sectorName'])
    data['stateDescription'] = label_encoder.fit_transform(data['stateDescription'])
    return data

def split_data(data, target_column='price', test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = data.drop([target_column, 'customers', 'revenue', 'sales'], axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)