import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Encode categorical variables
    le = LabelEncoder()
    data['stateDescription'] = le.fit_transform(data['stateDescription'])
    data['sectorName'] = le.fit_transform(data['sectorName'])
    
    return data

def split_data(data, target_column='price'):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)