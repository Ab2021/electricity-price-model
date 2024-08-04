import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    label_encoder = LabelEncoder()
    
    for column in ['sectorName', 'stateDescription']:
        data[column] = label_encoder.fit_transform(data[column])
    
    data = data.drop(['customers', 'revenue', 'sales'], axis=1)
    
    return data

def split_features_target(data):
    X = data.drop('price', axis=1)
    y = data['price']
    return X, y