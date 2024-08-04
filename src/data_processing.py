import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Handle missing values
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Impute numeric columns with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

    # Impute categorical columns with most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    return data

def split_data(data, target_column='price', test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)