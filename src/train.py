import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import joblib
from data_processing import load_and_preprocess_data, split_features_target
from model import train_model, evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting model training pipeline")
    
    data = load_and_preprocess_data('data/clean_data.csv')
    X, y = split_features_target(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    train_mse, train_rmse = evaluate_model(model, X_train, y_train)
    test_mse, test_rmse = evaluate_model(model, X_test, y_test)
    
    logging.info(f"Train RMSE: {train_rmse:.4f}")
    logging.info(f"Test RMSE: {test_rmse:.4f}")
    
    joblib.dump(model, 'electricity_price_model.joblib')
    logging.info("Model saved successfully")

if __name__ == "__main__":
    main()