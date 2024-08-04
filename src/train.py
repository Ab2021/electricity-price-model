from data_processing import load_and_preprocess_data, split_data
from model import train_model, evaluate_model
import joblib

def main(file_path='data/clean_data.csv'):
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)

    # Save the model
    joblib.dump(model, 'electricity_price_model.joblib')
    print("Model saved as 'electricity_price_model.joblib'")

if __name__ == "__main__":
    main()