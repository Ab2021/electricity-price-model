from src.data_processing import load_data, preprocess_data, split_data
from src.feature_engineering import engineer_features
from src.model import create_model, tune_model, evaluate_model, EnsembleModel
import joblib

def main(file_path='data/clean_data.csv'):
    # Load and preprocess data
    data = load_data(file_path)
    data = engineer_features(data)
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data)

    # Train and tune multiple models
    models = []
    for model_type in ['rf', 'gb', 'xgb', 'lgb']:
        tuned_model = tune_model(X_train, y_train, model_type=model_type)
        models.append(tuned_model)
        
        train_metrics = evaluate_model(tuned_model, X_train, y_train)
        test_metrics = evaluate_model(tuned_model, X_test, y_test)
        
        print(f"{model_type.upper()} Train Metrics:", train_metrics)
        print(f"{model_type.upper()} Test Metrics:", test_metrics)

    # Create and evaluate ensemble model
    ensemble = EnsembleModel(models)
    ensemble_train_metrics = evaluate_model(ensemble, X_train, y_train)
    ensemble_test_metrics = evaluate_model(ensemble, X_test, y_test)
    
    print("Ensemble Train Metrics:", ensemble_train_metrics)
    print("Ensemble Test Metrics:", ensemble_test_metrics)

    # Save the ensemble model
    joblib.dump(ensemble, 'best_model.joblib')

if __name__ == "__main__":
    main()