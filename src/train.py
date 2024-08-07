
import logging
import sys
from typing import Dict, Any
import joblib
from .data_processing import load_and_preprocess_data, split_data, get_feature_importance
from .model import train_model, evaluate_model, hyperparameter_tuning
from .utils.parallel_utils import parallel_cross_validate
from .visualization import plot_feature_importance

def main(file_path: str = 'data/clean_data.csv') -> Dict[str, Any]:
    """
    Main function to run the electricity price prediction model.

    This function orchestrates the entire process of loading data, preprocessing,
    training the model, evaluating its performance, and saving the results.

    Args:
        file_path (str, optional): Path to the CSV file containing the data. 
                                   Defaults to 'data/clean_data.csv'.

    Returns:
        Dict[str, Any]: A dictionary containing the following keys:
            - 'train_metrics': Evaluation metrics on the training data
            - 'test_metrics': Evaluation metrics on the test data
            - 'cv_scores': Cross-validation scores
            - 'feature_importance': Feature importance scores
            - 'best_hyperparameters': Best hyperparameters found during tuning

    Raises:
        Exception: If any error occurs during the process, it's logged and re-raised.
    """
    try:
        # Load and preprocess data
        data = load_and_preprocess_data(file_path)
        X_train, X_test, y_train, y_test = split_data(data)

        # Perform hyperparameter tuning
        tuning_results = hyperparameter_tuning(X_train, y_train)
        logging.info(f"Best Hyperparameters: {tuning_results['best_params']}")

        # Train model with best hyperparameters
        model = train_model(X_train, y_train, params=tuning_results['best_params'])

        # Evaluate model
        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        logging.info(f"Train Metrics: {train_metrics}")
        logging.info(f"Test Metrics: {test_metrics}")

        # Perform parallel cross-validation
        cv_scores = parallel_cross_validate(model, X_train, y_train)
        logging.info(f"Cross-validation scores: {cv_scores}")

        # Get feature importance
        feature_importance = get_feature_importance(model, data.drop('price', axis=1).columns)
        plot_feature_importance(feature_importance)

        # Save the model
        joblib.dump(model, 'electricity_price_model.joblib')
        logging.info("Model saved as 'electricity_price_model.joblib'")

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'best_hyperparameters': tuning_results['best_params']
        }

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        results = main()
        logging.info(f"Results: {results}")
    except Exception as e:
        logging.error(f"An error occurred in main execution: {str(e)}")
        sys.exit(1)