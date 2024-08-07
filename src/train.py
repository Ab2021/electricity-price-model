"""
This module contains the main training pipeline for the electricity price prediction project.
"""

from .data_processing import load_and_preprocess_data, split_data, get_feature_importance
from .model import train_model, evaluate_model, hyperparameter_tuning
from .utils.parallel_utils import parallel_cross_validate
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def plot_feature_importance(feature_importance: Dict[str, float], output_path: str = 'feature_importance.png') -> None:
    """
    Plot feature importance.

    Args:
        feature_importance (Dict[str, float]): Dictionary of feature names and their importance scores.
        output_path (str, optional): Path to save the plot. Defaults to 'feature_importance.png'.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(feature_importance.values())[:10], y=list(feature_importance.keys())[:10])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(file_path: str = 'data/clean_data.csv') -> Dict[str, Any]:
    """
    Main function to run the electricity price prediction model.

    Args:
        file_path (str, optional): Path to the CSV file containing the data. Defaults to 'data/clean_data.csv'.

    Returns:
        Dict[str, Any]: Dictionary containing model performance metrics and feature importance.
    """
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data)

    # Perform hyperparameter tuning
    tuning_results = hyperparameter_tuning(X_train, y_train)
    print("Best Hyperparameters:", tuning_results['best_params'])

    # Train model with best hyperparameters
    model = train_model(X_train, y_train, params=tuning_results['best_params'])

    # Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Train Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)

    # Perform parallel cross-validation
    cv_scores = parallel_cross_validate(model, X_train, y_train)
    print("Cross-validation scores:", cv_scores)

    # Get feature importance
    feature_importance = get_feature_importance(model, data.drop('price', axis=1).columns)
    plot_feature_importance(feature_importance)

    # Save the model
    joblib.dump(model, 'electricity_price_model.joblib')
    print("Model saved as 'electricity_price_model.joblib'")

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'best_hyperparameters': tuning_results['best_params']
    }

if __name__ == "__main__":
    results = main()
    print("Results:", results)