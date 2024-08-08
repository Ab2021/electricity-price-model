import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def plot_feature_importance(feature_importance: Dict[str, float], output_path: str = 'feature_importance.png') -> None:
    """
    Plot feature importance.

    Args:
        feature_importance (Dict[str, float]): Dictionary of feature names and their importance scores.
        output_path (str, optional): Path to save the plot. Defaults to 'feature_importance.png'.
    """
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    sns.barplot(x=importances[:10], y=features[:10])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()