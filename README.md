# Electricity Price Prediction Project

## Overview
This project implements a machine learning model to predict electricity prices based on historical data. It uses an ensemble of Random Forest, Gradient Boosting, XGBoost, and LightGBM models to make predictions.

## Project Structure
```
electricity-price-prediction/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── data/
│   └── clean_data.csv
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_train.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── Dockerfile
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/electricity-price-prediction.git
   cd electricity-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your data is in the `data/` directory as `clean_data.csv`.

2. Run the training script:
   ```
   python src/train.py
   ```

3. The trained model will be saved as `best_model.joblib`.

## Running Tests

To run the tests, execute:

```
pytest
```

## Docker

To build and run the Docker container:

```
docker build -t electricity-price-prediction .
docker run -v $(pwd)/data:/app/data electricity-price-prediction
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline:

1. Runs tests on every push and pull request.
2. Trains the model if tests pass.
3. Builds a Docker image.

## Input Data

The `clean_data.csv` file should contain the following columns:
- year
- month
- stateDescription
- sectorName
- price (target variable)
- revenue
- sales
- customers

## Model Performance

The current model achieves the following performance:
- Train MSE: 0.31
- Test MSE: 0.73