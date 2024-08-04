
# Electricity Price Prediction Model

![Tests](https://github.com/Ab2021/electricity-price-prediction/workflows/Run%20Tests/badge.svg)
![CI](https://github.com/Ab2021/electricity-price-prediction/workflows/CI/badge.svg)

## Project Overview

This project implements a machine learning model to predict electricity prices based on historical data. It uses a Random Forest Regressor and includes data processing, model training, and evaluation components.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Ab2021/electricity-price-prediction.git
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

3. The trained model will be saved as `electricity_price_model.joblib`.

## Running Tests

To run the tests, execute:

```
pytest
```

## Docker

To build and run the Docker container:

```
docker build -t electricity-price-prediction .
docker run electricity-price-prediction
```