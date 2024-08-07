# Electricity Price Prediction

This project implements a machine learning model to predict electricity prices based on historical data.

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
│   ├── utils/
│   │   ├── init.py
│   │   └── parallel_utils.py
│   ├── init.py
│   ├── data_processing.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── init.py
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_train.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```
## Setup and Installation

1. Clone the repository:
```shell
git clone https://github.com/Ab2021/electricity-price-prediction.git
cd electricity-price-prediction
```
2. Create a virtual environment and activate it:
```shell
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install the required packages:
```shell
pip install -r requirements.txt
```
4. Place your data file (`clean_data.csv`) in the `data/` directory.

## Running the Model

To train the model and generate predictions, run:
```shell
python -m src.train
```

This will:
- Load and preprocess the data
- Perform hyperparameter tuning
- Train the model
- Evaluate the model
- Generate feature importance plot
- Save the trained model


## Docker

To build and run the Docker container:
```shell
docker build -t electricity-price-prediction .
docker run -v $(pwd)/data:/app/data electricity-price-prediction
```

## CI/CD

This project uses GitHub Actions for CI/CD. The workflow is defined in `.github/workflows/ci-cd.yml`.


## Results 
```
{'train_metrics': {'MSE': 0.041613860591040136, 'RMSE': 0.20399475628319502, 'MAE': 0.05541989799970074, 'R2': 0.9973069575228023}, 'test_metrics': {'MSE': 0.1717697277118982, 'RMSE': 0.41445111619091846, 'MAE': 0.14672366643555934, 'R2': 0.9888444868629134}, 'cv_scores': array([0.98699627, 0.98274219, 0.98123332, 0.97157564, 0.98467078]), 'feature_importance': {'price_lag_1': 0.2824700046113075, 'price_lag_3': 0.2326788627388238, 'price_lag_6': 0.23231904822664928, 'revenue': 0.07578251508724365, 'sales': 0.06160624387205448, 'customers': 0.032474337593836, 'year': 0.02918614449325012, 'stateDescription': 0.02615637577847093, 'sectorName': 0.012504378619725282, 'month': 0.008770738276698335, 'price_rolling_mean_6': 0.0028076920404266864, 'season': 0.0017268879993636978, 'price_per_customer': 0.0007869689696421799, 'sales_per_customer': 0.0007297870789207975, 'price_rolling_mean_3': 1.461358727166454e-08}, 'best_hyperparameters': {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None}}
```
