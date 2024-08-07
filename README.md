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
