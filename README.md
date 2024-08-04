
# Electricity Price Prediction Model

This project contains a machine learning model for predicting electricity prices based on a static historical dataset. It uses MLflow for experiment tracking and model versioning.

## Setup

1. Clone the repository:


```shell
git clone https://github.com/Ab2021/electricity-price-model.git
cd electricity-price-model
```


2. Install dependencies:



```shell
pip install -r requirements.txt
```


3. Set up MLflow:
    ## MLflow Setup

    1. Start the MLflow server:
    ```shell
    ./start_mlflow_server.sh
    ```
    2. Set the MLFLOW_TRACKING_URI environment variable:
    ```shell
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    ```
    3. Access the MLflow UI by opening a web browser and navigating to:
    http://localhost:5000

    - Set the `MLFLOW_TRACKING_URI` environment variable to your MLflow server URL

4. Run tests:


```shell
python -m unittest discover tests
```


5. Train the model:


```shell
python src/train.py
```


## MLflow Integration

This project uses MLflow for:
- Experiment tracking: Hyperparameters, metrics, and artifacts are logged for each run.
- Model versioning: Trained models are registered in the MLflow Model Registry.

To view experiments and models:
1. Start the MLflow UI: `mlflow ui`
2. Open a web browser and go to `http://localhost:5000`

## Deployment

This project uses GitHub Actions for CI/CD. See `.github/workflows/ci-cd.yml` for details.

To deploy manually:

1. Build Docker image:


```shell
docker build -t electricity-price-model .
```

2. Push to GitHub Container Registry:

```shell
docker push [ghcr.io/YOUR_USERNAME/electricity-price-model:latest](http://ghcr.io/YOUR_USERNAME/electricity-price-model:latest)
```


3. Deploy to Kubernetes:


```shell
kubectl apply -f deployment.yaml
```


## Project Structure

- `src/`: Contains the main Python scripts
- `tests/`: Contains unit tests and integration tests
- `data/`: Contains the static dataset
- `mlruns/`: MLflow tracking files (if using local storage)
- `Dockerfile`: Defines the container for the application
- `deployment.yaml`: Kubernetes deployment configuration
- `.github/workflows/`: Contains GitHub Actions workflow file

## Monitoring and Logging

Logging is implemented in the Python scripts. For monitoring, set up Prometheus and Grafana in your Kubernetes cluster. MLflow provides additional monitoring for model performance and experiment tracking.