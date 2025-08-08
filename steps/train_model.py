import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str = "LinearRegression",
) -> RegressorMixin:
    """Step to train the machine learning model on the injested data.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training labels.
        y_test (pd.DataFrame): Testing labels.
        config (ModelNameConfig): Configuration for the model name.
    Returns:
        RegressorMixin: The trained machine learning model.
    """
    model = None
    try:
        if model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        logging.info("Model training completed successfully.")
        return trained_model
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e