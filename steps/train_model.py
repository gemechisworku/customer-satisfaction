import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
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
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
        
        logging.info("Model training completed successfully.")
        return trained_model
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e