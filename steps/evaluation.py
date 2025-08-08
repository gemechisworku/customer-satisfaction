import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple [
    Annotated[float, "R2 Score"],
    Annotated[float, "RMSE Score"]
]:
    try:
        prediction = model.predict(X_test)
        mse_evaluator = MSE()
        mse = mse_evaluator.evaluate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_evaluator = R2()
        r2 = r2_evaluator.evaluate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2)

        rmse_evaluator = RMSE()
        rmse = rmse_evaluator.evaluate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e