import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple

@step
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

        r2_evaluator = R2()
        r2 = r2_evaluator.evaluate_scores(y_test, prediction)

        rmse_evaluator = RMSE()
        rmse = rmse_evaluator.evaluate_scores(y_test, prediction)

        return r2, rmse
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e