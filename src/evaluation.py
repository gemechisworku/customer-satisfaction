import logging
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import numpy as np

class Evaluation(ABC):
    """Abstract base class for evaluation strategies."""
    
    @abstractmethod
    def evaluate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate the model with the provided test data."""
        pass


class MSE(Evaluation):
    """Mean Squared Error evaluation strategy."""

    def evaluate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate the model using Mean Squared Error.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The Mean Squared Error (MSE) score."""
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """R-squared evaluation strategy."""

    def evaluate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate the model using R-squared.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: The R2 score."""

        try:
            logging.info("Calculating R2 Score...")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2 Score: {e}")
            raise e
class RMSE(Evaluation):
    """Root Mean Squared Error evaluation strategy."""

    def evaluate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate the model using Root Mean Squared Error.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        Returns:
            float: The Root Mean Squared Error (RMSE) score."""
        
        try:
            logging.info("Calculating RMSE...")
            rmse = root_mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE: {e}")
            raise e