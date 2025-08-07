from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """Abstract base class for machine learning models."""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model with the provided data."""
        pass

class LinearRegressionModel(Model):
    """Concrete class for linear regression model."""
    
    def train(self, X_train, y_train, **kwargs):
        """Train the linear regression model.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            **kwargs: Additional keyword arguments for the model.
        Returns:
            LinearRegression: The trained linear regression model.
        """
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model trained successfully.")
            return model
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e