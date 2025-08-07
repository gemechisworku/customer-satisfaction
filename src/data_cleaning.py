import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract base class for data handling strategies.
    This class defines the interface for data processing steps."""

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreProcessingStrategy(DataStrategy):
    """Concrete class for data pre-processing strategy."""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pre-process the data by handling missing values and encoding categorical variables.
        Args:
            data (pd.DataFrame): The input data to be pre-processed.    
        Returns:
            pd.DataFrame: The pre-processed data.
        """
        try:
            # Drop unnecessary columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ], 
            axis=1)

            # drop non-numeric columns for simplicity
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            # Fill missing values with the median of each column
            # List of columns to fill with median
            columns_to_fill = [
                'geolocation_zip_code_prefix',
                'geolocation_lat',
                'geolocation_lng',
                'order_item_id',
                'price',
                'freight_value',
                'product_name_lenght',
                'product_description_lenght',
                'product_photos_qty',
                'product_weight_g',
                'product_length_cm',
                'product_height_cm',
                'product_width_cm',
                'seller_zip_code_prefix',
                'payment_sequential',
                'payment_installments',
                'payment_value',
                'review_score'
            ]

            # Fill each column with its median
            for col in columns_to_fill:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())

            return data
        except Exception as e:
            logging.error("Error in data pre-processing: {}".format(e))
            raise e 

class DataDivideStrategy(DataStrategy):
    """Concrete class for data division strategy."""
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        """Divide the data into training and testing sets.
        Args:
            data (pd.DataFrame): The input data to be divided.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The training and testing data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
        
class DataCleaning:
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        """Initialize the DataCleaning class with a data strategy.
        Args:
            data (pd.DataFrame): The input data to be cleaned.
            strategy (DataStrategy): The strategy to be used for data handling.
        """
        self.data = data
        self.strategy = strategy
    def handle_data(self) -> Union[pd.DataFrame, pd.DataFrame]:
        """Handle the data using the specified strategy.
        Returns:
            Union[pd.DataFrame, pd.DataFrame]: The processed data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in data cleaning: {}".format(e))
            raise e