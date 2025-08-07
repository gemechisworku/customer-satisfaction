import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessingStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple [
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"]
]:
    """Step to clean and preprocess the data.
    
    Args:
        data (pd.DataFrame): The input data to be cleaned.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Cleaned training and testing data.
    """
    try:
        preprocess_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_division = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_division.handle_data()
        logging.info("Data cleaning and preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e