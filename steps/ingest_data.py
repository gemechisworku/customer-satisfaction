import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, file_path: str):
        """Initialize the IngestData class.
        Args:            
            file_path (str): Path to the CSV file to be ingested.
        """
        self.file_path = file_path
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file.
        Returns:
            pd.DataFrame: Data loaded from the CSV file.
        """
        self.logger.info(f"Ingesting the data from {self.file_path}")
        return pd.read_csv(self.file_path)


@step
def ingest_df(file_path: str) -> pd.DataFrame:
    """Step to ingest data from a CSV file.
    Args:
        file_path (str): Path to the CSV file to be ingested.
    """
    try:
        ingester = IngestData(file_path)
        return ingester.load_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e