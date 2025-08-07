from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(file_path: str):
    data = ingest_df(file_path)
    clean_df(data)
    train_model(data)
    evaluate_model(data)