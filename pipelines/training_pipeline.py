from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(file_path: str):
    df = ingest_df(file_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse_score = evaluate_model(model, X_test, y_test)
    