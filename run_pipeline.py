from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Define the path to the CSV file
    file_path = "data/all_olist_customers_dataset.csv"

    # Run the training pipeline with the specified file path
    training_pipeline(file_path=file_path)

