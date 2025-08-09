from typing import Optional
import numpy as np
import pandas as pd
import logging as logger
# from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow import MLFLOW_MODEL_DEPLOYER_FLAVOR
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

from pydantic import BaseModel

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW_MODEL_DEPLOYER_FLAVOR])

class DeploymentTriggerConfig(BaseModel):
    """Configuration for the deployment trigger."""
    min_accuracy: float = 0

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy

@step
def custom_mlflow_model_deployer(
    model: object,
    deploy_decision: bool,
    workers: int,
    timeout: int,
) -> Optional[MLFlowDeploymentService]:
    if not deploy_decision:
        logger.info("Skipping model deployment based on evaluation result.")
        return None

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    try:
        # ✅ Cast the base deployer to MLflow-specific deployer
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # ✅ Correct call to MLflowModelDeployer.deploy_model()
        service = model_deployer.deploy_model(
            model=model,
            model_name="model",
            service_name="zenml-model",
            workers=workers,
            timeout=timeout,
            blocking=True,         # This avoids daemon usage
            silent_daemon=True     # Avoids trying to use unsupported features
        )
        logger.info(f"Model deployed successfully at: {service.prediction_url}")
        return service

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    file_path: str = "data/all_olist_customers_dataset.csv",
):
    df = ingest_df(file_path=file_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    r2_score, rmse_score = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    deployment_decision = deployment_trigger(accuracy=r2_score, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))
    custom_mlflow_model_deployer(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout  
    )