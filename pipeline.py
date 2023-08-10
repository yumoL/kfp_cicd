import warnings
warnings.filterwarnings("ignore")

import kfp.dsl as dsl
from kfp.aws import use_aws_secret

from components.pull_data import pull_data
from components.preprocess import preprocess
from components.train import train
from components.deploy import deploy_model

@dsl.pipeline(
    name="redwine-pipeline",
    description="An example pipeline that deploys a redwine model"
)
def pipeline(
    url: str,
    target: str,
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint_url: str,
    model_name: str,
    alpha: float,
    l1_ratio: float
):
    """
    Define a pipeline
    Args:
        url: Url for downloading the dataset
        target: Target column name of the dataset
        mlflow_experiment_name: Name of the MLflow experiment
        mlflow_tracking_uri: URI of MLflow's tracking server
        mlflow_s3_endpoint_url: URL of MLflow's artifact store
        model_name: The file name of the saved model. It is also the name of the KServe inference service. 
        alpha, l1_ratio: Hyperparameters that need to be configured
    """
    pull_task = pull_data(url)
    
    # The preprocess component uses the output data of the pull_data component, 
    # e.g., the downloaded dataset as its input
    preprocess_task = preprocess(data=pull_task.outputs["data"])
    
    train_task = train(
        train_set=preprocess_task.outputs["train_set"],
        test_set=preprocess_task.outputs["test_set"],
        target=target,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint_url=mlflow_s3_endpoint_url,
        model_name=model_name,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
    
    # The train component uploads the trained model to the MLflow service
    # so it needs to access the required credentials of the MLflow service's MinIO storage service.
    # These credentials (username and password) have been deployed as a secret named "aws-secret" to the 
    # Kubernetes cluster.
    train_task.apply(use_aws_secret(secret_name="aws-secret"))
    

    deploy_model_task = deploy_model(
        model_name = model_name,
        storage_uri=train_task.outputs["storage_uri"]
    )
