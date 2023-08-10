from typing import NamedTuple
from kfp.v2.dsl import (
    component,
    Input,
    Dataset,
)

@component(
    base_image="python:3.10",
    packages_to_install=["pandas~=1.4.2", "numpy", "scikit-learn~=1.0.2", "mlflow~=2.3.2", "boto3~=1.21.0"]
)
def train(
    train_set: Input[Dataset],
    test_set: Input[Dataset],
    mlflow_experiment_name: str,
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint_url: str,
    model_name: str,
    alpha: float,
    l1_ratio: float,
    target: str = "quality"
) -> NamedTuple("Output", [("storage_uri", str), ("run_id", str),]):
    """
    Train a model, save the training metadata and the resulted model artifact to MLflow
    Args:
        train_set: File where the training dataset is saved
        test_set: File where the test dataset is saved
        mlflow_experiment_name: Name of the MLflow experiment
        mlflow_tracking_uri: URI of MLflow's tracking server
        mlflow_s3_endpoint_url: URL of MLflow's artifact store
        model_name: The file name of the saved model
        alpha, l1_ratio: Hyperparameters that need to be configured
        target: Target column name
    
    Returns: 
        namedtuple("Output", ["storage_uri", "run_id"]) where storage_uri is the URI of the saved model artifact 
        in the Mlflow's artifact store and run_id the ID of the MLflow run that produces the model
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import mlflow
    import mlflow.sklearn
    import os
    import logging
    import pickle
    from collections import namedtuple
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint_url
    
    # Load data
    train = pd.read_csv(train_set.path)
    test = pd.read_csv(test_set.path)
    
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop([target], axis=1)
    test_x = test.drop([target], axis=1)
    train_y = train[[target]]
    test_y = test[[target]]
    
    logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    logger.info(f"Using MLflow experiment: {mlflow_experiment_name}")
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        
        logger.info("Fitting model...")
        model.fit(train_x, train_y)
        
        logger.info("Predicting...")
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        logger.info("RMSE: %s" % rmse)
        logger.info("MAE: %s" % mae)
        logger.info("R2: %s" % r2)
        
        logger.info("Logging parameters and metrics to MLflow")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        logger.info("Logging trained model")
        mlflow.sklearn.log_model(
            model,
            model_name,
            registered_model_name="ElasticnetWineModel",
            serialization_format="pickle"
        )
        
        logger.info("Logging predictions artifact to MLflow")
        np.save("predictions", predicted_qualities)
        mlflow.log_artifact(
            local_path="predictions.npy",
            artifact_path="predicted_qualities"
        )
        
        # Prepare output
        output = namedtuple("Output", ["storage_uri", "run_id"])
        return output(mlflow.get_artifact_uri(artifact_path=model_name), run_id)
   