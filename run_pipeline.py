import kfp

from pipeline import pipeline

# Specify pipeline argument values
arguments = {
    "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "target": "quality",
    "mlflow_tracking_uri": "http://mlflow.mlflow.svc.cluster.local:5000",
    "mlflow_s3_endpoint_url": "http://mlflow-minio-service.mlflow.svc.cluster.local:9000",
    "mlflow_experiment_name": "demo-notebook",
    "model_name": "wine-quality",
    "alpha": 0.5,
    "l1_ratio": 0.5,
}

run_name = "demo-run"
experiment_name = "demo-experiment"

client = kfp.Client()

client.create_run_from_pipeline_func(
    pipeline_func=pipeline,
    run_name=run_name,
    experiment_name=experiment_name,
    arguments=arguments,
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
    enable_caching=True
)