from kfp.v2.dsl import (
    component
)

@component(
    base_image="python:3.9",
    packages_to_install=["kserve~=0.10.1"],
)
def deploy_model(model_name: str, storage_uri: str):
    """
    Deploy the model as a inference service to KServe
    Args:
        model_name: the name of the deployed inference service
        storage_uri: the URI of the saved model in MLflow's artifact store
    """
    from kubernetes import client
    from kserve import KServeClient
    from kserve import constants
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    model_uri = storage_uri
    logger.info("MODEL URI:", model_uri)
    namespace = "kserve-inference"
    kserve_version="v1beta1"
    api_version = constants.KSERVE_GROUP + '/' + kserve_version
    
    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=model_name,
            namespace=namespace,
            annotations={'sidecar.istio.io/inject':'false'}
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name='kserve-sa',
                sklearn=V1beta1SKLearnSpec(
                    storage_uri=model_uri
                )
            )
        )
    )
    kserve = KServeClient()
    try:
        kserve.create(inferenceservice=isvc)
    except RuntimeError:
        kserve.patch(name=model_name, inferenceservice=isvc, namespace=namespace)
    