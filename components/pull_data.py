from kfp.v2.dsl import (
    component,
    Output,
    Dataset,
)

@component(
    base_image="python:3.10",
    packages_to_install=["pandas~=1.4.2"],
)
def pull_data(url: str, data: Output[Dataset]):
    """
    Download a dataset and save it to a file
    Args:
        url: Dataset URL
        data: File where the downloaded dataset is saved
    """
    import pandas as pd
    df = pd.read_csv(url, sep=";")
    df.to_csv(data.path, index=None)