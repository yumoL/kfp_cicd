from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
)

@component(
    base_image="python:3.10",
    packages_to_install=["pandas~=1.4.2", "scikit-learn~=1.0.2"],
)
def preprocess(
    data: Input[Dataset],
    train_set: Output[Dataset],
    test_set: Output[Dataset],
    target: str = "quality"
):
    """
    Read a dataset from a file, split it into a training and test dataset, and save the training and test dataset
    into separate files
    Args:
        data: File where the dataset is read from
        train_set: File where the training dataset is saved
        test_set: File where the test dataset is saved
        target: Target column name  
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data = pd.read_csv(data.path)
    
    # Split the data into training and test sets. (0.75, 0.25) split
    train, test = train_test_split(data)
    
    # Save the training and test datasets into separate files
    train.to_csv(train_set.path, index=None)
    test.to_csv(test_set.path, index=None)
    