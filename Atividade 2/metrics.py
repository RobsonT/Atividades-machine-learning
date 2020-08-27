import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy score for the data passed

    Args:
        y_true (numpy array): the real data
        y_pred (numpy array): the predicted data

    Returns:
        float: the accuracy score for the data passed
    """
    sample_length = y_true.shape[0]
    return (np.sum(y_true == y_pred) / sample_length)