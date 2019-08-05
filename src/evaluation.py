import numpy as np


def validate_lengths(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception('Must be the same lengths')


def mse(truth, prediction):
    validate_lengths(truth, prediction)
    n = len(truth)
    error_acc = sum((truth[idx] - prediction[idx]) ** 2 for idx in range(n))
    return float(error_acc / n)


def smse(truth, prediction):
    mse_value = mse(truth, prediction)
    return float(mse_value / np.var(truth))
