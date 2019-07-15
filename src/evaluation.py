import numpy as np


def validate_lengths(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception('Must be the same lengths')


def smse(series1, series2):
    validate_lengths(series1, series2)
    n = len(series1)
    error_acc = 0
    for idx in range(n):
        e = series1[idx] - series2[idx]
        error_acc += np.sign(e) * (e ** 2)
    return float(error_acc / n)


def mse(x, y):
    validate_lengths(x, y)
    return np.mean((x - y) ** 2)
