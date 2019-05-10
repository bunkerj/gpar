import numpy as np


def smse(series1, series2):
    if len(series1) != len(series2):
        raise Exception('Cannot align time series')
    n = len(series1)
    error_acc = 0
    for idx in range(n):
        e = series1[idx] - series2[idx]
        error_acc += np.sign(e) * (e ** 2)
    return error_acc / n
