import numpy as np


def mse(x, y):
    return np.sum((x - y) ** 2)


def concat_right_column(matrix, col):
    if matrix is None:
        return np.array(col).reshape((len(col), 1))
    else:
        if len(col.shape) == 1:
            col = col.reshape((len(col), 1))
        return np.concatenate((matrix, col), axis=1)


def stack_all_columns(columns):
    full_stack = None
    for column in columns:
        full_stack = concat_right_column(full_stack, column)
    return full_stack
