import numpy as np


def should_update_max(max_value, current_value):
    return max_value is None or current_value > max_value


def create_synthetic_output(func, X):
    return np.array(list(map(func, X))).reshape((len(X), 1))


def slice_column(matrix, col_id):
    return matrix[:, col_id].reshape((matrix.shape[0], 1))


def concat_right_column(matrix, col):
    if matrix is None:
        return np.array(col).reshape((len(col), 1))
    else:
        return np.concatenate((matrix, col), axis=1)


def map_to_col_vector(f, X):
    return np.array(list(map(f, X))).reshape((X.shape[0], 1))


def map_and_stack_outputs(funcs, X):
    Y = None
    for f in funcs:
        single_Y = map_to_col_vector(f, X)
        Y = single_Y if Y is None else concat_right_column(Y, single_Y)
    return Y
