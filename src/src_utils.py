import GPy
import numpy as np

from kernels import get_linear_input_dependent_kernel


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
        if len(col.shape) == 1:
            col = col.reshape((len(col), 1))
        return np.concatenate((matrix, col), axis=1)


def map_to_col_vector(f, X):
    return np.array(list(map(f, X))).reshape((X.shape[0], 1))


def stack_all_columns(columns):
    full_stack = None
    for column in columns:
        full_stack = concat_right_column(full_stack, column)
    return full_stack


def map_and_stack_outputs(funcs, X):
    Y = None
    for f in funcs:
        single_Y = map_to_col_vector(f, X)
        Y = single_Y if Y is None else concat_right_column(Y, single_Y)
    return Y


def get_igp_predictions(X_obs, Y_obs, X_new, kernel_function, num_restarts):
    stacked_means = None
    stacked_vars = None
    for out_id in range(Y_obs.shape[1]):
        means, variances = \
            get_single_igp_prediction(X_obs, Y_obs, X_new,
                                      kernel_function, num_restarts, out_id)
        stacked_means = concat_right_column(stacked_means, means)
        stacked_vars = concat_right_column(stacked_vars, variances)
    return stacked_means, stacked_vars


def get_single_igp_prediction(X_obs, Y_obs, X_new, kernel_function, num_restarts, out_id=0):
    single_Y = slice_column(Y_obs, out_id)
    kernel = kernel_function(X_obs, X_obs)
    m = GPy.models.GPRegression(X_obs, single_Y, kernel)
    m.optimize_restarts(num_restarts, verbose=False)
    return m.predict(X_new)
