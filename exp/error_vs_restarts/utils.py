import numpy as np
from src_utils import slice_column
from evaluation import smse
from regression.gpar_regression import GPARRegression
from matplotlib import pyplot as plt


def get_total_smse(Y_true, model_means):
    total_smse = 0
    for out_id in range(2):
        single_gpar_means = slice_column(model_means, out_id)
        true_means = slice_column(Y_true, out_id)
        total_smse += smse(true_means, single_gpar_means)
    return total_smse


def get_total_mse_values_and_ordering_index(X_obs, Y_obs, X_new, Y_true,
                                            kernel_function, num_restarts_values):
    """
    Returns:
    - the sum of the MSE of all outputs index by the number of restarts
    - the index (num of restarts) where the correct order was retrieved
    """
    n_restarts = len(num_restarts_values)
    total_mse_values = np.zeros((n_restarts, 1))
    correct_order_index = None
    for idx in range(n_restarts):
        num_restart = int(num_restarts_values[idx])
        gpar_model = GPARRegression(X_obs, Y_obs,
                                    kernel_function, num_restarts=num_restart)
        ordering = gpar_model.get_ordering()
        means, variances = gpar_model.predict(X_new)
        total_mse_values[idx] = get_total_smse(Y_true, means)
        if correct_order_index is None and ordering == (1, 2, 3):
            correct_order_index = num_restart
    return total_mse_values, correct_order_index


def plot_log_likelihood_vs_restarts(total_mse_values, correct_order_index,
                                    num_restarts_values, title=None):
    plt.plot(num_restarts_values, total_mse_values)
    if title is not None:
        plt.title('')
    if correct_order_index is not None:
        plt.axvline(correct_order_index, color='r')
    plt.ylabel('Total MSE for all outputs')
    plt.xlabel('Number of Restarts')