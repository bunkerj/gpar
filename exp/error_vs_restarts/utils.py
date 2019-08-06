import os
import pickle
import numpy as np
from src_utils import slice_column
from evaluation import smse
from constants import NUM_RESTARTS_VALUES_PATH, OUTPUTS_PATH
from regression.gpar_regression import GPARRegression
from matplotlib import pyplot as plt


def get_total_smse(Y_true, model_means):
    n = Y_true.shape[1]
    total_smse = 0
    for out_id in range(n):
        single_gpar_means = slice_column(model_means, out_id)
        true_means = slice_column(Y_true, out_id)
        total_smse += smse(true_means, single_gpar_means)
    return total_smse


def get_total_smse_values_and_ordering_index(X_obs, Y_obs, X_new, Y_true,
                                             kernel_function,
                                             num_restarts_values,
                                             num_avg_samples):
    """
    Returns:
    - the sum of the MSE of all outputs index by the number of restarts
    - the index (num of restarts) where the correct order was retrieved
    """
    n_restarts = len(num_restarts_values)
    total_mse_values = np.zeros((n_restarts, 1))
    for idx in range(n_restarts):
        for _ in range(num_avg_samples):
            num_restart = int(num_restarts_values[idx])
            gpar_model = GPARRegression(X_obs, Y_obs,
                                        kernel_function,
                                        num_restarts=num_restart)
            means, variances = gpar_model.predict(X_new)
            total_mse_values[idx] = get_total_smse(Y_true, means) / num_avg_samples
    return total_mse_values


def extract_name(filename):
    filename_no_ext = filename.split('.')[0]
    characteristics = filename_no_ext.split('_')[1:]
    return '{}/{} '.format(*characteristics)


def plot_error_vs_restarts(n_rows, n_cols):
    with open(NUM_RESTARTS_VALUES_PATH, 'rb') as file:
        num_restarts_values = pickle.load(file)

    files = sorted(os.listdir(OUTPUTS_PATH))

    for idx, filename in enumerate(files):
        plt.subplot(n_rows, n_cols, idx + 1)
        name = extract_name(filename)
        path = os.path.join(OUTPUTS_PATH, filename)
        with open(path, 'rb') as file:
            total_mse_values = pickle.load(file)
        plt.plot(num_restarts_values, total_mse_values)
        plt.title(name)
        plt.ylabel('Total SMSE')
        plt.xlabel('Number of Restarts')
        plt.grid()
