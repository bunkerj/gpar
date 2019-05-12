import numpy as np
from matplotlib import pyplot as plt


def should_update_max(max_value, current_value):
    return max_value is None or current_value > max_value


def create_synthetic_output(func, X):
    return np.array(list(map(func, X))).reshape((len(X), 1))


def slice_column(matrix, col_id):
    return matrix[:, col_id].reshape((matrix.shape[0], 1))


def plot_single_output(figure_id, X_old, Y_old, X_new, mean, var):
    """Construct plot containing the predictions and observations."""
    ub = mean + 2 * np.sqrt(var)
    lb = mean - 2 * np.sqrt(var)
    plt.figure(figure_id)
    plt.plot(X_new, mean, label='Prediction')
    plt.scatter(X_old, Y_old, color='b', marker='x', label='Observations')
    plt.fill_between(
        X_new.flatten(),
        lb.flatten(),
        ub.flatten(),
        alpha=0.2,
        edgecolor='b')
    plt.grid(True)


def concat_right_column(matrix, col):
    return np.concatenate((matrix, col), axis=1)
