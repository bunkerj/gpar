import numpy as np
from math import sin, cos, pi
from matplotlib import pyplot as plt

NOISE_VAR = 0.05


def eps():
    return np.random.normal(0, NOISE_VAR)


def y1(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4 + eps()


def y2(x):
    return cos(y1(x)) ** 2 + sin(3 * x) + eps()


def y3(x):
    return y2(x) * (y1(x) ** 2) + 3 * x + eps()


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
    plt.plot(X_new, mean)
    plt.scatter(X_old, Y_old, color='b', marker='x')
    plt.fill_between(
        X_new.flatten(),
        lb.flatten(),
        ub.flatten(),
        alpha=0.2,
        edgecolor='b')
    plt.grid(True)
