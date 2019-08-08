import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from src.src_utils import slice_column


def plot_bq_integral_gp_dist(integral_base, integral_bq, integral_std_bq, label):
    ub = np.max((integral_base + 0.25, integral_bq + 2 * integral_std_bq))
    lb = np.min((integral_base - 0.25, integral_bq - 2 * integral_std_bq))
    x_gauss = np.linspace(lb, ub, 1000).flatten()
    y_gauss = stats.norm.pdf(x_gauss, integral_bq, integral_std_bq).flatten()
    plt.plot(x_gauss, y_gauss, label=label)


def plot_bq_integrand_truth(custom_func, start, end):
    n_new = 1000
    X_new = np.linspace(start, end, n_new).reshape((n_new, 1))
    y_single_means_true = custom_func(X_new)
    plt.plot(X_new, y_single_means_true, label='True Function')


def slice_if_needed(data, out_idx):
    return slice_column(data, out_idx) if data.shape[1] > 1 else data


def plot_bq_integrand_gp(m, start, end, label, out_idx, display_var=False):
    n_new = 1000
    X_new = np.linspace(start, end, n_new).reshape((n_new, 1))
    means_pred, vars_pred = m.predict(X_new)

    y_single_means_pred = slice_if_needed(means_pred, out_idx)
    y_single_vars_pred = slice_if_needed(vars_pred, out_idx)

    plt.plot(X_new, y_single_means_pred, label=label)

    if display_var:
        ub_means_pred = y_single_means_pred + 2 * np.sqrt(y_single_vars_pred)
        lb_means_pred = y_single_means_pred - 2 * np.sqrt(y_single_vars_pred)
        plt.fill_between(
            X_new.flatten(),
            lb_means_pred.flatten(),
            ub_means_pred.flatten(),
            alpha=0.2,
            edgecolor='b')
