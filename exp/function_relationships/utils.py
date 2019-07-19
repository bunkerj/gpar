import numpy as np
from evaluation import mse
from matplotlib import pyplot as plt
from src_utils import slice_column


def get_num_subplots(plot_shape):
    return plot_shape[0] * plot_shape[1]


def specify_plot_location(out_id, figure_id_start, plot_shape):
    num_subplots = get_num_subplots(plot_shape)
    plt.figure(figure_id_start + (out_id // num_subplots))
    plt.subplot(plot_shape[0], plot_shape[1], (out_id % num_subplots) + 1)


def get_visible_index_bool(n, percent_visible):
    return np.random.rand(n) < (percent_visible / 100)


def plot_bar_plot(values, labels):
    plt.bar(range(len(values)), values, tick_label=labels)


def plot_observations(X_obs, Y_obs, out_id):
    single_Y = slice_column(Y_obs, out_id)
    plt.scatter(X_obs, single_Y, color='b', marker='x', label='Observations')


def plot_single_output(X, stacked_means, stacked_vars, out_id, label, display_var=False):
    """Construct plot containing the predictions and observations."""
    means = slice_column(stacked_means, out_id)
    plt.plot(X, means, label=label)
    if display_var:
        variances = slice_column(stacked_vars, out_id)
        ub = means + 2 * np.sqrt(variances)
        lb = means - 2 * np.sqrt(variances)
        plt.fill_between(
            X.flatten(),
            lb.flatten(),
            ub.flatten(),
            alpha=0.2,
            edgecolor='b')


def plot_truth(X_new, Y_true, out_id):
    single_Y = slice_column(Y_true, out_id)
    plt.plot(X_new, single_Y, label='Truth')


def initialize_labels(n, initial_labels):
    if initial_labels is None:
        return ['Y{}'.format(i + 1) for i in range(n)]
    else:
        return initial_labels


def plot_all_outputs(model_means, model_vars, igp_means, igp_vars,
                     X_new, Y_true, X_obs, Y_obs,
                     figure_id_start=0, initial_labels=None, plot_shape=(1, 3)):
    """Plot all GPAR outputs against: observations, igp, truth."""
    labels = initialize_labels(Y_true.shape[1], initial_labels)
    for out_id, label in enumerate(labels):
        specify_plot_location(out_id, figure_id_start, plot_shape)
        plot_observations(X_obs, Y_obs, out_id)
        plot_single_output(X_new, model_means, model_vars, out_id, 'GPAR', True)
        plot_single_output(X_new, igp_means, igp_vars, out_id, 'IGP', False)
        plot_truth(X_new, Y_true, out_id)
        if (out_id + 1) % get_num_subplots(plot_shape) == 0:
            plt.legend(loc='upper left')
        plt.title('{}'.format(label))
        plt.grid(True)


def plot_mse_values(model_means, igp_means, Y_true,
                    figure_id_start=0, initial_labels=None, plot_shape=(1, 3)):
    labels = initialize_labels(Y_true.shape[1], initial_labels)
    for out_id, label in enumerate(labels):
        specify_plot_location(out_id, figure_id_start, plot_shape)
        single_gpar_means = slice_column(model_means, out_id)
        single_igp_means = slice_column(igp_means, out_id)
        true_means = slice_column(Y_true, out_id)
        gpar_mse = mse(true_means, single_gpar_means)
        igp_mse = mse(true_means, single_igp_means)
        plot_bar_plot([gpar_mse, igp_mse], ['GPAR', 'IGP'])
        plt.title('{} MSE'.format(label))
