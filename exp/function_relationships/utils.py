import numpy as np
from src.evaluation import smse
from matplotlib import pyplot as plt
from src.src_utils import slice_column

NUM_SUBPLOTS = 3


def specify_subplot(out_id):
    return plt.subplot(1, NUM_SUBPLOTS, (out_id % NUM_SUBPLOTS) + 1)


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


def initialize_labels(gpar_ordering, initial_labels):
    if initial_labels is None:
        return ['Y{}'.format(idx + 1) for idx in gpar_ordering]
    else:
        return initial_labels


def configure_plots(idx, figure_id_start):
    plt.figure(figure_id_start + (idx // NUM_SUBPLOTS))
    specify_subplot(idx)


def plot_all_outputs(model_means, model_vars, igp_means, igp_vars,
                     gpar_ordering, X_new, Y_true, X_obs, Y_obs,
                     figure_id_start=0, initial_labels=None):
    """Plot all GPAR outputs against: observations, igp, truth."""
    labels = initialize_labels(gpar_ordering, initial_labels)
    for idx, label in enumerate(labels):
        configure_plots(idx, figure_id_start)
        out_id = gpar_ordering[idx]
        plot_observations(X_obs, Y_obs, out_id)
        plot_single_output(X_new, model_means, model_vars, out_id, 'GPAR', True)
        plot_single_output(X_new, igp_means, igp_vars, out_id, 'IGP', False)
        plot_truth(X_new, Y_true, out_id)
        if (idx + 1) % NUM_SUBPLOTS == 0:
            plt.legend(loc='upper right')
        plt.title('{}'.format(label))
        plt.grid(True)


def plot_mse_values(model_means, igp_means, Y_true, gpar_ordering,
                    figure_id_start=0, initial_labels=None):
    labels = initialize_labels(gpar_ordering, initial_labels)
    for idx, label in enumerate(labels):
        configure_plots(idx, figure_id_start)
        out_id = gpar_ordering[idx]
        single_gpar_means = slice_column(model_means, out_id)
        single_igp_means = slice_column(igp_means, out_id)
        true_means = slice_column(Y_true, out_id)
        gpar_smse = smse(true_means, single_gpar_means)
        igp_smse = smse(true_means, single_igp_means)
        plot_bar_plot([gpar_smse, igp_smse], ['GPAR', 'IGP'])
        plt.title('{} SMSE'.format(label))
