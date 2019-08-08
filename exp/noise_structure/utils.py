import numpy as np
from matplotlib import pyplot as plt
from src.regression.igp_regression import IGPRegression
from src.src_utils import slice_column, concat_right_column, stack_all_columns


def plot_noise(figure_id_start, X, noise, title):
    plt.figure(figure_id_start)
    n_plots = noise.shape[1] // 2
    plt.suptitle(title)
    for idx in range(n_plots):
        plt.subplot(1, 3, idx + 1)
        plt.scatter(slice_column(noise, 2 * idx),
                    slice_column(noise, 2 * idx + 1), s=1, c=X, cmap='magma')


def get_prediction_noise(Y_new, model_predictor, split_observations):
    prediction_noise = None
    for idx, Y in enumerate(split_observations):
        means, variances = model_predictor(Y)
        prediction_noise = concat_right_column(prediction_noise,
                                               Y_new[:, 0] - means[:, 0])
        prediction_noise = concat_right_column(prediction_noise,
                                               Y_new[:, idx + 1] - means[:, 1])
    return prediction_noise


def get_split_outputs(Y):
    """Split Y into Y1/Y2 for every output scheme."""
    n = Y.shape[1] - 1
    split_observations = []
    for scheme_idx in range(1, n + 1):
        relevant_cols = [Y[:, 0], Y[:, scheme_idx]]
        split_observations.append(stack_all_columns(relevant_cols))
    return split_observations


def plot_noise_histogram(nrows, ncols, idx, Y1, Y2, title=None):
    plt.subplot(nrows, ncols, idx + 1)
    plt.title(title)
    plt.hist2d(Y1, Y2, (75, 75), cmap=plt.cm.jet)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)


def get_igp_output_samples(X_obs, Y_obs, x_value, kernel_function, num_restarts, out_id, n):
    igp_model = IGPRegression(X_obs, Y_obs, kernel_function, num_restarts=num_restarts)
    igp_means, igp_vars = igp_model.single_predict(x_value, out_id)
    return np.random.normal(float(igp_means), np.sqrt(float(igp_vars)), n)


def get_gpar_output_samples(igp_means, igp_vars):
    return np.random.normal(igp_means, np.sqrt(igp_vars)).flatten()
