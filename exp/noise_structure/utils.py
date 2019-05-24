import GPy
from matplotlib import pyplot as plt
from src_utils import slice_column, concat_right_column, stack_all_columns


def plot_noise(figure_id_start, X, noise, title):
    plt.figure(figure_id_start)
    n_plots = noise.shape[1] // 2
    plt.suptitle(title)
    for idx in range(n_plots):
        plt.subplot(1, 3, idx + 1)
        plt.scatter(slice_column(noise, 2 * idx),
                    slice_column(noise, 2 * idx + 1), s=1, c=X, cmap='magma')


def get_single_igp_prediction(X_obs, Y_obs, X_new, out_id, kernel_function, num_restarts):
    single_Y = slice_column(Y_obs, out_id)
    kernel = kernel_function(X_obs, X_obs)
    m = GPy.models.GPRegression(X_obs, single_Y, kernel)
    m.optimize_restarts(num_restarts, verbose=False)
    return m.predict(X_new)


def get_igp_predictions(X_obs, Y_obs, X_new, kernel_function, num_restarts):
    stacked_means = None
    stacked_vars = None
    n = Y_obs.shape[1]
    for out_id in range(n):
        means, variances = get_single_igp_prediction(X_obs, Y_obs, X_new, out_id,
                                                     kernel_function, num_restarts)
        stacked_means = concat_right_column(stacked_means, means)
        stacked_vars = concat_right_column(stacked_vars, variances)
    return stacked_means, stacked_vars


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


def plot_noise_histogram(nrows, ncols, idx, Y1, Y2, title):
    plt.subplot(nrows, ncols, idx + 1)
    plt.hist2d(Y1, Y2, (75, 75), cmap=plt.cm.jet)
    plt.title(title)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
