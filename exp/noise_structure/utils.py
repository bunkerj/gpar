import GPy
from matplotlib import pyplot as plt
from src_utils import slice_column, concat_right_column


def plot_noise(figure_id_start, X, noise):
    plt.figure(figure_id_start)
    n_plots = noise.shape[1] - 1
    for idx in range(1, n_plots + 1):
        plt.subplot(1, 3, idx)
        plt.scatter(slice_column(noise, 0),
                    slice_column(noise, idx), s=1, c=X, cmap='magma')


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
