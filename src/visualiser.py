import GPy
import numpy as np
from evaluation import mse
from matplotlib import pyplot as plt
from utils import slice_column, concat_right_column
from kernels import get_linear_input_dependent_kernel

NUM_SUBPLOTS = 3


class Visualiser:
    def __init__(self, gpar_model, X_obs, Y_obs, X_new, Y_true):
        self.X_obs = X_obs
        self.Y_obs = Y_obs
        self.X_new = X_new
        self.Y_true = Y_true
        self.output_dim = Y_obs.shape[1]
        self.num_restarts = 10

        means, variances = gpar_model.predict(X_new)
        self.gpar_means = means
        self.gpar_vars = variances

        means, variances = self.get_igp_predictions()
        self.igp_means = means
        self.igp_vars = variances

    def get_igp_predictions(self):
        stacked_means = None
        stacked_vars = None
        for out_id in range(self.output_dim):
            means, variances = self._get_single_igp_prediction(out_id)
            stacked_means = concat_right_column(stacked_means, means)
            stacked_vars = concat_right_column(stacked_vars, variances)
        return stacked_means, stacked_vars

    def _get_single_igp_prediction(self, out_id):
        single_Y = slice_column(self.Y_obs, out_id)
        kernel = get_linear_input_dependent_kernel(self.X_obs, self.X_obs)
        m = GPy.models.GPRegression(self.X_obs, single_Y, kernel)
        m.optimize_restarts(self.num_restarts, verbose=False)
        return m.predict(self.X_new)

    def print_mse_values(self):
        for out_id in range(self.output_dim):
            single_gpar_means = slice_column(self.gpar_means, out_id)
            single_igp_means = slice_column(self.igp_means, out_id)
            true_means = slice_column(self.Y_true, out_id)
            print('\nGPAR MSE for output {}: {}'.format(
                out_id + 1, mse(true_means, single_gpar_means)))
            print('Single GP MSE for output {}: {}'.format(
                out_id + 1, mse(true_means, single_igp_means)))

    def specify_subplot(self, out_id):
        return plt.subplot(1, NUM_SUBPLOTS, (out_id % NUM_SUBPLOTS) + 1)

    def _plot_observations(self, out_id):
        single_Y = slice_column(self.Y_obs, out_id)
        plt.scatter(self.X_obs, single_Y, color='b', marker='x', label='Observations')

    def _plot_single_output(self, out_id, stacked_means, stacked_vars, label, display_var=False):
        """Construct plot containing the predictions and observations."""
        means = slice_column(stacked_means, out_id)
        plt.plot(self.X_new, means, label=label)
        if display_var:
            variances = slice_column(stacked_vars, out_id)
            ub = means + 2 * np.sqrt(variances)
            lb = means - 2 * np.sqrt(variances)
            plt.fill_between(
                self.X_new.flatten(),
                lb.flatten(),
                ub.flatten(),
                alpha=0.2,
                edgecolor='b')

    def _plot_truth(self, out_id):
        single_Y = slice_column(self.Y_true, out_id)
        plt.plot(self.X_new, single_Y, label='Truth')

    def plot_all_outputs(self):
        """Plot all GPAR outputs against: observations, igp, truth."""
        for out_id in range(self.output_dim):
            plt.figure(out_id // NUM_SUBPLOTS)
            self.specify_subplot(out_id)
            self._plot_observations(out_id)
            self._plot_single_output(out_id, self.gpar_means, self.gpar_vars, 'GPAR', True)
            self._plot_single_output(out_id, self.igp_means, self.igp_vars, 'IGP', False)
            self._plot_truth(out_id)
            if (out_id + 1) % NUM_SUBPLOTS == 0:
                plt.legend(loc='upper left')
            plt.title('Y{}'.format(out_id + 1))
            plt.grid(True)
