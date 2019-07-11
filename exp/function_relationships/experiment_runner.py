from matplotlib import pyplot as plt
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from utils import plot_mse_values, plot_all_outputs


class ExperimentRunner:
    def __init__(self, X_obs, Y_obs, X_new, Y_true, kernel_function,
                 num_restarts=10, num_inducing=None, labels=None):
        self.X_obs = X_obs.astype(float)
        self.Y_obs = Y_obs.astype(float)
        self.X_new = X_new.astype(float)
        self.Y_true = Y_true.astype(float)
        self.kernel_function = kernel_function
        self.num_restarts = num_restarts
        self.num_inducing = num_inducing
        self.labels = labels
        self.has_trained_models = False

    def _get_model_predictions(self, model_class):
        """Returns means and variances."""
        model = model_class(self.X_obs, self.Y_obs, self.kernel_function,
                            num_restarts=self.num_restarts,
                            num_inducing=self.num_inducing)
        return model.predict(self.X_new)

    def _get_gpar_predictions(self):
        return self._get_model_predictions(GPARRegression)

    def _get_igp_predictions(self):
        return self._get_model_predictions(IGPRegression)

    def _get_all_model_predictions(self):
        gpar_predictions = self._get_gpar_predictions()
        igp_predictions = self._get_igp_predictions()
        self.has_trained_models = True
        return gpar_predictions, igp_predictions

    def _plot_results(self, predictions):
        if not self.has_trained_models:
            raise Exception('Cannot plot results from untrained models')

        gpar_predictions, igp_predictions = predictions
        gpar_means, gpar_vars = gpar_predictions
        igp_means, igp_vars = igp_predictions

        plot_mse_values(gpar_means, igp_means, self.Y_true,
                        figure_id_start=0, initial_labels=self.labels)
        plot_all_outputs(gpar_means, gpar_vars, igp_means, igp_vars,
                         self.X_new, self.Y_true, self.X_obs, self.Y_obs,
                         figure_id_start=1, initial_labels=self.labels)
        plt.show()

    def run(self):
        predictions = self._get_all_model_predictions()
        self._plot_results(predictions)
