from matplotlib import pyplot as plt
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from utils import plot_mse_values, plot_all_outputs


class ExperimentRunner:
    def __init__(self, X_obs, Y_obs, X_new, Y_true, kernel_function,
                 num_restarts=10, num_inducing=None,
                 labels=None, figure_start=0,
                 manual_ordering=None):
        self.X_obs = X_obs
        self.Y_obs = Y_obs
        self.X_new = X_new
        self.Y_true = Y_true
        self.kernel_function = kernel_function
        self.num_restarts = num_restarts
        self.num_inducing = num_inducing
        self.labels = labels
        self.has_trained_models = False
        self.figure_start = figure_start
        self.manual_ordering = manual_ordering

    def _get_model(self, model_class, **kwargs):
        return model_class(self.X_obs, self.Y_obs, self.kernel_function,
                           num_restarts=self.num_restarts,
                           num_inducing=self.num_inducing,
                           **kwargs)

    def _get_gpar_predictions_and_ordering(self):
        model = self._get_model(GPARRegression,
                                manual_ordering=self.manual_ordering)
        return model.predict(self.X_new), model.get_ordering()

    def _get_igp_predictions(self):
        model = self._get_model(IGPRegression)
        return model.predict(self.X_new)

    def _get_all_model_predictions_and_ordering(self):
        gpar_predictions, gpar_ordering = self._get_gpar_predictions_and_ordering()
        igp_predictions = self._get_igp_predictions()
        self.has_trained_models = True
        return gpar_predictions, igp_predictions, gpar_ordering

    def _plot_results(self, predictions, gpar_ordering):
        if not self.has_trained_models:
            raise Exception('Cannot plot results from untrained models')

        gpar_predictions, igp_predictions = predictions
        gpar_means, gpar_vars = gpar_predictions
        igp_means, igp_vars = igp_predictions

        plot_mse_values(gpar_means, igp_means, self.Y_true, gpar_ordering,
                        figure_id_start=self.figure_start, initial_labels=self.labels)
        plot_all_outputs(gpar_means, gpar_vars, igp_means, igp_vars, gpar_ordering,
                         self.X_new, self.Y_true, self.X_obs, self.Y_obs,
                         figure_id_start=self.figure_start + 1, initial_labels=self.labels)
        plt.show()

    def run(self):
        gpar_predictions, igp_predictions, gpar_ordering \
            = self._get_all_model_predictions_and_ordering()
        predictions = [gpar_predictions, igp_predictions]
        self._plot_results(predictions, gpar_ordering)
