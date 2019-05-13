import GPy
import numpy as np
from utils import should_update_max, slice_column, concat_right_column


class GPARRegression:
    def __init__(self, X, Y, kernel_function, manual_ordering=None, num_restarts=10):
        # Each output stream should correspond to a column.
        self.Y = Y
        self.X = X
        self.n = X.shape[0]
        self._get_kernel = kernel_function
        self.num_restarts = num_restarts
        self.ordering = manual_ordering

        models, ordering = self._get_models_and_ordering(manual_ordering)
        self.gaussian_process_dict = dict(zip(ordering, models))
        self.ordering = ordering

    def _get_models_and_ordering(self, manual_ordering):
        if manual_ordering is None:
            return self._get_gp_models_with_ordering()
        else:
            models = self._get_gp_models(manual_ordering)
            return models, manual_ordering

    def _get_gp_models(self, manual_ordering):
        models = []
        current_X = self.X
        for out_id in manual_ordering:
            m = self._get_trained_gp_model(current_X, out_id)
            models.append(m)
            current_X = self._augment_X(current_X, out_id)
        return tuple(models)

    def _get_gp_models_with_ordering(self):
        """
        Return models and their respective order.
        Use likelihood to establish the conditional ordering.
        """
        models = []
        ordering = []
        current_X = self.X
        output_count = self.Y.shape[1]
        remaining_output_ids = list(range(output_count))
        for _ in range(output_count):
            max_log_likelihood_model, max_log_likelihood_id = \
                self._get_max_log_likelihood_models(current_X, remaining_output_ids)
            models.append(max_log_likelihood_model)
            ordering.append(max_log_likelihood_id)
            remaining_output_ids.remove(max_log_likelihood_id)
            current_X = self._augment_X(current_X, max_log_likelihood_id)
        return tuple(models), tuple(ordering)

    def _augment_X(self, current_X, out_id):
        y = slice_column(self.Y, out_id)
        return concat_right_column(current_X, y)

    def _get_max_log_likelihood_models(self, current_X, remaining_output_ids):
        """Get the ID of the GP with the max likelihood."""
        max_log_likelihood_id = None
        max_log_likelihood_value = None
        max_log_likelihood_model = None
        for out_id in remaining_output_ids:
            m = self._get_trained_gp_model(current_X, out_id)
            log_likelihood = m.log_likelihood()
            if should_update_max(max_log_likelihood_value, log_likelihood):
                max_log_likelihood_id = out_id
                max_log_likelihood_value = log_likelihood
                max_log_likelihood_model = m
        return max_log_likelihood_model, max_log_likelihood_id

    def _get_trained_gp_model(self, current_X, out_id):
        y = slice_column(self.Y, out_id)
        kernel = self._get_kernel(self.X, current_X)
        m = GPy.models.GPRegression(current_X, y, kernel)
        m.optimize_restarts(self.num_restarts, verbose=True)
        return m

    def _stack_in_order(self, data_dict):
        """Stack data according to the order defined in dictionary keys."""
        result = None
        for id in range(len(data_dict)):
            data = data_dict[id]
            if result is None:
                result = data
            else:
                result = concat_right_column(result, data)
        return result

    def predict(self, X_new):
        """Preform prediction using the conditional ordering."""
        current_X = X_new
        mean_dict = {}
        var_dict = {}
        for out_id in self.ordering:
            m = self.gaussian_process_dict[out_id]
            mean, var = m.predict(current_X)
            current_X = concat_right_column(current_X, mean)
            mean_dict[out_id] = mean
            var_dict[out_id] = var
        return self._stack_in_order(mean_dict), self._stack_in_order(var_dict)
