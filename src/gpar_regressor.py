import GPy
import numpy as np
from matplotlib import pyplot as plt
from utils import should_update_max, slice_column


class GPARRegression:
    def __init__(self, X, Y, kernel_function, num_restarts=10):
        # Each output stream should correspond to a column.
        self.Y = Y
        self.X = X
        self.n = X.shape[0]
        self.kernel_function = kernel_function
        self.num_restarts = num_restarts
        self.ordering = self._get_ordering()
        self.gaussian_process_dict = self.get_gaussian_process_dict()

    def get_gaussian_process_dict(self):
        gaussian_process_dict = {}
        current_X = self.X
        for out_id in self.ordering:
            m = self._get_trained_gp_model(current_X, out_id)
            y = slice_column(self.Y, out_id)
            current_X = self._augment_input(current_X, y)
            gaussian_process_dict[out_id] = m
        return gaussian_process_dict

    def _get_ordering(self):
        """Use likelihood to establish the conditional ordering."""
        ordering = []
        remaining_output_ids = list(range(self.Y.shape[1]))
        current_X = self.X
        for _ in range(self.Y.shape[1] - 1):
            max_log_likelihood_id = \
                self._get_max_log_likelihood_id(current_X, remaining_output_ids)
            y = slice_column(self.Y, max_log_likelihood_id)
            current_X = self._augment_input(current_X, y)
            ordering.append(max_log_likelihood_id)
            remaining_output_ids.remove(max_log_likelihood_id)
        ordering.append(remaining_output_ids[0])
        return tuple(ordering)

    def _get_max_log_likelihood_id(self, current_X, remaining_output_ids):
        """Get the ID of the GP with the max likelihood."""
        max_log_likelihood_value = None
        max_log_likelihood_id = None
        for out_id in remaining_output_ids:
            log_likelihood = self._get_log_likelihood(current_X, out_id)
            if should_update_max(max_log_likelihood_value, log_likelihood):
                max_log_likelihood_value = log_likelihood
                max_log_likelihood_id = out_id
        return max_log_likelihood_id

    def _augment_input(self, current_X, y):
        return np.concatenate((current_X, y), axis=1)

    def _get_log_likelihood(self, current_X, out_id):
        m = self._get_trained_gp_model(current_X, out_id)
        return m.log_likelihood()

    def _get_trained_gp_model(self, current_X, out_id):
        y = slice_column(self.Y, out_id)
        kernel = self.kernel_function(input_dim=current_X.shape[1])
        m = GPy.models.GPRegression(current_X, y, kernel)
        m.optimize_restarts(self.num_restarts, verbose=False)
        return m

    def stack_in_order(self, data_dict):
        result = None
        for id in range(len(data_dict)):
            data = data_dict[id]
            if result is None:
                result = data
            else:
                result = np.concatenate((result, data), axis=1)
        return result

    def predict(self, X_new):
        """Preform prediction using the conditional ordering."""
        current_X = X_new
        mean_dict = {}
        var_dict = {}
        for out_id in self.ordering:
            m = self.gaussian_process_dict[out_id]
            mean, var = m.predict(current_X)
            current_X = self._augment_input(current_X, mean)
            mean_dict[out_id] = mean
            var_dict[out_id] = var
        return self.stack_in_order(mean_dict), self.stack_in_order(var_dict)
