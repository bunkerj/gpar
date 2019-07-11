import gpflow
from regression.regression import Regression
from src_utils import should_update_max, slice_column, concat_right_column


class GPARRegression(Regression):
    def __init__(self, X, Y, kernel_function, manual_ordering=None, num_restarts=10,
                 is_zero_noise=False, num_inducing=None, init_likelihood_var=0.0001):
        super().__init__(X, Y, kernel_function,
                         num_restarts=num_restarts,
                         is_zero_noise=is_zero_noise,
                         num_inducing=num_inducing,
                         init_likelihood_var=init_likelihood_var)
        models, ordering = self._get_models_and_ordering(manual_ordering)
        self.models = models
        self.ordering = ordering
        self.gaussian_process_dict = dict(zip(ordering, models))
        self.print_ordering()

    def get_ordering(self):
        return self.ordering

    def get_gp_dict(self):
        return self.gaussian_process_dict

    def get_ordering_string(self):
        ordering_arr = ['Y{}'.format(out_id + 1)
                        for out_id in self.ordering]
        return ', '.join(ordering_arr)

    def print_ordering(self):
        ordering_string = self.get_ordering_string()
        print('\nOutput ordering: {}'.format(ordering_string))

    def _get_models_and_ordering(self, manual_ordering):
        print('Training GPAR model...')
        if manual_ordering is None:
            results = self._get_gp_models_with_ordering()
            print('Done.')
            return results
        else:
            models = self._get_gp_models(manual_ordering)
            print('Done.')
            return models, manual_ordering

    def _print_iteration(self, curr_iter, total_iter_count):
        print('Iteration {}/{}...'.format(curr_iter, total_iter_count))

    def _get_gp_models(self, manual_ordering):
        if hasattr(self, 'models'):
            return self.models
        models = []
        current_X = self.X_obs
        for iter, out_id in enumerate(manual_ordering):
            self._print_iteration(iter + 1, len(manual_ordering))
            m = self._get_trained_gp_model(current_X, out_id)
            models.append(m)
            current_X = self.augment_X(current_X, out_id)
        return tuple(models)

    def _get_gp_models_with_ordering(self):
        """
        Return models and their respective order.
        Use likelihood to establish the conditional ordering.
        """
        models = []
        ordering = []
        current_X = self.X_obs
        output_count = self.Y_obs.shape[1]
        remaining_output_ids = list(range(output_count))
        for iter in range(output_count):
            self._print_iteration(iter + 1, output_count)
            max_log_likelihood_model, max_log_likelihood_id = \
                self._get_max_log_likelihood_models(current_X, remaining_output_ids)
            models.append(max_log_likelihood_model)
            ordering.append(max_log_likelihood_id)
            remaining_output_ids.remove(max_log_likelihood_id)
            current_X = self.augment_X(current_X, max_log_likelihood_id)
        return tuple(models), tuple(ordering)

    def augment_X(self, current_X, out_id):
        if current_X is None:
            return self.X_obs
        y = slice_column(self.Y_obs, out_id)
        return concat_right_column(current_X, y)

    def _get_max_log_likelihood_models(self, current_X, remaining_output_ids):
        """Get the ID of the GP with the max likelihood."""
        max_log_likelihood_id = None
        max_log_likelihood_value = None
        max_log_likelihood_model = None
        for out_id in remaining_output_ids:
            m = self._get_trained_gp_model(current_X, out_id)
            log_likelihood = m.compute_log_likelihood()
            if should_update_max(max_log_likelihood_value, log_likelihood):
                max_log_likelihood_id = out_id
                max_log_likelihood_value = log_likelihood
                max_log_likelihood_model = m
        return max_log_likelihood_model, max_log_likelihood_id

    def _get_trained_gp_model(self, current_X, out_id):
        y = slice_column(self.Y_obs, out_id)
        kernel = self._get_kernel(self.X_obs, current_X)
        m = self._get_model(current_X, y, kernel)
        m.likelihood.variance = self.init_likelihood_var
        if self.is_zero_noise:
            m.likelihood.variance = 0.00001
            m.likelihood.variance.trainable = False
        gpflow.train.ScipyOptimizer().minimize(m)
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

    def predict_f(self, X_new):
        """Preform prediction using the conditional ordering."""
        current_X = X_new
        mean_dict = {}
        var_dict = {}
        for out_id in self.ordering:
            m = self.gaussian_process_dict[out_id]
            mean, var = m.predict_f(current_X)
            current_X = concat_right_column(current_X, mean)
            mean_dict[out_id] = mean
            var_dict[out_id] = var
        return self._stack_in_order(mean_dict), self._stack_in_order(var_dict)

    def predict_single_output(self, X_new, out_id):
        m = self.gaussian_process_dict[out_id]
        return m.predict_f(X_new)
