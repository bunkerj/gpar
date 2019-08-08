import gpflow
from src.regression.regression import Regression
from src.src_utils import slice_column, concat_right_column


class IGPRegression(Regression):
    def __init__(self, X_obs, Y_obs, kernel_function, num_restarts=0,
                 is_zero_noise=False, num_inducing=None, init_likelihood_var=0.0001):
        super().__init__(X_obs, Y_obs, kernel_function,
                         num_restarts=num_restarts,
                         is_zero_noise=is_zero_noise,
                         num_inducing=num_inducing,
                         init_likelihood_var=init_likelihood_var)
        self.models = self.get_gp_models()

    def get_gp_models(self):
        if hasattr(self, 'models'):
            return self.models
        models = []
        for out_id in range(self.Y_obs.shape[1]):
            single_y = slice_column(self.Y_obs, out_id)
            kernel = self._get_kernel(self.X_obs, self.X_obs)
            m = self._get_model(self.X_obs, single_y, kernel)
            m.likelihood.variance = self.init_likelihood_var
            if self.is_zero_noise:
                m.likelihood.variance = 0.00001
                m.likelihood.variance.trainable = False
            self._optimize_model(m)
            models.append(m)
        return tuple(models)

    def predict_f(self, X_new):
        stacked_means = None
        stacked_vars = None
        for out_id in range(self.Y_obs.shape[1]):
            means, variances = \
                self.single_predict(X_new, out_id)
            stacked_means = concat_right_column(stacked_means, means)
            stacked_vars = concat_right_column(stacked_vars, variances)
        return stacked_means, stacked_vars

    def single_predict(self, X_new, out_id=0):
        m = self.models[out_id]
        return m.predict_f(X_new)
