import GPy
from src_utils import slice_column, concat_right_column


class IGPRegression:
    def __init__(self, X_obs, Y_obs, kernel_function, num_restarts=10):
        # Each output stream should correspond to a column.
        self.Y_obs = Y_obs
        self.X_obs = X_obs
        self.n = X_obs.shape[0]
        self._get_kernel = kernel_function
        self.num_restarts = num_restarts
        self.models = self.get_gp_models()

    def get_gp_models(self):
        if hasattr(self, 'models'):
            return self.models
        models = []
        for out_id in range(self.Y_obs.shape[1]):
            single_y = slice_column(self.Y_obs, out_id)
            kernel = self._get_kernel(self.X_obs, self.X_obs)
            m = GPy.models.GPRegression(self.X_obs, single_y, kernel)
            m.optimize_restarts(self.num_restarts, verbose=False)
            models.append(m)
        return tuple(models)

    def predict(self, X_new):
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
        return m.predict(X_new)
