from GPy.models import GPRegression, SparseGPRegression


class Regression:
    def __init__(self, X_obs, Y_obs, kernel_function, num_restarts=10,
                 is_zero_noise=False, num_inducing=None):
        # Each output stream should correspond to a column.
        self.Y_obs = Y_obs
        self.X_obs = X_obs
        self.n = X_obs.shape[0]
        self._get_kernel = kernel_function
        self.num_restarts = num_restarts
        self.is_zero_noise = is_zero_noise
        self.num_inducing = num_inducing

    def _get_model(self, *base_args):
        return (GPRegression(*base_args)
                if self.num_inducing is None
                else SparseGPRegression(*base_args,
                                        num_inducing=self.num_inducing))
