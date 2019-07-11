from gpflow.models import GPR, SGPR


class Regression:
    def __init__(self, X_obs, Y_obs, kernel_function, num_restarts=10,
                 is_zero_noise=False, num_inducing=None, init_likelihood_var=0.0001):
        # Each Y_obs column should correspond to an output stream.
        self.Y_obs = Y_obs
        self.X_obs = X_obs
        self.n = X_obs.shape[0]
        self._get_kernel = kernel_function
        self.num_restarts = num_restarts
        self.is_zero_noise = is_zero_noise
        self.num_inducing = num_inducing
        self.init_likelihood_var = init_likelihood_var

    def _optimize_model(self, m, num_restarts=10):
        pass

    def _get_model(self, *base_args):
        return GPR(*base_args)
