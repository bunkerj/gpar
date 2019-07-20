import gpflow
import numpy as np


class Regression:
    def __init__(self, X_obs, Y_obs, kernel_function, num_restarts=0,
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

    def _randomize_parameters(self, m):
        trainables = m.read_trainables()
        for key in trainables:
            trainables[key] = np.random.normal(0, 1)
        m.assign(trainables)

    def _copy_tensor_dict(self, tensor_dict):
        copy_dict = {}
        for key in tensor_dict:
            copy_dict[key] = float(tensor_dict[key])
        return copy_dict

    def _optimize_model(self, m):
        max_likelihood = None
        max_trainables = None
        for _ in range(0, self.num_restarts + 1):
            gpflow.train.ScipyOptimizer().minimize(m)
            current_likelihood = m.compute_log_likelihood()
            if max_likelihood is None or current_likelihood > max_likelihood:
                max_likelihood = current_likelihood
                max_trainables = self._copy_tensor_dict(m.read_trainables())
            self._randomize_parameters(m)
        m.assign(max_trainables)

    def _get_model(self, *base_args):
        return gpflow.models.GPR(*base_args)
