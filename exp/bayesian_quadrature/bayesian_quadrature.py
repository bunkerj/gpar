import numpy as np
import scipy.integrate as integrate
from src_utils import stack_all_columns


class BayesianQuadrature:
    def __init__(self, global_model):
        self.global_model = global_model

    def integrate_monte_carlo(self, integrand, bounds, args, n_points=10000):
        pass

    def get_augmented_input(self, X, m, input_dim):
        if input_dim > 1:
            # Augment input for GPAR
            means, vars = self.global_model.predict(X)
            ordered_means = means[:, self.global_model.get_ordering()]
            Y = ordered_means[:, :(input_dim - 1)]
            return stack_all_columns([X, Y])
        else:
            return X

    def single_integrand(self, *args):
        raw_input, m, fixed_input = args
        input_dim = fixed_input.shape[1]
        X = np.array(raw_input).reshape((1, 1))
        X_aug = self.get_augmented_input(X, m, input_dim)
        return m.kern.K(X_aug, fixed_input)

    def double_integrand(self, *args):
        raw_input1, raw_input2, m, input_dim = args
        input1 = np.array(raw_input1).reshape((1, 1))
        input2 = np.array(raw_input2).reshape((1, 1))
        input1_aug = self.get_augmented_input(input1, m, input_dim)
        input2_aug = self.get_augmented_input(input2, m, input_dim)
        return m.kern.K(input1_aug, input2_aug)

    def get_precision_matrix(self, m, X_obs):
        n = X_obs.shape[0]
        input_dim = X_obs.shape[1]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                cov = m.kern.K(X_obs[i].reshape((1, input_dim)),
                               X_obs[j].reshape((1, input_dim)))
                C[i, j] = cov
                C[j, i] = cov
        return np.linalg.pinv(C + np.eye(n, n) * m.Gaussian_noise.variance)

    def get_integration_bounds(self, start, end):
        bounds = [[start, end]]
        return bounds

    def get_kernel_integral_values(self, X_obs, m, start, end):
        n_obs = X_obs.shape[0]
        approx_single_int = np.zeros((n_obs, 1))
        for i in range(n_obs):
            n = len(X_obs[i, :])
            fixed_input = X_obs[i, :].reshape((1, n))
            bounds = self.get_integration_bounds(start, end)
            result = integrate.nquad(self.single_integrand, bounds, args=(m, fixed_input))
            approx_single_int[i] = result[0]
        return approx_single_int

    def get_kernel_integral_constant(self, X_obs, m, start, end):
        bounds = self.get_integration_bounds(start, end)
        double_bounds = bounds + bounds
        input_dim = X_obs.shape[1]
        result = integrate.nquad(self.double_integrand, double_bounds, args=(m, input_dim))
        return result[0]

    def predict(self, m, X_obs, y_single_obs, start, end):
        kernel_int_vect = self.get_kernel_integral_values(X_obs, m, start, end)
        c = self.get_kernel_integral_constant(X_obs, m, start, end)
        C_inv = self.get_precision_matrix(m, X_obs)

        tmp = kernel_int_vect.transpose().dot(C_inv)
        est_mean = tmp.dot(y_single_obs)
        est_var = c - tmp.dot(kernel_int_vect)
        est_std = np.sqrt(est_var)

        return est_mean, est_std
