import numpy as np
import scipy.integrate as integrate
from src_utils import stack_all_columns, sample_from_bounds


class BayesianQuadrature:
    def __init__(self, global_model, is_monte_carlo=False, n_samples=100):
        self.global_model = global_model
        self.is_monte_carlo = is_monte_carlo
        self.n_samples = n_samples

    def _get_mc_samples(self, bounds_list):
        return [sample_from_bounds(bounds_list) for _ in range(self.n_samples)]

    def _get_mc_constant(self, bounds_list):
        v = 1
        for bounds in bounds_list:
            v *= (bounds[1] - bounds[0])
        return v

    def _integrate_monte_carlo(self, integrand, bounds_list, args):
        samples = self._get_mc_samples(bounds_list)
        outputs = []
        for sample in samples:
            outputs.append(integrand(*sample, *args))
        return np.mean(outputs) * self._get_mc_constant(bounds_list)

    def _get_augmented_input(self, X, m, input_dim):
        if input_dim > 1:
            # Augment input for GPAR (ignored when using IGPs)
            means, vars = self.global_model.predict_f(X)
            ordered_means = means[:, self.global_model.get_ordering()]
            Y = ordered_means[:, :(input_dim - 1)]
            return stack_all_columns([X, Y])
        else:
            return X

    def _single_integrand(self, *args):
        raw_input, m, fixed_input = args
        input_dim = fixed_input.shape[1]
        X = np.array(raw_input).reshape((1, 1))
        X_aug = self._get_augmented_input(X, m, input_dim)
        return m.kern.compute_K_symm(X_aug, fixed_input)

    def _double_integrand(self, *args):
        raw_input1, raw_input2, m, input_dim = args
        input1 = np.array(raw_input1).reshape((1, 1))
        input2 = np.array(raw_input2).reshape((1, 1))
        input1_aug = self._get_augmented_input(input1, m, input_dim)
        input2_aug = self._get_augmented_input(input2, m, input_dim)
        return m.kern.compute_K_symm(input1_aug, input2_aug)

    def _get_precision_matrix(self, m, X_obs):
        n = X_obs.shape[0]
        input_dim = X_obs.shape[1]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                cov = m.kern.compute_K_symm(X_obs[i].reshape((1, input_dim)),
                                            X_obs[j].reshape((1, input_dim)))
                C[i, j] = cov
                C[j, i] = cov
        return np.linalg.pinv(C + np.eye(n) * m.kern.variance.value)

    def _get_integration_bounds_list(self, start, end):
        return [[start, end]]

    def _get_kernel_integral_values(self, X_obs, m, start, end):
        n_obs = X_obs.shape[0]
        approx_single_int = np.zeros((n_obs, 1))
        for i in range(n_obs):
            n = len(X_obs[i, :])
            fixed_input = X_obs[i, :].reshape((1, n))
            bounds_list = self._get_integration_bounds_list(start, end)
            approx_single_int[i] = self._integrate(self._single_integrand,
                                                   bounds_list, args=(m, fixed_input))
        return approx_single_int

    def _integrate(self, integrand, bounds_list, args):
        if self.is_monte_carlo:
            return self._integrate_monte_carlo(integrand, bounds_list, args)
        else:
            return integrate.nquad(integrand, bounds_list, args)[0]

    def _get_kernel_integral_constant(self, X_obs, m, start, end):
        bounds_list = self._get_integration_bounds_list(start, end)
        double_bounds_list = bounds_list + bounds_list
        input_dim = X_obs.shape[1]
        return self._integrate(self._double_integrand, double_bounds_list, args=(m, input_dim))

    def predict_f(self, m, X_obs, y_single_obs, start, end):
        kernel_int_vect = self._get_kernel_integral_values(X_obs, m, start, end)
        c = self._get_kernel_integral_constant(X_obs, m, start, end)
        C_inv = self._get_precision_matrix(m, X_obs)

        tmp = kernel_int_vect.transpose().dot(C_inv)
        est_mean = tmp.dot(y_single_obs)
        est_var = c - tmp.dot(kernel_int_vect)
        est_std = np.sqrt(est_var)

        return est_mean, est_std
