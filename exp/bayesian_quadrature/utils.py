import numpy as np
import scipy.integrate as integrate


def single_integrand(x, m, X_obs, i):
    single_input = x * np.ones((1, 1))
    const_input = X_obs[i, 0].reshape((1, 1))
    return m.kern.K(single_input, const_input)


def double_integrand(x1, x2, m):
    input1 = x1 * np.ones((1, 1))
    input2 = x2 * np.ones((1, 1))
    return m.kern.K(input1, input2)


def get_covariance_matrix(m, X_obs):
    n = len(X_obs)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = m.kern.K(X_obs[i].reshape((1, 1)), X_obs[j].reshape((1, 1)))
            C[i, j] = cov
            C[j, i] = cov
    return C + np.eye(n, n) * 0.1


def get_kernel_integral_values(X_obs, end, m, n_obs, start):
    approx_single_int = np.zeros((n_obs, 1))
    for i in range(n_obs):
        result = integrate.quad(single_integrand, start, end, args=(m, X_obs, i))
        approx_single_int[i] = result[0]
    return approx_single_int


def bayesian_quadrature(m, X_obs, y_single_obs, start, end):
    n_obs = X_obs.shape[0]
    approx_single_int = get_kernel_integral_values(X_obs, end, m, n_obs, start)
    result = integrate.nquad(double_integrand, [[start, end], [start, end]], args=(m,))
    c = result[0]
    C = get_covariance_matrix(m, X_obs)
    C_inv = np.linalg.inv(C)
    tmp = approx_single_int.transpose().dot(C_inv)

    est_mean = tmp.dot(y_single_obs)
    est_var = c - tmp.dot(approx_single_int)
    est_std = np.sqrt(est_var)

    return est_mean, est_std
