import numpy as np
import scipy.integrate as integrate


def single_integrand(*args):
    m, fixed_input = args[-2:]
    n = len(args[:-2])
    X = np.array(args[:-2]).reshape((n, 1))
    return m.kern.K(X, fixed_input)


def double_integrand(*args):
    m = args[-1]
    n = (len(args) - 1) // 2
    X1 = np.array(args[:n]).reshape((n, 1))
    X2 = np.array(args[n:2 * n]).reshape((n, 1))
    input1 = X1 * np.ones((1, 1))
    input2 = X2 * np.ones((1, 1))
    return m.kern.K(input1, input2)


def get_precision_matrix(m, X_obs):
    n = len(X_obs)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = m.kern.K(X_obs[i].reshape((1, 1)), X_obs[j].reshape((1, 1)))
            C[i, j] = cov
            C[j, i] = cov
    return np.linalg.pinv(C)


def get_kernel_integral_values(X_obs, end, m, n_obs, start):
    approx_single_int = np.zeros((n_obs, 1))
    for i in range(n_obs):
        n = len(X_obs[i, :])
        fixed_input = X_obs[i, :].reshape((n, 1))
        result = integrate.nquad(single_integrand, [[start, end]],
                                 args=(m, fixed_input))
        approx_single_int[i] = result[0]
    return approx_single_int


def get_kernel_integral_constant(start, end, m):
    result = integrate.nquad(double_integrand, [[start, end], [start, end]],
                             args=(m,))
    return result[0]


def bayesian_quadrature(m, X_obs, y_single_obs, start, end):
    n_obs = X_obs.shape[0]
    kernel_int_vect = get_kernel_integral_values(X_obs, end, m, n_obs, start)
    c = get_kernel_integral_constant(start, end, m)
    C_inv = get_precision_matrix(m, X_obs)

    tmp = kernel_int_vect.transpose().dot(C_inv)
    est_mean = tmp.dot(y_single_obs)
    est_var = c - tmp.dot(kernel_int_vect)
    est_std = np.sqrt(est_var)

    return est_mean, est_std
