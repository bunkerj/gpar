import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
from matplotlib import pyplot as plt


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


def get_kernel_integral_values(X_obs, m, start, end):
    n_obs = X_obs.shape[0]
    approx_single_int = np.zeros((n_obs, 1))
    for i in range(n_obs):
        n = len(X_obs[i, :])
        fixed_input = X_obs[i, :].reshape((n, 1))
        result = integrate.nquad(single_integrand, [[start, end]],
                                 args=(m, fixed_input))
        approx_single_int[i] = result[0]
    return approx_single_int


def get_kernel_integral_constant(m, start, end):
    result = integrate.nquad(double_integrand, [[start, end], [start, end]],
                             args=(m,))
    return result[0]


def bayesian_quadrature(m, X_obs, y_single_obs, start, end):
    kernel_int_vect = get_kernel_integral_values(X_obs, m, start, end)
    c = get_kernel_integral_constant(m, start, end)
    C_inv = get_precision_matrix(m, X_obs)

    tmp = kernel_int_vect.transpose().dot(C_inv)
    est_mean = tmp.dot(y_single_obs)
    est_var = c - tmp.dot(kernel_int_vect)
    est_std = np.sqrt(est_var)

    return est_mean, est_std


def plot_bq_integral_gp_dist(integral_base, integral_bq, integral_std_bq):
    ub = np.max((integral_base + 0.25, integral_bq + 2 * integral_std_bq))
    lb = np.min((integral_base - 0.25, integral_bq - 2 * integral_std_bq))
    x_gauss = np.linspace(lb, ub, 1000).flatten()
    y_gauss = stats.norm.pdf(x_gauss, integral_bq, integral_std_bq).flatten()
    plt.plot(x_gauss, y_gauss)


def plot_bq_integrad_truth(custom_func, start, end):
    n_new = 1000
    X_new = np.linspace(start, end, n_new).reshape((n_new, 1))
    y_single_means_true = custom_func(X_new)
    plt.plot(X_new, y_single_means_true, label='True Function')


def plot_bq_integrand_gp(m, start, end, label, display_var=False):
    n_new = 1000
    X_new = np.linspace(start, end, n_new).reshape((n_new, 1))
    y_single_means_pred, y_single_vars_pred = m.predict(X_new)
    plt.plot(X_new, y_single_means_pred, label=label)

    if display_var:
        ub_means_pred = y_single_means_pred + 2 * np.sqrt(y_single_vars_pred)
        lb_means_pred = y_single_means_pred - 2 * np.sqrt(y_single_vars_pred)
        plt.fill_between(
            X_new.flatten(),
            lb_means_pred.flatten(),
            ub_means_pred.flatten(),
            alpha=0.2,
            edgecolor='b')
