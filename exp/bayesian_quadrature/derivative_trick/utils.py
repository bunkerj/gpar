import gpflow
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def contains_none(arr):
    return any([v is None for v in arr])


def execute_tensor(tensor):
    sess = gpflow.get_default_session()
    return sess.run(tensor)


def predict_integral(m, X_int, X_obs, Y_obs):
    X_lb = 0 * X_int + X_int[0]
    K = m.kern.compute_K_symm(X_obs)
    K_noisy = K + float(m.likelihood.variance.value) * np.eye(K.shape[0])
    K_s = execute_tensor(m.kern.K_s(X_int, X_obs) - m.kern.K_s(X_lb, X_obs))
    K_ss = execute_tensor(m.kern.K_ss(X_int) - 2 * m.kern.K_ss(X_int, X_lb) + m.kern.K_ss(X_lb))
    K_inv = np.linalg.inv(K_noisy)
    tmp = K_s.dot(K_inv)
    means = tmp.dot(Y_obs)
    variances = K_ss - tmp.dot(K_s.transpose())
    return means, variances


def plot_gaussian(integral_truth, mean, variance, label=None):
    std = np.sqrt(variance)
    ub = mean + 2 * std
    lb = mean - 2 * std
    x_gauss = np.linspace(lb, ub, 1000).flatten()
    y_gauss = stats.norm.pdf(x_gauss, mean, std).flatten()
    plt.title('Integral Distribution')
    plt.plot(x_gauss, y_gauss, label=label)
    plt.axvline(mean, label='Approximation', color='r', linestyle='--')
    plt.axvline(integral_truth, label='Truth', color='g', linestyle='--')
    plt.legend(loc='upper right')
    plt.grid()


def plot_variances(x_test, means, variances):
    ub = means + 2 * np.sqrt(variances)
    lb = means - 2 * np.sqrt(variances)
    plt.fill_between(
        x_test.flatten(),
        lb.flatten(),
        ub.flatten(),
        alpha=0.2,
        edgecolor='b')


def plot_gp_vs_truth(X_true, Y_true, X_obs, Y_obs, X_test, means, variances, title):
    plt.title(title)
    if not contains_none([X_obs, Y_obs]):
        plt.scatter(X_obs, Y_obs,
                    color='b', marker='x',
                    label='Observations')
    plt.plot(X_true, Y_true, label='Truth')
    plt.plot(X_test, means, label='GP Mean')
    plot_variances(X_test, means, variances)
    plt.legend(loc='upper right')
    plt.grid()
