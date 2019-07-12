import numpy as np
from scipy import stats, special
from numpy import sin, cos, pi, exp, sqrt
from src_utils import concat_right_column

# ------------------------------ Function Relationships  ------------------------------ #

NOISE_VAR = 0.05


def eps():
    return np.random.normal(0, NOISE_VAR)


def add_noise(f):
    return lambda x: f(x) + eps()


def y1_exp1(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4


def y2_exp1(x):
    return cos(y1_exp1(x)) ** 2 + sin(3 * x)


def y3_exp1(x):
    return y2_exp1(x) * (y1_exp1(x) ** 2) + 3 * x


synthetic_functions = (y1_exp1, y2_exp1, y3_exp1)

noisy_synthetic_functions = (
    add_noise(synthetic_functions[0]),
    add_noise(synthetic_functions[1]),
    add_noise(synthetic_functions[2]))


# ------------------------------ Noise Structure ------------------------------ #


def f1_exp2(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4


def f2_exp2(x):
    theta1 = 2 * pi
    theta2 = 2 * pi
    theta3 = 2 * pi
    theta4 = 2 * pi
    return (1 / 5) * exp(2 * x) * \
           (theta1 * cos(theta2 * pi * x) + theta3 * cos(theta4 * pi * x)) + \
           sqrt(2 * x)


def get_noise_matrix(n, means=(0, 0), std_devs=(0.1, 0.04)):
    if len(means) != len(std_devs):
        raise Exception('Length mismatch')
    noise_matrix = None
    for i in range(len(means)):
        eps = np.random.normal(means[i], std_devs[i], size=(n, 1))
        noise_matrix = concat_right_column(noise_matrix, eps)
    return noise_matrix


def y_exp2(X, is_noisy):
    n = len(X)
    Y = np.zeros((n, 4))
    noise_matrix = get_noise_matrix(n) if is_noisy else None
    for idx in range(n):
        Y[idx, 0] = f1_exp2(X[idx])
        Y[idx, 1] = f2_exp2(X[idx])
        Y[idx, 2] = f2_exp2(X[idx])
        Y[idx, 3] = f2_exp2(X[idx])
        if is_noisy:
            Y[idx, 0] += noise_matrix[idx, 0]
            Y[idx, 1] += (np.sin(2 * np.pi * X[idx]) ** 2) * noise_matrix[idx, 0] + \
                         (np.cos(2 * np.pi * X[idx]) ** 2) * noise_matrix[idx, 1]
            Y[idx, 2] += np.sin(2 * np.pi * noise_matrix[idx, 0]) + noise_matrix[idx, 1]
            Y[idx, 3] += np.sin(2 * np.pi * X[idx]) * noise_matrix[idx, 0] + noise_matrix[idx, 1]
    return Y


# ------------------------------ Misc ------------------------------ #


def bessel_integrand(n, x, t):
    return (1 / np.pi) * np.cos(n * t - x * np.sin(t))


bessel_integrands = (
    lambda x: bessel_integrand(0, x, 10),
    lambda x: bessel_integrand(1, x, 10),
    lambda x: bessel_integrand(2, x, 10))

bessel_functions = (
    lambda x: special.jv(0, x),
    lambda x: special.jv(1, x),
    lambda x: special.jv(2, x))

struve_functions = (
    lambda x: special.struve(0, x),
    lambda x: special.struve(1, x),
    lambda x: special.struve(2, x))

# gaussian_functions = (
#     lambda x: stats.norm.pdf(x, 0, 1),
#     lambda x: stats.norm.pdf(x, 0, 2),
#     lambda x: stats.norm.pdf(x, 0, 4))


gaussian_functions = (
    lambda x: stats.norm.pdf(x, 0, 1),
    lambda x: stats.norm.pdf(x, 0, 1),
    lambda x: stats.norm.pdf(x, 0, 1))

custom_functions = (
    lambda x: 3 * x + x ** 0.5 + np.sin(10 * x) + 5,
    lambda x: 3 * x + np.sin(10 * x) + 5,
    lambda x: x ** 0.5 + np.cos(10 * x) + 10
)
