import numpy as np
from numpy import sin, cos, pi, exp, sqrt

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


y1_exp1_noisy = add_noise(y1_exp1)
y2_exp1_noisy = add_noise(y2_exp1)
y3_exp1_noisy = add_noise(y3_exp1)

synthetic_functions = (y1_exp1, y2_exp1, y3_exp1)
noisy_synthetic_functions = (y1_exp1_noisy, y2_exp1_noisy, y3_exp1_noisy)


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


def get_noise_matrix(n):
    eps1_values = np.random.normal(0, 0.1, size=(n, 1))
    eps2_values = np.random.normal(0, 0.04, size=(n, 1))
    return np.concatenate((eps1_values, eps2_values), axis=1)


def y_exp2_clean(X, is_noisy):
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
