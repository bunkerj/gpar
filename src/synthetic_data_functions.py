import numpy as np
from math import sin, cos, pi, exp, sqrt

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


def y_exp2(noise_func, x):
    n = len(x)
    eps1_values = np.random.normal(0, 0.1, size=(n, 1))
    eps2_values = np.random.normal(0, 0.1, size=(n, 1))
    Y = np.zeros((n, 2))
    for idx in range(n):
        curr_x = x[idx]
        Y[idx, 0] = f1_exp2(curr_x) + eps1_values[idx]
        Y[idx, 1] = noise_func(curr_x, eps1_values[idx], eps2_values[idx])
    return Y


def scheme1_noise_component(curr_x, eps1, eps2):
    return f2_exp2(curr_x) + \
           (sin(2 * pi * curr_x) ** 2) * eps1 + \
           (cos(2 * pi * curr_x) ** 2) * eps2


def scheme2_noise_component(curr_x, eps1, eps2):
    return f2_exp2(curr_x) + \
           sin(pi * eps1) + \
           eps2


def scheme3_noise_component(curr_x, eps1, eps2):
    return f2_exp2(curr_x) + \
           sin(pi * curr_x) * eps1 + \
           eps2


def y_scheme1_exp2(x):
    return y_exp2(scheme1_noise_component, x)


def y_scheme2_exp2(x):
    return y_exp2(scheme2_noise_component, x)


def y_scheme3_exp2(x):
    return y_exp2(scheme3_noise_component, x)


y1_exp1_noisy = add_noise(y1_exp1)
y2_exp1_noisy = add_noise(y2_exp1)
y3_exp1_noisy = add_noise(y3_exp1)

synthetic_functions = (y1_exp1, y2_exp1, y3_exp1)
noisy_synthetic_functions = (y1_exp1_noisy, y2_exp1_noisy, y3_exp1_noisy)
