import numpy as np
from scipy import stats, special
from numpy import sin, cos, pi, exp, sqrt
from src.src_utils import concat_right_column

NOISE_STD = 0.05


def get_noisy_functions(functions):
    return tuple(add_noise(func) for func in functions)


# ------------------------------ Function Relationships  ------------------------------ #


def eps():
    return np.random.normal(0, NOISE_STD)


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


# ------------------------------ Low Complexity ------------------------------ #
def y1_low(x):
    return 5 * np.sin(x) + 5


def y2_low(x):
    return 7 * y1_low(x) ** (0.5 * x) + 12


low_complexity_functions = (y1_low, y2_low)

noisy_low_complexity_functions = get_noisy_functions(low_complexity_functions)


# ------------------------------ Medium Complexity ------------------------------ #

def y1_medium(x):
    return 5 * (x ** 2) + 100 * np.sin(x) + 5


def y2_medium(x):
    return 0.005 * y1_medium(x) + x / y1_medium(x) + 200


def y3_medium(x):
    return x ** 2 + y1_medium(x) + 100 * np.cos(y2_medium(x))


medium_complexity_functions = (y1_medium, y2_medium, y3_medium)

noisy_medium_complexity_functions = get_noisy_functions(medium_complexity_functions)


# ------------------------------ High Complexity ------------------------------ #

def y1_high(x):
    return 5 * (np.cos(x) ** 2) + x * np.sin(np.cos(x)) ** 2 + 5


def y2_high(x):
    return 5 * np.sin(0.0001 * y1_high(x) ** 3) + x * y1_high(x) * np.cos(np.sin(x)) ** 2 + 100


def y3_high(x):
    return 0.0001 * (y2_high(np.cos(x) * y1_high(x)) ** 2) + x * np.sin(np.sin(x)) ** 3 - 200


def y4_high(x):
    return (y1_high(x) ** 2) / y2_high(x) + y2_high(x) / y3_high(x) + (y3_high(x) ** 2) / y1_high(x)


def y5_high(x):
    return np.sqrt(y1_high(x) * y4_high(x)) + y2_high(x) + y3_high(x) + 10 * np.cos(x)


high_complexity_functions = (y1_high, y2_high, y3_high, y4_high, y5_high)

noisy_high_complexity_functions = get_noisy_functions(high_complexity_functions)


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
    lambda x: special.jv(0, x) + special.jv(1, x),
    lambda x: special.jv(0, x) + special.jv(1, x) + special.jv(2, x))

struve_functions = (
    lambda x: special.struve(0, x),
    lambda x: special.struve(1, x),
    lambda x: special.struve(2, x))

gaussian_functions = (
    lambda x: stats.norm.pdf(x, 1, np.sqrt(2)),
    lambda x: stats.norm.pdf(x, -3, np.sqrt(1)),
    lambda x: stats.norm.pdf(x, 0, np.sqrt(4)))

custom_functions = (
    lambda x: x ** 0.5 + np.sin(x) + 5,
    lambda x: 0.5 * x + np.sin(x) + 5,
    lambda x: np.cos(x) ** 2 + np.sin(x)
)
