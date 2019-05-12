import numpy as np
from math import sin, cos, pi

NOISE_VAR = 0.05


def eps():
    return np.random.normal(0, NOISE_VAR)


def y1(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4


def y2(x):
    return cos(y1(x)) ** 2 + sin(3 * x)


def y3(x):
    return y2(x) * (y1(x) ** 2) + 3 * x


def y1_noisy(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4 + eps()


def y2_noisy(x):
    return cos(y1_noisy(x)) ** 2 + sin(3 * x) + eps()


def y3_noisy(x):
    return y2_noisy(x) * (y1(x) ** 2) + 3 * x + eps()
