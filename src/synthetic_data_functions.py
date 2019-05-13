import numpy as np
from math import sin, cos, pi

NOISE_VAR = 0.05


def eps():
    return np.random.normal(0, NOISE_VAR)


def add_noise(f):
    return lambda x: f(x) + eps()


def y1(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4


def y2(x):
    return cos(y1(x)) ** 2 + sin(3 * x)


def y3(x):
    return y2(x) * (y1(x) ** 2) + 3 * x


y1_noisy = add_noise(y1)
y2_noisy = add_noise(y2)
y3_noisy = add_noise(y3)
