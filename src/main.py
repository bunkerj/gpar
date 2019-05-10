import GPy
import numpy as np
from math import sin, cos, pi
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression

NOISE_VAR = 0.05


def eps():
    return np.random.normal(0, NOISE_VAR)


def y1(x):
    return -sin(10 * pi * (x + 1)) / (2 * x + 1) - x ** 4 + eps()


def y2(x):
    return cos(y1(x)) ** 2 + sin(3 * x) + eps()


def y3(x):
    return y2(x) * (y1(x) ** 2) + 3 * x + eps()


n = 25
np.random.seed(7)
X = np.linspace(0, 1, n).reshape((n, 1))
Y = np.array(list(map(y3, X))).reshape((n, 1))

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.1)
m = GPARRegression(X, Y, kernel)
m.optimize_restarts(num_restarts=10)

X_new = np.linspace(-0.5, 1, 1000).reshape((1000, 1))
m.plot(X_new, figure_id=1)
plt.show()
