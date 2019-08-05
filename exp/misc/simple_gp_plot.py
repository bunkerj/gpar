import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.kern import RatQuad, RBF, Matern32, Linear, Cosine, Brownian


def f(x):
    return 12 * np.sin(1 * x) + 7 * x - 12


def plot_gp_info(m, x_test, subplot_args):
    means, vars = m.predict(x_test)
    ub = means + 2 * np.sqrt(vars)
    lb = means - 2 * np.sqrt(vars)

    plt.subplot(*subplot_args)
    plt.scatter(x_obs, y_obs, marker='x', label='Observations')
    plt.plot(x_test, means, label='GP Mean')
    plt.plot(x_test, y_truth, label='Truth')
    plt.fill_between(
        x_test.flatten(),
        lb.flatten(),
        ub.flatten(),
        alpha=0.2,
        edgecolor='b')
    plt.legend(loc='lower right')
    # plt.title('GP Model {}'.format(subplot_args[-1]))


x_test = np.linspace(-3, 8, 1000).reshape((-1, 1))
# x_obs = np.array([-np.inf]).reshape((-1, 1))
x_obs = np.array([0.5, 2.0, 3.0, 4.5, 5.0]).reshape((-1, 1))
y_truth = f(x_test)
# y_obs = np.array([0]).reshape((-1, 1))
y_obs = f(x_obs)

model_configs = [
    (GPy.mappings.Constant(1, 1, 10), Cosine(1)),
    (GPy.mappings.Constant(1, 1, 4), RBF(1)),
    (GPy.mappings.Constant(1, 1, -3), Matern32(1))
]

for i in range(len(model_configs)):
    mf, kernel = model_configs[i]
    m = GPy.models.GPRegression(x_obs, y_obs, kernel=kernel, mean_function=mf)
    m.optimize_restarts(50)
    plot_gp_info(m, x_test, (1, len(model_configs), i + 1))

plt.show()
