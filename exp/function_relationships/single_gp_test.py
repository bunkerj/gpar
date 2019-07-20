import gpflow
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 3 * np.sin(5 * x) + 2 * x + 10


def randomize_parameters(m):
    trainables = m.read_trainables()
    for key in trainables:
        trainables[key] = np.random.gamma(2, 2)
    m.assign(trainables)


def copy_tensor_dict(tensor_dict):
    copy_dict = {}
    for key in tensor_dict:
        copy_dict[key] = float(tensor_dict[key])
    return copy_dict


def optimize(m, n_restarts):
    max_likelihood = None
    max_trainables = None
    for _ in range(0, n_restarts + 1):
        gpflow.train.ScipyOptimizer().minimize(m)
        current_likelihood = m.compute_log_likelihood()
        if max_likelihood is None or current_likelihood > max_likelihood:
            print('{} > {}'.format(current_likelihood, max_likelihood))
            max_likelihood = current_likelihood
            max_trainables = copy_tensor_dict(m.read_trainables())
        randomize_parameters(m)
    m.assign(max_trainables)


X_obs = np.linspace(0, 10, 25).reshape((-1, 1))
Y_obs = f(X_obs) + np.random.normal(0, 0.1, size=X_obs.shape)

X_test = np.linspace(0, 10, 10000).reshape((-1, 1))
Y_true = f(X_test)

k = gpflow.kernels.RBF(1)
m = gpflow.models.GPR(X_obs, Y_obs, k)

optimize(m, 25)
print(m.compute_log_likelihood())

means, variances = m.predict_f(X_test)
ub = means + 2 * np.sqrt(variances)
lb = means - 2 * np.sqrt(variances)

plt.scatter(X_obs, Y_obs, label='Observations')
plt.plot(X_test, means, label='GP Mean')
plt.fill_between(
    X_test.flatten(),
    lb.flatten(),
    ub.flatten(),
    alpha=0.2,
    edgecolor='b',
    color='b')
plt.plot(X_test, Y_true, label='Truth')
plt.legend(loc='upper right')
plt.grid()
plt.show()
