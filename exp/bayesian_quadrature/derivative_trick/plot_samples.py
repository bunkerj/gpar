import numpy as np
from matplotlib import pyplot as plt
from src.kernels import ExponentialDecay
from exp.bayesian_quadrature.derivative_trick.kernels import RBF_dd
from gpflow.kernels import Matern12, Matern32, Matern52, \
    RBF, Constant, Linear, Cosine, RationalQuadratic


def plot_kernel_sample(k, xmin=0, xmax=6):
    xx = np.linspace(xmin, xmax, 100)[:, None]
    K = k.compute_K_symm(xx)
    plt.plot(xx, np.random.multivariate_normal(np.zeros(100), K, N_SAMPLES).T)
    plt.title(k.__class__.__name__)


N_SAMPLES = 5
NROWS = 1
NCOLS = 1

kernels = [ExponentialDecay]

assert len(kernels) == NROWS * NCOLS

plt.subplots(NROWS, NCOLS, figsize=(12, 6))

for idx, kernel in enumerate(kernels):
    row = idx // NCOLS
    col = idx % NCOLS
    plt.subplot(NROWS, NCOLS, idx + 1)
    plot_kernel_sample(kernel())

plt.show()
