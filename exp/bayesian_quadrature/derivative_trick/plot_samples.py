import gpflow
import numpy as np
from kernels import RBF_dd
from matplotlib import pyplot as plt
from gpflow.kernels import Matern12, Matern32, Matern52, \
    RBF, Constant, Linear, Cosine


def plot_kernel_sample(k, ax, xmin=0, xmax=6):
    xx = np.linspace(xmin, xmax, 100)[:, None]
    K = k.compute_K_symm(xx)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
    ax.set_title(k.__class__.__name__)


NROWS = 2
NCOLS = 4

kernels = [Matern12, Matern32, Matern52, RBF,
           Constant, Linear, Cosine, RBF_dd]

assert len(kernels) == NROWS * NCOLS

f, axes = plt.subplots(NROWS, NCOLS, figsize=(12, 6))

for idx, kernel in enumerate(kernels):
    row = idx // NCOLS
    col = idx % NCOLS
    plot_kernel_sample(kernel(1), axes[row, col])

axes[0, 0].set_ylim(-3, 3)

plt.show()
