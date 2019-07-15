import numpy as np
from scipy import stats


def basic_func(x):
    return np.sin(2 * x) + x + 15


def gaussian_pdf(x):
    return stats.norm.pdf(x)
