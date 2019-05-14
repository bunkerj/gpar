import numpy as np
from utils import *
from synthetic_data_functions import y_scheme2_exp2
from matplotlib import pyplot as plt

# Construct synthetic outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = y_scheme2_exp2(X_new)
