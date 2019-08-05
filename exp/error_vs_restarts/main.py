import pickle
import numpy as np
from matplotlib import pyplot as plt
from kernels import get_full_rbf_kernel
from utils import plot_error_vs_restarts, get_total_smse_values_and_ordering_index
from src_utils import map_and_stack_outputs
from synthetic_functions import low_complexity_functions, noisy_low_complexity_functions

KERNEL_FUNCTION = get_full_rbf_kernel

TRUE_FUNCTIONS = low_complexity_functions
NOISY_FUNCTIONS = noisy_low_complexity_functions
NUM_AVG_SAMPLES = 5
FILENAME = '../../data/low_complexity.pickle'
IS_LOADING = False

# Construct synthetic observations
n = 50
X_obs = np.linspace(0, 5, n).reshape((n, 1))
Y_obs = map_and_stack_outputs(NOISY_FUNCTIONS, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 5, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(TRUE_FUNCTIONS, X_new)

# num_restarts_list = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]
num_restarts_list = [1, 5]
num_restarts_values = np.array(num_restarts_list) \
    .reshape((len(num_restarts_list), 1))

if IS_LOADING:
    with open(FILENAME, 'rb') as file:
        total_mse_values = pickle.load(file)
else:
    total_mse_values = get_total_smse_values_and_ordering_index(
        X_obs, Y_obs, X_new, Y_true, KERNEL_FUNCTION, num_restarts_values, NUM_AVG_SAMPLES)
    with open(FILENAME, 'wb') as file:
        pickle.dump(total_mse_values, file)

plot_error_vs_restarts(total_mse_values, num_restarts_values)
plt.show()
