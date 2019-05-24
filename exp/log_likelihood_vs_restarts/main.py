import numpy as np
from matplotlib import pyplot as plt
from kernels import get_non_linear_input_dependent_kernel
from utils import plot_log_likelihood_vs_restarts, get_total_mse_values_and_ordering_index
from src_utils import map_and_stack_outputs
from synthetic_data_functions import synthetic_functions, noisy_synthetic_functions

KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n = 50
X_obs = np.linspace(0, 1, n).reshape((n, 1))
Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# num_restarts_list = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]
num_restarts_list = [1, 5, 10, 20, 35]
num_restarts_values = np.array(num_restarts_list).reshape((len(num_restarts_list), 1))

total_mse_values, correct_order_index = get_total_mse_values_and_ordering_index(
    X_obs, Y_obs, X_new, Y_true, KERNEL_FUNCTION, num_restarts_values)

plot_log_likelihood_vs_restarts(total_mse_values, correct_order_index, num_restarts_values)
plt.show()