import numpy as np
from src_utils import map_and_stack_outputs
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from igp_regression import IGPRegression
from kernels import get_non_linear_input_dependent_kernel
from synthetic_data_functions import synthetic_functions, noisy_synthetic_functions
from utils import plot_mse_values, plot_all_outputs

np.random.seed(17)

NUM_RESTARTS = 35
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n = 50
X_obs = np.linspace(0, 1, n).reshape((n, 1))
Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# Get GPAR predictions
gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
gpar_model.print_ordering()
gpar_means, gpar_vars = gpar_model.predict(X_new)

# Get IGP predictions
igp_model = IGPRegression(X_obs, Y_obs, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
igp_means, igp_vars = igp_model.predict(X_new)

# Display results
plot_mse_values(gpar_means, igp_means, Y_true, figure_id_start=0)
plot_all_outputs(gpar_means, gpar_vars, igp_means, igp_vars,
                 X_new, Y_true, X_obs, Y_obs, figure_id_start=1)
plt.show()
