import numpy as np
from utils import map_and_stack_outputs
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from visualiser import Visualiser
from kernels import get_non_linear_input_dependent_kernel
from synthetic_data_functions import synthetic_functions, noisy_synthetic_functions

np.random.seed(17)

# Output stream of interest
NUM_RESTARTS = 5
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n = 50
X_obs = np.linspace(0, 1, n).reshape((n, 1))
Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# Get predictions from GPAR
gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
gpar_model.print_ordering()
means, variances = gpar_model.predict(X_new)

# Display results
visualiser = Visualiser(means, variances, X_obs, Y_obs, X_new, Y_true)
visualiser.plot_mse_values(0)
visualiser.plot_all_outputs(1)
plt.show()
