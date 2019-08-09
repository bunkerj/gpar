import numpy as np
from matplotlib import pyplot as plt
from src.src_utils import map_and_stack_outputs
from exp.function_relationships.experiment_runner import ExperimentRunner
from src.kernels import get_non_linear_input_dependent_kernel, get_full_rbf_kernel
from src.synthetic_functions import gaussian_functions, bessel_functions

NUM_RESTARTS = 1000
KERNEL_FUNCTION = get_full_rbf_kernel
START = -8
END = 0.5
FUNCTIONS = gaussian_functions
N_OBS = 5

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
Y_obs = map_and_stack_outputs(FUNCTIONS, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(START, END, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(FUNCTIONS, X_new)

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_new, Y_true, KERNEL_FUNCTION, NUM_RESTARTS)
exp.run()

plt.show()
