import numpy as np
from src_utils import map_and_stack_outputs
from experiment_runner import ExperimentRunner
from kernels import get_non_linear_input_dependent_kernel
from synthetic_functions import synthetic_functions, noisy_synthetic_functions

np.random.seed(17)

NUM_RESTARTS = 35
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n_obs = 50
X_obs = np.linspace(0, 1, n_obs).reshape((n_obs, 1))
Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_new, Y_true, KERNEL_FUNCTION, NUM_RESTARTS)
exp.run()
