import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from src.src_utils import map_and_stack_outputs
from exp.function_relationships.experiment_runner import ExperimentRunner
from src.synthetic_functions import synthetic_functions, noisy_synthetic_functions
from src.kernels import get_non_linear_input_dependent_kernel

NUM_RESTARTS = 100
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n_obs = 50
X_obs = np.linspace(0, 1, n_obs).reshape((n_obs, 1))
Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# Run baseline experiment
all_orders = permutations(range(Y_obs.shape[1]))
for idx, order in enumerate(all_orders):
    print('{} --- {}'.format(idx, order))
    exp = ExperimentRunner(X_obs, Y_obs, X_new, Y_true,
                           KERNEL_FUNCTION, NUM_RESTARTS,
                           figure_start=2 * idx, manual_ordering=order)
    exp.run()

plt.show()
