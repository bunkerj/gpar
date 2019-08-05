import os
import pickle
import numpy as np
from kernels import get_full_rbf_kernel
from utils import get_total_smse_values_and_ordering_index
from src_utils import map_and_stack_outputs
from constants import NUM_RESTARTS_VALUES_PATH
from synthetic_functions import \
    low_complexity_functions, noisy_low_complexity_functions, \
    medium_complexity_functions, noisy_medium_complexity_functions, \
    high_complexity_functions, noisy_high_complexity_functions

KERNEL_FUNCTION = get_full_rbf_kernel
NUM_AVG_SAMPLES = 1

AGGREGATE_FUNCTIONS_DICT = {
    'low': (low_complexity_functions, noisy_low_complexity_functions),
    'medium': (medium_complexity_functions, noisy_medium_complexity_functions),
    'high': (high_complexity_functions, noisy_high_complexity_functions)
}

for key in AGGREGATE_FUNCTIONS_DICT.keys():
    true_functions, noisy_functions = AGGREGATE_FUNCTIONS_DICT[key]
    filename = 'results/outputs/{}_complexity.pickle'.format(key)

    # Construct synthetic observations
    n = 50
    X_obs = np.linspace(0, 5, n).reshape((n, 1))
    Y_obs = map_and_stack_outputs(noisy_functions, X_obs)

    # Construct true outputs
    n_new = 1000
    X_new = np.linspace(0, 5, n_new).reshape((n_new, 1))
    Y_true = map_and_stack_outputs(true_functions, X_new)

    num_restarts_values = np.array([1, 5, 10, 20, 40, 50]).reshape((-1, 1))

    total_mse_values = get_total_smse_values_and_ordering_index(
        X_obs, Y_obs, X_new, Y_true, KERNEL_FUNCTION,
        num_restarts_values, NUM_AVG_SAMPLES)

    if not os.path.exists(NUM_RESTARTS_VALUES_PATH):
        with open(NUM_RESTARTS_VALUES_PATH, 'wb') as file:
            pickle.dump(num_restarts_values, file)

    with open(filename, 'wb') as file:
        pickle.dump(total_mse_values, file)
