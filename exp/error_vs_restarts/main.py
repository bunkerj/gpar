import os
import pickle
import numpy as np
from kernels import get_full_rbf_kernel, get_non_linear_input_dependent_kernel
from src_utils import map_and_stack_outputs
from utils import get_total_smse_values
from constants import OUTPUTS_PATH, NUM_RESTARTS_VALUES_PATH
from synthetic_functions import \
    low_complexity_functions, noisy_low_complexity_functions, \
    medium_complexity_functions, noisy_medium_complexity_functions, \
    high_complexity_functions, noisy_high_complexity_functions

NUM_AVG_SAMPLES = 20

AGGREGATE_FUNCTIONS_DICT = {
    'low_rbf': (low_complexity_functions, noisy_low_complexity_functions, get_full_rbf_kernel),
    'medium_rbf': (medium_complexity_functions, noisy_medium_complexity_functions, get_full_rbf_kernel),
    'high_rbf': (high_complexity_functions, noisy_high_complexity_functions, get_full_rbf_kernel),
    'low_custom_kernel': (
        low_complexity_functions, noisy_low_complexity_functions, get_non_linear_input_dependent_kernel),
    'medium_custom_kernel': (
        medium_complexity_functions, noisy_medium_complexity_functions, get_non_linear_input_dependent_kernel),
    'high_custom_kernel': (
        high_complexity_functions, noisy_high_complexity_functions, get_non_linear_input_dependent_kernel)
}

for i, key in enumerate(AGGREGATE_FUNCTIONS_DICT.keys()):
    true_functions, noisy_functions, kernel = AGGREGATE_FUNCTIONS_DICT[key]
    filename = '{}{}_{}.pickle'.format(OUTPUTS_PATH, i, key)

    # Construct synthetic observations
    n = 50
    X_obs = np.linspace(0, 5, n).reshape((n, 1))
    Y_obs = map_and_stack_outputs(noisy_functions, X_obs)

    # Construct true outputs
    n_new = 1000
    X_new = np.linspace(0, 5, n_new).reshape((n_new, 1))
    Y_true = map_and_stack_outputs(true_functions, X_new)

    num_restarts_values = np.array([1, 5, 10, 20, 40, 50]).reshape((-1, 1))

    total_mse_values = get_total_smse_values(
        X_obs, Y_obs, X_new, Y_true, kernel,
        num_restarts_values, NUM_AVG_SAMPLES)

    if not os.path.exists(NUM_RESTARTS_VALUES_PATH):
        with open(NUM_RESTARTS_VALUES_PATH, 'wb') as file:
            pickle.dump(num_restarts_values, file)

    with open(filename, 'wb') as file:
        pickle.dump(total_mse_values, file)
