import numpy as np
from src_utils import map_and_stack_outputs
from synthetic_functions import synthetic_functions, noisy_synthetic_functions


def generate_data():
    # Construct synthetic observations
    n_obs = 50
    X_obs = np.linspace(0, 1, n_obs).reshape((n_obs, 1))
    Y_obs = map_and_stack_outputs(noisy_synthetic_functions, X_obs)

    # Construct true outputs
    n_new = 1000
    X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
    Y_true = map_and_stack_outputs(synthetic_functions, X_new)

    return X_obs, Y_obs, X_new, Y_true
