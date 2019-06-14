import GPy
import numpy as np
from experiment_runner import ExperimentRunner
from utils import get_visible_index_bool
from data_loader import get_processed_data

np.random.seed(17)


def custom_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    Y_dim = current_X.shape[1] - X_dim
    k1 = GPy.kern.RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
    k_linear = GPy.kern.Linear(input_dim=X_dim, active_dims=list(range(X_dim)))
    if Y_dim > 0:
        k2 = GPy.kern.RatQuad(input_dim=X_dim, active_dims=list(range(X_dim)))
        return k1 + k_linear + k2
    return k1 + k_linear


NUM_RESTARTS = 35
NUM_INDUCING = 100
PERCENT_VISIBLE = 30
KERNEL_FUNCTION = custom_kernel
DATA_SRC = ['GOOG', 'NDXT']

X_true, Y_true, labels = get_processed_data(DATA_SRC)

# Construct observations
visible_index_bool = get_visible_index_bool(X_true.shape[0], PERCENT_VISIBLE)

for i in range(500, 801):
    visible_index_bool[i] = False

X_obs = X_true[visible_index_bool]
Y_obs = Y_true[visible_index_bool]

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_true, Y_true, KERNEL_FUNCTION,
                       num_restarts=NUM_RESTARTS, labels=labels)
exp.run()
