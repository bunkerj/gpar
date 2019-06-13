import numpy as np
from experiment_runner import ExperimentRunner
from kernels import get_non_linear_input_dependent_kernel
from utils import get_visible_index_bool
from data_loader import get_processed_data

np.random.seed(17)

NUM_RESTARTS = 10
NUM_INDUCING = 100
PERCENT_VISIBLE = 5
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
DATA_SRC = ['GOOG', 'NDXT']

X_true, Y_true, labels = get_processed_data(DATA_SRC)

# Construct observations
visible_index_bool = get_visible_index_bool(X_true.shape[0], PERCENT_VISIBLE)
X_obs = X_true[visible_index_bool]
Y_obs = Y_true[visible_index_bool]

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_true, Y_true, KERNEL_FUNCTION, NUM_RESTARTS)
exp.run()
