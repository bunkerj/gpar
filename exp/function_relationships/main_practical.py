import numpy as np
from data_loader import get_processed_data
from kernels import get_non_linear_input_dependent_kernel
from exp.function_relationships.utils import get_visible_index_bool
from exp.function_relationships.experiment_runner import ExperimentRunner

np.random.seed(17)

NUM_RESTARTS = 0
NUM_INDUCING = 100
PERCENT_VISIBLE = 30
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
DATA_SRC = ['GOOG', 'NDXT', 'NQNA']

X_true, Y_true, labels = get_processed_data(DATA_SRC)

# Construct observations
visible_index_bool = get_visible_index_bool(X_true.shape[0], PERCENT_VISIBLE)

for i in range(500, 801):
    visible_index_bool[i] = False

X_obs = X_true[visible_index_bool]
Y_obs = Y_true[visible_index_bool]

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_true, Y_true, KERNEL_FUNCTION,
                       num_restarts=NUM_RESTARTS, labels=labels,
                       num_inducing=NUM_INDUCING)
exp.run()
