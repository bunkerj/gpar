import numpy as np
from kernels import get_exponential_decay_kernel
from exp.function_relationships.experiment_runner import ExperimentRunner
from exp.freeze_thaw.model_aggregator import ModelAggregator

N_OBS = 5
N_EPOCHS = 7
NUM_RESTARTS = 35
KERNEL_FUNCTION = get_exponential_decay_kernel

HYP_LIST = [
    (1, 32),
    (2, 64),
    (3, 128)
]

# Train all models
model_aggregator = ModelAggregator(HYP_LIST, N_EPOCHS)
model_aggregator.train_all_models()
losses = model_aggregator.get_all_losses()
Y_true = losses

# Get epochs for truth
X_true = np.arange(1, Y_true.shape[0] + 1).reshape((-1, 1)).astype(float)

# Get observations
X_obs = X_true[0:N_OBS]
Y_obs = Y_true[0:N_OBS, :]

# Get test points
X_test = X_true

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_test, Y_true, KERNEL_FUNCTION, NUM_RESTARTS)
exp.run()
