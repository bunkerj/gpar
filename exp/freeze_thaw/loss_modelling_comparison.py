import numpy as np
from exp.freeze_thaw.aggregators.model_aggregator import ModelAggregator
from exp.function_relationships.experiment_runner import ExperimentRunner
from src.kernels import get_exponential_decay_kernel
from src.src_utils import sample_from_bounds, stack_all_columns

N_MODELS = 6
N_EPOCHS = 30
N_OBS = 20
NUM_RESTARTS = 35
KERNEL_FUNCTION = get_exponential_decay_kernel
PLOT_SHAPE = (2, 3)

# Construct all models
bounds_list = [[1, 10], [5, 500]]
hyp_list = [sample_from_bounds(bounds_list) for _ in range(N_MODELS)]

# Train all models
model_aggregator = ModelAggregator(hyp_list)
model_aggregator.train_all_models(N_EPOCHS)
losses = model_aggregator.get_all_losses()
Y_true = stack_all_columns(losses.values())

# Get epochs for truth
X_true = np.arange(1, Y_true.shape[0] + 1) \
    .reshape((-1, 1)).astype(float)

# Get observations
X_obs = X_true[:N_OBS]
Y_obs = Y_true[:N_OBS, :]

# Get test points
X_test = X_true

# Run experiment
exp = ExperimentRunner(X_obs, Y_obs, X_test, Y_true,
                       KERNEL_FUNCTION, NUM_RESTARTS,
                       plot_shape=PLOT_SHAPE,
                       legend_loc='upper right')
exp.run()
