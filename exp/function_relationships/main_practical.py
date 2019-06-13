import numpy as np
from matplotlib import pyplot as plt
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from kernels import get_non_linear_input_dependent_kernel
from utils import plot_mse_values, plot_all_outputs
from src_utils import load_data_from_csv, normalize_data, \
    get_visible_index_bool, stack_all_columns

np.random.seed(17)

NUM_RESTARTS = 35
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

goog_data = load_data_from_csv('GOOG_data.csv')
ndxt_data = load_data_from_csv('NDXT_data.csv')
nqna_data = load_data_from_csv('NQNA_data.csv')

index = goog_data['close'].index.values.reshape((len(goog_data), 1))
visible_index_bool = get_visible_index_bool(index, 0.5)

goog_close = normalize_data(goog_data['close']).values.reshape((len(index), 1))
ndxt_close = normalize_data(ndxt_data['close']).values.reshape((len(index), 1))
nqna_close = normalize_data(nqna_data['close']).values.reshape((len(index), 1))

# Construct synthetic observations
n_obs = sum(visible_index_bool)
goog_close_visible = goog_close[visible_index_bool].reshape((n_obs, 1))
ndxt_close_visible = ndxt_close[visible_index_bool].reshape((n_obs, 1))
nqna_close_visible = nqna_close[visible_index_bool].reshape((n_obs, 1))
X_obs = index[visible_index_bool].reshape((n_obs, 1))
Y_obs = stack_all_columns((goog_close_visible, ndxt_close_visible, nqna_close_visible))

# Construct true outputs
n_new = len(index)
X_new = index
Y_true = stack_all_columns((goog_close, ndxt_close, nqna_close))

# Get GPAR predictions
gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION,
                            num_restarts=NUM_RESTARTS, num_inducing=150)
gpar_model.print_ordering()
gpar_means, gpar_vars = gpar_model.predict(X_new)

# Get IGP predictions
igp_model = IGPRegression(X_obs, Y_obs, KERNEL_FUNCTION,
                          num_restarts=NUM_RESTARTS, num_inducing=150)
igp_means, igp_vars = igp_model.predict(X_new)

# Display results
plot_mse_values(gpar_means, igp_means, Y_true, figure_id_start=0)
plot_all_outputs(gpar_means, gpar_vars, igp_means, igp_vars,
                 X_new, Y_true, X_obs, Y_obs, figure_id_start=1)
plt.show()
