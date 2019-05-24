import numpy as np
from matplotlib import pyplot as plt
from synthetic_data_functions import y_exp2_clean
from kernels import get_non_linear_input_dependent_kernel
from gpar_regressor import GPARRegression
from utils import plot_noise, get_igp_predictions
from src_utils import slice_column

NUM_RESTARTS = 10
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

START = 0
END = 1

# Construct synthetic outputs
n_true = 100000
X = np.linspace(START, END, n_true).reshape((n_true, 1))
Y_true_noisy = y_exp2_clean(X, is_noisy=True)
Y_true = y_exp2_clean(X, is_noisy=False)
noise_true = Y_true_noisy - Y_true

# Construct observations
n_obs = 100
X_obs = np.linspace(START, END, n_obs).reshape((n_obs, 1))
Y_obs = y_exp2_clean(X_obs, is_noisy=True)

# Construct get outputs at desired locations
Y_new = y_exp2_clean(X, is_noisy=True)

# Get predictions from GPAR
gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
means_gpar, variances_gpar = gpar_model.predict(X)
noise_gpar = Y_new - means_gpar

# Get predictions from IGP
means_igp, variances_igp = get_igp_predictions(X_obs, Y_obs, X, KERNEL_FUNCTION, NUM_RESTARTS)
noise_igp = Y_new - means_igp

# Display results
plot_noise(0, X, noise_true, 'Truth')
plot_noise(1, X, noise_gpar, 'GPAR')
plot_noise(2, X, noise_igp, 'IGP')
plt.show()
