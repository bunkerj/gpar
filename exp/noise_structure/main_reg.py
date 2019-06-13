import numpy as np
from matplotlib import pyplot as plt
from synthetic_functions import y_exp2
from kernels import get_non_linear_input_dependent_kernel
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from utils import plot_noise, get_split_outputs, get_prediction_noise

NUM_RESTARTS = 10
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

START = 0
END = 1

# Construct synthetic outputs
n_true = 100000
X = np.linspace(START, END, n_true).reshape((n_true, 1))
Y_true_noisy = y_exp2(X, is_noisy=True)
Y_true = y_exp2(X, is_noisy=False)
split_output_true = get_split_outputs(Y_true)

# Construct observations
n_obs = 100
X_obs = np.linspace(START, END, n_obs).reshape((n_obs, 1))
Y_obs = y_exp2(X_obs, is_noisy=True)
split_output_obs = get_split_outputs(Y_obs)

# Get outputs at desired locations
Y_new_noisy = y_exp2(X, is_noisy=True)


# Get predictions
def true_predictor(Y_true):
    return Y_true, np.zeros(Y_true.shape)


def gpar_predictor(Y):
    m = GPARRegression(X_obs, Y, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
    return m.predict(X)


def igp_predictor(Y):
    igp_model = IGPRegression(X_obs, Y, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
    return igp_model.predict(X)


noise_true = get_prediction_noise(Y_true_noisy, true_predictor, split_output_true)
noise_gpar = get_prediction_noise(Y_new_noisy, gpar_predictor, split_output_obs)
noise_igp = get_prediction_noise(Y_new_noisy, igp_predictor, split_output_obs)

# Display results
plot_noise(0, X, noise_true, 'Truth')
plot_noise(1, X, noise_gpar, 'GPAR')
plot_noise(2, X, noise_igp, 'IGP')
plt.show()
