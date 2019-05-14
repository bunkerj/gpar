import numpy as np
from utils import slice_column
from matplotlib import pyplot as plt
from synthetic_data_functions import y_exp2_clean
from kernels import get_non_linear_input_dependent_kernel
from gpar_regressor import GPARRegression

NUM_RESTARTS = 200
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel


def plot_noise(figure_id_start, X, noise):
    plt.figure(figure_id_start)
    n_plots = noise.shape[1] - 1
    for idx in range(1, n_plots + 1):
        plt.subplot(1, 3, idx)
        plt.scatter(slice_column(noise, 0),
                    slice_column(noise, idx), s=1, c=X, cmap='magma')


# Construct synthetic outputs
n_true = 100000
X_true = np.linspace(0, 1, n_true).reshape((n_true, 1))
Y_true_noisy = y_exp2_clean(X_true, is_noisy=True)
Y_true = y_exp2_clean(X_true, is_noisy=False)
noise_true = Y_true_noisy - Y_true

# Get predictions from GPAR
n_obs = 100
X_obs = np.linspace(0, 1, n_obs).reshape((n_obs, 1))
Y_obs = y_exp2_clean(X_obs, is_noisy=True)

n_new = 100000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
Y_new = y_exp2_clean(X_new, is_noisy=True)

gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
means, variances = gpar_model.predict(X_new)
noise_model = Y_new - means

# Display results
plot_noise(0, X_new, noise_true)
plot_noise(1, X_new, noise_model)
plt.show()
