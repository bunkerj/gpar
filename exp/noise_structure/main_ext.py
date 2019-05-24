import numpy as np
from synthetic_data_functions import y_exp2
from kernels import get_non_linear_input_dependent_kernel
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from src_utils import stack_all_columns
from utils import get_split_outputs, plot_noise_histogram

NUM_RESTARTS = 50
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
START = 0
END = 2

n = 100000
x_values = [0.2, 0.5, 1.7]

for out_idx in [0]:
    # Construct observations
    n_obs = 200
    X_obs = np.linspace(START, END, n_obs).reshape((n_obs, 1))
    Y_obs = y_exp2(X_obs, is_noisy=True)
    split_output_obs = get_split_outputs(Y_obs)

    # Set up GPAR model
    m = GPARRegression(X_obs, split_output_obs[out_idx],
                       KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
    ordering = m.get_ordering()
    print(ordering)

    # Print noise structure
    for input_idx in range(len(x_values)):
        x_value = x_values[input_idx]
        X = x_value * np.ones((n, 1))
        Y = y_exp2(X, True)
        plot_idx = input_idx

        plot_noise_histogram(2, len(x_values), plot_idx,
                             Y[:, 0], Y[:, out_idx + 1], x_value)

        base_out_idx = 0 if ordering[0] == 0 else (out_idx + 1)
        X_new = stack_all_columns([X, Y[:, base_out_idx]])
        gpar_means, gpar_vars = \
            m.predict_single_output(X_new, ordering[1])
        gpar_samples = np.random.normal(gpar_means, np.sqrt(gpar_vars))

        Y1 = Y[:, 0] if ordering[0] == 0 else gpar_samples.flatten()
        Y2 = gpar_samples.flatten() if ordering[0] == 0 else Y[:, base_out_idx]
        plot_noise_histogram(2, len(x_values), plot_idx + len(x_values),
                             Y1, Y2, x_value)

plt.show()

# def gaussian_pdf(x, m, v):
#     norm = 1 / np.sqrt(2 * np.pi * v)
#     coeff = -1 / (2 * v)
#     return norm * np.exp(coeff * (x - m) ** 2)
#
#
# xlist = np.linspace(-3, 3, 100)
# ylist = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(xlist, ylist)
# Z = gaussian_pdf(X, 0, 4) * gaussian_pdf(Y, 0, 5)
# plt.contourf(X, Y, Z)
# plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
