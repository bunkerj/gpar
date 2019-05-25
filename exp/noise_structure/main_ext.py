import numpy as np
from synthetic_data_functions import y_exp2
from kernels import get_linear_input_dependent_kernel
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from src_utils import stack_all_columns
from utils import get_split_outputs, plot_noise_histogram, \
    get_igp_output_samples, get_gpar_output_samples

START = 0
END = 2
N_OBS = 150
N_NEW = 100000
NUM_RESTARTS = 50
X_VALUES = [0.2, 0.5, 1.7]
KERNEL_FUNCTION = get_linear_input_dependent_kernel

for scheme_idx in [0, 1, 2]:
    # Construct observations
    X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
    Y_obs = y_exp2(X_obs, is_noisy=True)
    split_output_obs = get_split_outputs(Y_obs)  # Y1 is paired with other Y values

    # Set up GPAR model
    m = GPARRegression(X_obs, split_output_obs[scheme_idx],
                       KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
    ordering = m.get_ordering()

    plt.figure(scheme_idx)
    plt.suptitle('Scheme {}'.format(scheme_idx + 1))

    # Print noise structure
    for input_idx in range(len(X_VALUES)):
        # Get true values
        x_value = np.array(X_VALUES[input_idx]).reshape((1, 1))
        X_new = float(x_value) * np.ones((N_NEW, 1))
        Y = y_exp2(X_new, True)
        plot_idx = input_idx
        title = 'x={}'.format(float(x_value))

        plot_noise_histogram(3, len(X_VALUES), plot_idx,
                             Y[:, 0], Y[:, scheme_idx + 1], title)

        # Get GPAR predictions
        indep_out_idx = 0 if ordering[0] == 0 else (scheme_idx + 1)

        gpar_means, gpar_vars = m.predict_single_output(X_new, indep_out_idx)
        Y_indep = get_gpar_output_samples(gpar_means, gpar_vars)

        X_stacked = stack_all_columns([X_new, Y_indep])
        gpar_means, gpar_vars = m.predict_single_output(X_stacked, ordering[1])
        Y_dep = get_gpar_output_samples(gpar_means, gpar_vars)

        Y1_gpar = Y_indep if ordering[0] == 0 else Y_dep
        Y2_gpar = Y_dep if ordering[0] == 0 else Y_indep

        plot_noise_histogram(3, len(X_VALUES), plot_idx + len(X_VALUES), Y1_gpar, Y2_gpar)

        # Get IGP predictions
        Y1_igp = get_igp_output_samples(X_obs, Y_obs, x_value,
                                        KERNEL_FUNCTION, NUM_RESTARTS, 0, N_NEW)
        Y2_igp = get_igp_output_samples(X_obs, Y_obs, x_value,
                                        KERNEL_FUNCTION, NUM_RESTARTS, scheme_idx + 1, N_NEW)

        plot_noise_histogram(3, len(X_VALUES), plot_idx + 2 * len(X_VALUES),
                             Y1_igp, Y2_igp)

plt.show()
