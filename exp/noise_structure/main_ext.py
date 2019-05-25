import numpy as np
from synthetic_data_functions import y_exp2
from kernels import get_non_linear_input_dependent_kernel
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from src_utils import stack_all_columns
from utils import get_split_outputs, plot_noise_histogram, get_igp_output_samples

NUM_RESTARTS = 35
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
START = 0
END = 2

n = 100000
x_values = [0.2, 0.5, 1.7]

for out_idx in [0, 1, 2]:
    # Construct observations
    n_obs = 200
    X_obs = np.linspace(START, END, n_obs).reshape((n_obs, 1))
    Y_obs = y_exp2(X_obs, is_noisy=True)
    split_output_obs = get_split_outputs(Y_obs)

    # Set up GPAR model
    m = GPARRegression(X_obs, split_output_obs[out_idx],
                       KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
    ordering = m.get_ordering()

    plt.figure(out_idx)
    plt.suptitle('Y{}'.format(out_idx + 1))

    # Print noise structure
    for input_idx in range(len(x_values)):
        # Get true values
        x_value = np.array(x_values[input_idx]).reshape((1, 1))
        X = float(x_value) * np.ones((n, 1))
        Y = y_exp2(X, True)
        plot_idx = input_idx
        title = 'x={}'.format(float(x_value))

        plot_noise_histogram(3, len(x_values), plot_idx,
                             Y[:, 0], Y[:, out_idx + 1], title)

        # Get GPAR predictions
        base_out_idx = 0 if ordering[0] == 0 else (out_idx + 1)
        X_new = stack_all_columns([X, Y[:, base_out_idx]])
        gpar_means, gpar_vars = \
            m.predict_single_output(X_new, ordering[1])
        gpar_samples = np.random.normal(gpar_means, np.sqrt(gpar_vars))

        Y1_gpar = Y[:, 0] if ordering[0] == 0 else gpar_samples.flatten()
        Y2_gpar = gpar_samples.flatten() if ordering[0] == 0 else Y[:, base_out_idx]
        plot_noise_histogram(3, len(x_values), plot_idx + len(x_values),
                             Y1_gpar, Y2_gpar)

        # Get IGP predictions
        Y1_igp = get_igp_output_samples(X_obs, Y_obs, x_value,
                                        KERNEL_FUNCTION, NUM_RESTARTS, 0, n)
        Y2_igp = get_igp_output_samples(X_obs, Y_obs, x_value,
                                        KERNEL_FUNCTION, NUM_RESTARTS, out_idx + 1, n)

        plot_noise_histogram(3, len(x_values), plot_idx + 2 * len(x_values),
                             Y1_igp, Y2_igp)

plt.show()
