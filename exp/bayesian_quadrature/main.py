import GPy
import numpy as np
import scipy.integrate as integrate
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from synthetic_functions import synthetic_functions
from src_utils import map_and_stack_outputs, slice_column
from matplotlib import pyplot as plt
from utils import bayesian_quadrature, plot_bq_integral_gp_dist, \
    plot_bq_integrand_gp, plot_bq_integrad_truth

np.random.seed(17)

NUM_RESTARTS = 35
FUNCTION_IDX = 0
# KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
KERNEL_FUNCTION = lambda X, Y: GPy.kern.RBF(X.shape[1])

# Define function to evaluate
# custom_func = lambda x: np.exp(-(x ** 2) - np.sin(3 * x) ** 2)
START = 0
END = 1
N_OBS = 10
TITLE = 'N_OBS: {}'.format(N_OBS)

N_PLOT_ROWS = 2
N_PLOT_COLS = 3

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
Y_obs = map_and_stack_outputs(synthetic_functions, X_obs)
curr_gpar_X_obs = None

# Train GPAR model
# m = GPy.models.GPRegression(X_obs, y_single_obs,
#                             KERNEL_FUNCTION(X_obs, X_obs))
gpar_model = GPARRegression(X_obs, Y_obs,
                            KERNEL_FUNCTION, is_zero_noise=True)
gpar_gps = gpar_model.get_gp_dict()
ordering = gpar_model.get_ordering()
gpar_model.print_ordering()

# Train IGP model
igp_model = IGPRegression(X_obs, Y_obs,
                          KERNEL_FUNCTION, is_zero_noise=False)
igp_gps = igp_model.get_gp_models()

# for i in range(Y_obs.shape[1]):
for idx, out_idx in enumerate([ordering[0], ordering[1]]):
    # Set preliminary variables
    m_gpar = gpar_gps[out_idx]
    m_igp = igp_gps[out_idx]
    y_single_obs = slice_column(Y_obs, out_idx)
    custom_func = synthetic_functions[out_idx]
    curr_gpar_X_obs = gpar_model.augment_X(curr_gpar_X_obs, out_idx)

    # Get integral through Bayesian Quadrature
    integral_bq_gpar, integral_std_bq_gpar = \
        bayesian_quadrature(m_gpar, curr_gpar_X_obs, y_single_obs, START, END)
    integral_bq_igp, integral_std_bq_igp = \
        bayesian_quadrature(m_igp, X_obs, y_single_obs, START, END)

    # Approximate integral of function (using standard numerical approach)
    result_base = integrate.quad(custom_func, START, END)
    integral_base = result_base[0]

    # Print numerical indicators
    print('\n--------------- Y{} ---------------'.format(out_idx + 1))
    print(m_gpar)
    print('Parameters: {}'.format(m_gpar.kern.param_array))
    print('Approx value: {}'.format(float(integral_base)))
    print('\nGPAR BQ mean: {}'.format(float(integral_bq_gpar)))
    print('GPAR BQ std: {}'.format(float(integral_std_bq_gpar)))
    print('\nIGP BQ mean: {}'.format(float(integral_bq_igp)))
    print('IGP BQ std: {}'.format(float(integral_std_bq_igp)))

    plt.suptitle(TITLE)

    # Create GP plot
    plt.subplot(N_PLOT_ROWS, N_PLOT_COLS, idx + 1)
    plt.title('Y{}'.format(out_idx + 1))
    plot_bq_integral_gp_dist(integral_base, integral_bq_gpar, integral_std_bq_gpar, 'GPAR Dist')
    plt.axvline(integral_base, color='r', label='Truth', linestyle='--')
    plt.axvline(integral_bq_gpar, color='b', label='GPAR BQ Mean', linestyle='--')
    plt.axvline(integral_bq_igp, color='g', label='IGP BQ Mean', linestyle='--')
    if idx + 1 == N_PLOT_COLS:
        plt.legend(loc='upper left')

    # Create truth vs prediction plot
    plt.subplot(N_PLOT_ROWS, N_PLOT_COLS, idx + 1 + N_PLOT_COLS)
    plot_bq_integrad_truth(custom_func, START, END)
    plot_bq_integrand_gp(gpar_model, START, END, 'GPAR Mean', out_idx, display_var=True)
    plot_bq_integrand_gp(m_igp, START, END, 'IGP Mean', out_idx, display_var=False)
    plt.scatter(X_obs, y_single_obs, s=20, marker='x', color='b', label='Observations')
    if idx + 1 + N_PLOT_COLS == N_PLOT_ROWS * N_PLOT_COLS:
        plt.legend(loc='upper left')

plt.show()
