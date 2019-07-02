import GPy
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from regression.gpar_regression import GPARRegression
from regression.igp_regression import IGPRegression
from synthetic_functions import synthetic_functions, gaussian_functions
from src_utils import map_and_stack_outputs, slice_column
from bayesian_quadrature import BayesianQuadrature
from kernels import get_non_linear_input_dependent_kernel
from utils import plot_bq_integral_gp_dist, \
    plot_bq_integrand_gp, plot_bq_integrand_truth

np.random.seed(17)

NUM_RESTARTS = 35
FUNCTION_IDX = 0
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
START = -5
END = 2
N_OBS = 15
TITLE = 'N_OBS: {}'.format(N_OBS)

N_PLOT_ROWS = 2
N_PLOT_COLS = 3

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
Y_obs = map_and_stack_outputs(gaussian_functions, X_obs)
curr_gpar_X_obs = None

# Train GPAR model
gpar_model = GPARRegression(X_obs, Y_obs, KERNEL_FUNCTION,
                            num_restarts=NUM_RESTARTS, is_zero_noise=True)
gpar_gps = gpar_model.get_gp_dict()
ordering = gpar_model.get_ordering()
gpar_bq = BayesianQuadrature(gpar_model)

# Train IGP model
igp_model = IGPRegression(X_obs, Y_obs, KERNEL_FUNCTION,
                          num_restarts=NUM_RESTARTS, is_zero_noise=True)
igp_gps = igp_model.get_gp_models()
igp_bq = BayesianQuadrature(igp_model)

for idx, out_idx in enumerate(ordering):
    # Set preliminary variables
    m_gpar = gpar_gps[out_idx]
    m_igp = igp_gps[out_idx]
    y_single_obs = slice_column(Y_obs, out_idx)
    curr_gpar_X_obs = gpar_model.augment_X(curr_gpar_X_obs, out_idx)

    # Get integral through Bayesian Quadrature
    integral_bq_gpar, integral_std_bq_gpar = \
        gpar_bq.predict(m_gpar, curr_gpar_X_obs, y_single_obs, START, END)
    integral_bq_igp, integral_std_bq_igp = \
        igp_bq.predict(m_igp, X_obs, y_single_obs, START, END)

    # Approximate integral of function (using standard numerical approach)
    custom_func = gaussian_functions[out_idx]
    result_base = integrate.quad(custom_func, START, END)
    integral_base = result_base[0]

    # Print numerical indicators
    print('\n--------------- Y{} ---------------'.format(out_idx + 1))
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
    plot_bq_integrand_truth(custom_func, START, END)
    plot_bq_integrand_gp(gpar_model, START, END, 'GPAR Mean', out_idx, display_var=True)
    plot_bq_integrand_gp(m_igp, START, END, 'IGP Mean', out_idx, display_var=False)
    plt.scatter(X_obs, y_single_obs, s=20, marker='x', color='b', label='Observations')
    if idx + 1 + N_PLOT_COLS == N_PLOT_ROWS * N_PLOT_COLS:
        plt.legend(loc='upper left')

plt.show()
