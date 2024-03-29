import scipy.integrate as integrate
from src.regression.gpar_regression import GPARRegression
from src.regression.igp_regression import IGPRegression
from src.synthetic_functions import gaussian_functions
from src.src_utils import map_and_stack_outputs
from src.kernels import full_RBF
from exp.bayesian_quadrature.baseline.bayesian_quadrature import BayesianQuadrature
from exp.bayesian_quadrature.baseline.utils import *

np.random.seed(17)

NUM_RESTARTS = 0
KERNEL_FUNCTION = full_RBF
START = -5
END = 0.5
N_OBS = 50
TITLE = 'Number of Observations: {}'.format(N_OBS)
FUNCTIONS = gaussian_functions

N_PLOT_ROWS = 3
N_PLOT_COLS = 3

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
Y_obs = map_and_stack_outputs(FUNCTIONS, X_obs)
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
    print('Computing GPAR integral for Y{}...'.format(out_idx + 1))
    integral_bq_gpar, integral_std_bq_gpar = \
        gpar_bq.predict_f(m_gpar, curr_gpar_X_obs, y_single_obs, START, END)
    print('Done')
    print('Computing IGP integral for Y{}...'.format(out_idx + 1))
    integral_bq_igp, integral_std_bq_igp = \
        igp_bq.predict_f(m_igp, X_obs, y_single_obs, START, END)
    print('Done')

    # Approximate integral of function (using standard numerical approach)
    custom_func = FUNCTIONS[out_idx]
    result_base = integrate.quad(custom_func, START, END)
    integral_base = result_base[0]

    # Print numerical indicators
    print('\n--------------- Y{} ---------------'.format(out_idx + 1))
    print('Parameters: {}'.format(extract_param_list(m_gpar)))
    print('Approx value: {}'.format(float(integral_base)))
    print('\nGPAR BQ mean: {}'.format(float(integral_bq_gpar)))
    print('GPAR BQ std: {}'.format(float(integral_std_bq_gpar)))
    print('\nIGP BQ mean: {}'.format(float(integral_bq_igp)))
    print('IGP BQ std: {}\n'.format(float(integral_std_bq_igp)))

    plt.suptitle(TITLE)

    # Create GP plot
    plt.subplot(N_PLOT_ROWS, N_PLOT_COLS, idx + 1)
    plt.title('Y{}'.format(out_idx + 1))
    plot_bq_integral_gp_dist(integral_base, integral_bq_gpar, integral_std_bq_gpar, 'GPAR Dist')
    plt.axvline(integral_base, color='r', label='Truth', linestyle='--')
    plt.axvline(integral_bq_gpar, color='b', label='GPAR BQ Mean', linestyle='--')
    plt.axvline(integral_bq_igp, color='g', label='IGP BQ Mean', linestyle='--')
    if idx + 1 == N_PLOT_COLS:
        plt.legend(loc='upper right')

    # Create truth vs prediction plot
    plt.subplot(N_PLOT_ROWS, N_PLOT_COLS, idx + 1 + N_PLOT_COLS)
    plot_bq_integrand_truth(custom_func, START, END)
    plot_bq_integrand_gp(gpar_model, START, END, 'GPAR Mean', out_idx, display_var=True)
    plot_bq_integrand_gp(m_igp, START, END, 'IGP Mean', out_idx, display_var=False)
    plt.scatter(X_obs, y_single_obs, s=20, marker='x', color='b', label='Observations')
    if idx + 1 + N_PLOT_COLS == 2 * N_PLOT_COLS:
        plt.legend(loc='upper right')

    # Create IGP vs GPAR error comparison plots
    plt.subplot(N_PLOT_ROWS, N_PLOT_COLS, idx + 1 + 2 * N_PLOT_COLS)
    abs_error_igp = np.abs(float(integral_base) - float(integral_bq_igp))
    abs_error_gpar = np.abs(float(integral_base) - float(integral_bq_gpar))
    errors = [abs_error_igp, abs_error_gpar]
    labels = ['IGP Abs Error', 'GPAR Abs Error']
    plt.bar(range(len(errors)), errors, tick_label=labels)

plt.show()
