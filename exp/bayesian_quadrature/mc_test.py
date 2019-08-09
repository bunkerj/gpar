import pickle
import numpy as np
import scipy.integrate as integrate
from src.kernels import get_full_rbf_kernel
from src.regression.igp_regression import IGPRegression
from exp.bayesian_quadrature.constants import RESULTS_DIR
from src.synthetic_functions import bessel_functions, gaussian_functions
from src.src_utils import map_and_stack_outputs, slice_column
from exp.bayesian_quadrature.bayesian_quadrature import BayesianQuadrature

ROOT_NAME = 'k4'
N_OBS = 50
FULL_NAME = '{}_{}.pickle'.format(ROOT_NAME, N_OBS)
NUM_RESTARTS = 100
KERNEL_FUNCTION = get_full_rbf_kernel
START = 0
END = 40
FUNCTIONS = bessel_functions

N_PLOT_ROWS = 3
N_PLOT_COLS = 3

N_SAMPLES_LIST = [5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]

# Approximate integral of function (using standard numerical approach)
custom_func = FUNCTIONS[0]
result_base = integrate.quad(custom_func, START, END)
integral_base = result_base[0]
print('Approx value: {}'.format(float(integral_base)))

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
Y_obs = map_and_stack_outputs((custom_func,), X_obs)
y_single_obs = slice_column(Y_obs, 0)
curr_gpar_X_obs = None

# Train IGP model
igp_model = IGPRegression(X_obs, Y_obs, KERNEL_FUNCTION,
                          num_restarts=NUM_RESTARTS,
                          is_zero_noise=True)
igp_gps = igp_model.get_gp_models()

means = []
stds = []

for n_samples in N_SAMPLES_LIST:
    igp_bq = BayesianQuadrature(igp_model, is_monte_carlo=True, n_samples=n_samples)

    print('Computing IGP integral ({} samples)...'.format(n_samples))
    integral_bq_igp, integral_std_bq_igp = \
        igp_bq.predict(igp_gps[0], X_obs, y_single_obs, START, END)

    means.append(float(integral_bq_igp))
    stds.append(float(integral_std_bq_igp))

print(means)
print(stds)

with open(RESULTS_DIR + FULL_NAME, 'wb') as file:
    results = {
        'index': N_SAMPLES_LIST,
        'means': means,
        'truth': float(integral_base),
    }
    pickle.dump(results, file)
