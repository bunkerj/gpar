import GPy
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
from kernels import get_non_linear_input_dependent_kernel
from synthetic_data_functions import y3_exp1
from matplotlib import pyplot as plt
from utils import bayesian_quadrature

np.random.seed(17)

NUM_RESTARTS = 10
FUNCTION_IDX = 0
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Define function to evaluate
custom_func = lambda x: y3_exp1(x)
START = 0
END = 1
N_OBS = 5

# Construct synthetic observations
X_obs = np.linspace(START, END, N_OBS).reshape((N_OBS, 1))
y_single_obs = custom_func(X_obs)

# Train GP model
m = GPy.models.GPRegression(X_obs, y_single_obs,
                            KERNEL_FUNCTION(X_obs, X_obs), noise_var=0.001)
# m = GPy.models.GPRegression(X_obs, y_single_obs,
#                             GPy.kern.RBF(1), noise_var=0.001)
m.optimize_restarts(NUM_RESTARTS, verbose=False)

# Get integral through Bayesian Quadrature
integral_bq, integral_std_bq = bayesian_quadrature(m, X_obs, y_single_obs, START, END)

# Approximate integral of function (using standard numerical approach)
result_base = integrate.quad(custom_func, START, END)
integral_base = result_base[0]

# Print numerical indicators
print('Approximated value: {}'.format(float(integral_base)))
print('Estimated mean: {}'.format(float(integral_bq)))
print('Estimated std: {}'.format(float(integral_std_bq)))

# Get True values
n_new = 1000
X_new = np.linspace(START, END, n_new).reshape((n_new, 1))
y_single_means_pred, y_single_vars_pred = m.predict(X_new)
ub_means_pred = y_single_means_pred + 2 * np.sqrt(y_single_vars_pred)
lb_means_pred = y_single_means_pred - 2 * np.sqrt(y_single_vars_pred)
y_single_means_true = custom_func(X_new)

# Create GP plot
plt.subplot(1, 2, 1)
ub = np.max((integral_base + 2, integral_bq + 3 * integral_std_bq))
lb = np.min((integral_base - 2, integral_bq - 3 * integral_std_bq))
x_gauss = np.linspace(lb, ub, 1000).flatten()
y_gauss = stats.norm.pdf(x_gauss, integral_bq, integral_std_bq).flatten()
plt.plot(x_gauss, y_gauss)
plt.axvline(integral_base, color='r')

# Create truth vs prediction plot
plt.subplot(1, 2, 2)
plt.plot(X_new, y_single_means_true, label='Truth')
plt.plot(X_new, y_single_means_pred, label='Predictions')
plt.fill_between(
    X_new.flatten(),
    lb_means_pred.flatten(),
    ub_means_pred.flatten(),
    alpha=0.2,
    edgecolor='b')
plt.legend(loc='upper left')
plt.show()
