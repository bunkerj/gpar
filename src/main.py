import GPy
import numpy as np
from utils import *
from synthetic_data_functions import *
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from evaluation import mse
from kernels import *

# Output stream of interest
OUTPUT_ID = 2
NUM_RESTARTS = 5
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n = 30
X = np.linspace(0, 1, n).reshape((n, 1))

Y3 = create_synthetic_output(y3_noisy, X)
Y2 = create_synthetic_output(y2_noisy, X)
Y1 = create_synthetic_output(y1_noisy, X)

outputs = (Y1, Y2, Y3)
Y = np.concatenate(outputs, axis=1)

# Construct input data for inference
n_new = 1000
X_new = np.linspace(-0, 1, n_new).reshape((n_new, 1))

# Construct ground truth output
synthetic_functions = (y1, y2, y3)
Y_ref = np.array(list(map(synthetic_functions[OUTPUT_ID], X_new))).reshape((n_new, 1))

# Get predictions from GPAR
m = GPARRegression(X, Y, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)
means, vars = m.predict(X_new)
Y_pred_mean = slice_column(means, OUTPUT_ID)
Y_pred_var = slice_column(vars, OUTPUT_ID)

# Get predictions from single Gaussian
m_single_gp = GPy.models.GPRegression(X, outputs[OUTPUT_ID], GPy.kern.RBF(input_dim=1))
m_single_gp.optimize_restarts(NUM_RESTARTS, verbose=False)
Y_pred_single_gp_mean, Y_pred_single_gp_var = m_single_gp.predict(X_new)

# Compare differences with ground truth
print('GPAR MSE: {}'.format(mse(Y_ref, Y_pred_mean)))
print('Single GP MSE: {}'.format(mse(Y_ref, Y_pred_single_gp_mean)))

# Generate plots
plot_single_output(1, X, outputs[OUTPUT_ID], X_new, Y_pred_mean, Y_pred_var)
plt.plot(X_new, Y_ref, label='Truth')
plt.legend(loc='upper left')
plt.title('GPAR')

plot_single_output(2, X, outputs[OUTPUT_ID], X_new, Y_pred_single_gp_mean, Y_pred_single_gp_var)
plt.plot(X_new, Y_ref, label='Truth')
plt.legend(loc='upper left')
plt.title('Single GP')

plt.show()
