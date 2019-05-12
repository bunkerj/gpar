import GPy
import numpy as np
from utils import *
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from evaluation import smse

# Output stream of interest
OUTPUT_ID = 2

# Construct synthetic observations
n = 50
X = np.linspace(0, 1, n).reshape((n, 1))
Y3 = create_synthetic_output(y3, X)
Y2 = create_synthetic_output(y2, X)
Y1 = create_synthetic_output(y1, X)

outputs = (Y1, Y2, Y3)
synthetic_functions = (y1, y2, y3)
Y = np.concatenate(outputs, axis=1)

# Construct input data for inference
n_new = 1000
X_new = np.linspace(-0, 1, n_new).reshape((n_new, 1))

# Construct ground truth output
Y_ref = np.array(list(map(synthetic_functions[OUTPUT_ID], X_new))).reshape((n_new, 1))

# Get predictions from GPAR
kernel_function = GPy.kern.RBF
m = GPARRegression(X, Y, kernel_function)
means, vars = m.predict(X_new)
Y_pred_mean = slice_column(means, OUTPUT_ID)
Y_pred_var = slice_column(vars, OUTPUT_ID)

# Get predictions from single Gaussian
m_single_gp = GPy.models.GPRegression(X, outputs[OUTPUT_ID], kernel_function(input_dim=1))
m_single_gp.optimize_restarts(10, verbose=False)
Y_pred_single_gp_mean, Y_pred_single_gp_var = m_single_gp.predict(X_new)

# Compare differences with ground truth
print('GPAR SMSE: {}'.format(smse(Y_ref, Y_pred_mean)))
print('Single GP SMSE: {}'.format(smse(Y_ref, Y_pred_single_gp_mean)))

# Generate plots
plot_single_output(1, X, outputs[OUTPUT_ID], X_new, Y_pred_mean, Y_pred_var)
plot_single_output(2, X, outputs[OUTPUT_ID], X_new, Y_pred_single_gp_mean, Y_pred_single_gp_var)
plt.show()
