import GPy
import numpy as np
from utils import *
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from evaluation import smse

n = 25
X = np.linspace(0, 1, n).reshape((n, 1))
Y3 = create_synthetic_output(y3, X)
Y2 = create_synthetic_output(y2, X)
Y1 = create_synthetic_output(y1, X)
Y = np.concatenate((Y1, Y2, Y3), axis=1)

m = GPARRegression(X, Y, GPy.kern.RBF)

n_new = 100
output_id = 2
X_new = np.linspace(-0, 1, n_new).reshape((n_new, 1))
Y_ref = np.array(list(map(y3, X_new))).reshape((n_new, 1))
means, vars = m.predict(X_new)
Y_pred_mean = slice_column(means, output_id)
Y_pred_var = slice_column(vars, output_id)

print(smse(Y_ref, Y_pred_mean))

plot_single_output(1, X, Y3, X_new, Y_pred_mean, Y_pred_var)
plt.plot(X_new, Y_ref)
plt.show()
