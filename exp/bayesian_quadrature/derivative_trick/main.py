import gpflow
import numpy as np
from scipy import integrate, stats
from matplotlib import pyplot as plt
from exp.bayesian_quadrature.derivative_trick.kernels import RBF_dd
from exp.bayesian_quadrature.derivative_trick.synthetic_functions import gaussian_pdf
from exp.bayesian_quadrature.derivative_trick.utils \
    import predict_integral, plot_gp_vs_truth, plot_gaussian

START = -10
END = 10
N_obs = 50
N_true = 1000
N_test = 1000
N_test_int = 100
X_TEST_SINGLE = 0.5
f = gaussian_pdf

# Get integrand truth
X_true = np.linspace(START, END, N_true).reshape((-1, 1))
Y_true = f(X_true)

# Get observations
X_obs = np.linspace(START, END, N_obs).reshape((-1, 1))
Y_obs = f(X_obs)

# Build model
k = RBF_dd()
m = gpflow.models.GPR(X_obs, Y_obs, kern=k, mean_function=None)
print(m)

# Train model
opt = gpflow.train.ScipyOptimizer()
m.likelihood.variance = 0.0001
m.likelihood.variance.trainable = False
opt.minimize(m)
print(m)

# Get integrand predictions
X_test = np.linspace(START - 10, END + 10, N_test).reshape((-1, 1))
means, variances = m.predict_f(X_test)

# Get integral truth
integral_truth = integrate.quad(f, START, END)[0]

# Get integral predictions
X_test_int = np.linspace(START, END, N_test_int).reshape((-1, 1))
integral_means, integral_vars = predict_integral(m, X_test_int, X_obs, Y_obs)

# Perform plotting
plt.subplot(1, 3, 1)
plot_gp_vs_truth(X_true, Y_true, X_obs, Y_obs, X_test,
                 means, variances, 'Integrand Fit')

plt.subplot(1, 3, 2)
Y_int_true = stats.norm.cdf(X_true)
integral_vars_diag = np.diag(integral_vars).reshape((-1, 1))
plot_gp_vs_truth(X_true, Y_int_true, None, None, X_test_int,
                 integral_means, integral_vars_diag, 'Integral Fit')

plt.subplot(1, 3, 3)
X_test_int_single = np.array([START, X_TEST_SINGLE], dtype=float).reshape((-1, 1))
integral_truth_single = integrate.quad(f, START, X_TEST_SINGLE)[0]
integral_means_single, integral_vars_single = predict_integral(m, X_test_int_single, X_obs, Y_obs)
approx_int_means_single = float(integral_means_single[1])
approx_int_vars_single = np.diag(integral_vars_single)[1]
plot_gaussian(integral_truth_single, approx_int_means_single,
              approx_int_vars_single, label='X=3.0')

plt.show()
