import gpflow
import numpy as np
from scipy import integrate
from utils import predict_integral, execute_tensor
from synthetic_functions import gaussian_pdf
from kernels import RBF_dd

NOISE = 0.0001


def explicitly_compute_variance(K, K_s, K_ss_entry):
    K_noisy = K + NOISE * np.eye(K.shape[0])
    K_inv = np.linalg.inv(K_noisy)
    tmp = K_s_auto.dot(K_inv)
    return K_ss_entry - np.diag(tmp.dot(K_s.transpose()))[-1]


def compute_manual_single_diff(X_obs, k_man_f, START, X_test_single):
    K_s_man = np.zeros((2, X_obs.shape[0]))
    for i in range(X_obs.shape[0]):
        tmp_f = lambda x: k_man_f(X_obs[i], x)
        K_s_man[1, i] = integrate.quad(tmp_f, START, X_test_single)[0]
    return K_s_man


START = -10
END = 10
X_SINGLE_TEST = 3.0
N_obs = 35
f = gaussian_pdf

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

X_test_int_single = np.array([START, X_SINGLE_TEST]).reshape((-1, 1))
integral_truth_single = integrate.quad(f, START, X_SINGLE_TEST)[0]
integral_means_single, integral_vars_single = predict_integral(m, X_test_int_single, X_obs, Y_obs)
approx_int_means_single = float(integral_means_single[1])
approx_int_vars_single = np.diag(integral_vars_single)[1]

var_scale = float(m.kern.variance.value)
len_scale = float(m.kern.lengthscale.value)

# k_ss_man_f = lambda x1, x2: var_scale * np.exp(-(0.5 / (len_scale ** 2)) * (x1 - x2) ** 2)
# k_s_man_f = lambda x1, x2: (var_scale / (len_scale ** 2)) * \
#                            (x1 - x2) * np.exp(-(0.5 / (len_scale ** 2)) * (x1 - x2) ** 2)
k_man_f = lambda x, y: (var_scale / (len_scale ** 4)) * \
                       np.exp(-(0.5 / (len_scale ** 2)) * (x - y) ** 2) * \
                       (len_scale ** 2 - x ** 2 + 2 * x * y - y ** 2)

X_lb = 0 * X_test_int_single + X_test_int_single[0]

K_auto = m.kern.compute_K_symm(X_obs)
K_s_auto = execute_tensor(m.kern.K_s(X_test_int_single, X_obs) - m.kern.K_s(X_lb, X_obs))
K_man = k_man_f(X_obs, X_obs.transpose())

K_ss_auto = execute_tensor(m.kern.K_ss(X_test_int_single)
                           - 2 * m.kern.K_ss(X_test_int_single, X_lb)
                           + m.kern.K_ss(X_lb))

K_ss_man_entry = integrate.nquad(
    k_man_f, [[START, X_SINGLE_TEST], [START, X_SINGLE_TEST]])[0]

K_s_man = compute_manual_single_diff(X_obs, k_man_f, START, X_SINGLE_TEST)

var_auto = approx_int_vars_single
var_auto2 = explicitly_compute_variance(K_auto, K_s_auto, np.diag(K_ss_auto)[-1])
var_man = explicitly_compute_variance(K_man, K_s_man, K_ss_man_entry)

print('\nAuto: {}'.format(var_auto2))
print('Manual: {}'.format(var_man))

K_diff = np.max(np.abs(K_auto - K_man))
K_s_diff = np.max(np.abs(K_s_auto - K_s_man))
K_ss_diff = np.max(np.abs(np.diag(K_ss_auto)[-1] - K_ss_man_entry))

print('\nK_diff: {}'.format(K_diff))
print('K_s_diff: {}'.format(K_s_diff))
print('K_ss_diff: {}'.format(K_ss_diff))
