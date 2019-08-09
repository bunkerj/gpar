import pickle
from src.data_source import generate_base_data
from src.src_utils import repeat_until_success
from exp.likelihood_std.constants import DATA_PATH
from src.regression.gpar_regression import GPARRegression
from src.kernels import get_non_linear_input_dependent_kernel, get_full_rbf_kernel

N_SAMPLES = 20
KERNEL = get_full_rbf_kernel

X_obs, Y_obs, X_new, Y_true = generate_base_data()

n_restarts_list = [1, 10, 20, 40, 60, 80, 100]

data_dict = {}

for n_restarts in n_restarts_list:
    for _ in range(N_SAMPLES):
        train_model = lambda: GPARRegression(X_obs, Y_obs, KERNEL,
                                             num_restarts=n_restarts)
        m = repeat_until_success(train_model)
        if n_restarts in data_dict:
            data_dict[n_restarts].append(m.get_total_log_likelihood())
        else:
            data_dict[n_restarts] = [m.get_total_log_likelihood()]

with open(DATA_PATH, 'wb') as file:
    pickle.dump(data_dict, file)
