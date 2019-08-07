import pickle
from exp.kernel_search.data_source import generate_data
from exp.kernel_search.utils import generate_kernels, get_total_log_likelihood, print_kernel
from exp.function_relationships.experiment_runner import ExperimentRunner
from kernels import get_non_linear_input_dependent_kernel
from matplotlib import pyplot as plt
from src.regression.gpar_regression import GPARRegression

DEPTH = 1
N_SAMPLES = 1
N_RESTARTS = 1

X_obs, Y_obs, X_new, Y_true = generate_data()

max_log_likelihood = None
best_kernel = None
best_kernel_string = None
current_kernels = []

likelihood_data = {}

for i in range(DEPTH):
    current_kernels = generate_kernels(current_kernels)
    print_kernel(current_kernels)
    print()
    for kernel in current_kernels:
        kernel.print()
        raw_kernel = kernel.get_raw_kernel()
        total_log_likelihood = get_total_log_likelihood(X_obs, Y_obs, raw_kernel, N_RESTARTS, N_SAMPLES)
        likelihood_data[kernel.get_kernel_string()] = total_log_likelihood
        if (max_log_likelihood is None) or (total_log_likelihood > max_log_likelihood):
            max_log_likelihood = total_log_likelihood
            best_kernel = raw_kernel
            best_kernel_string = kernel.get_kernel_string()

# --------------------------- Compute Error and Display Fit --------------------------- #


exp_kernel_search = ExperimentRunner(X_obs, Y_obs, X_new, Y_true,
                                     best_kernel, N_RESTARTS,
                                     figure_start=0)
exp_kernel_search.run()

exp_regular = ExperimentRunner(X_obs, Y_obs, X_new, Y_true,
                               get_non_linear_input_dependent_kernel,
                               N_RESTARTS, figure_start=10)
exp_regular.run()

# --------------------------- Print and Save Relevant Data --------------------------- #

with open('results/likelihood_data', 'wb') as file:
    pickle.dump(likelihood_data, file)

m = GPARRegression(X_obs, Y_obs, get_non_linear_input_dependent_kernel, num_restarts=N_RESTARTS)

print('-----------------------------------------')
print('[{}] log-likelihood: {}'.format(best_kernel_string, max_log_likelihood))
print('Baseline log-likelihood: {}'.format(m.get_total_log_likelihood()))

plt.show()
