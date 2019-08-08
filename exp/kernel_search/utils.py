from exp.kernel_search.aggregate_kernel import AggregateKernel
from src.regression.gpar_regression import GPARRegression
from exp.kernel_search.constants import BASE_KERNEL_CONFIGS, INITIAL_BASE_KERNEL_CONFIGS
from src.src_utils import repeat_until_success


def enhance_kernel(new_kernels, current_original_kernel, config_list):
    for config in config_list:
        if current_original_kernel is None:
            kernel, has_index = config
            current_kernel = AggregateKernel(kernel, has_index)
        else:
            kernel, operator, has_index = config
            current_kernel = AggregateKernel()
            current_kernel.copy(current_original_kernel)
            current_kernel.update(kernel, operator, has_index)
        new_kernels.append(current_kernel)


def generate_kernels(current_kernels):
    """Returns list of new kernels based on current kernels."""
    new_kernels = []
    if len(current_kernels) == 0:
        enhance_kernel(new_kernels, None, INITIAL_BASE_KERNEL_CONFIGS)
    else:
        for current_original_kernel in current_kernels:
            enhance_kernel(new_kernels, current_original_kernel, BASE_KERNEL_CONFIGS)
    return new_kernels


def get_total_log_likelihood(X_obs, Y_obs, kernel, n_restarts, n_samples):
    total_log_likelihood = 0
    for j in range(n_samples):
        train_model = lambda: GPARRegression(X_obs, Y_obs, kernel,
                                             num_restarts=n_restarts)
        m = repeat_until_success(train_model)
        total_log_likelihood += m.get_total_log_likelihood() / n_samples
    return total_log_likelihood


def print_kernel(aggregate_kernel_list):
    for aggregate_kernel in aggregate_kernel_list:
        aggregate_kernel.print()
