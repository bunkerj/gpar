from exp.kernel_search.constants import KERNEL_ADD, KERNEL_MULTIPLY


class AggregateKernel():
    def __init__(self, kernel=None, has_index=None):
        if kernel is None or has_index is None:
            self.raw_kernel = None
            self.kernel_string = None
        else:
            self.raw_kernel = self._get_initial_raw_kernel(kernel, has_index)
            self.kernel_string = '{}{}'.format(kernel.__name__, self._get_index_string(has_index))

    def copy(self, aggregate_kernel):
        self.raw_kernel = aggregate_kernel.get_raw_kernel()
        self.kernel_string = aggregate_kernel.get_kernel_string()

    def update(self, kernel, operator, has_index):
        self._update_raw_kernel(kernel, operator, has_index)
        self._update_kernel_string(kernel, operator, has_index)

    def _update_raw_kernel(self, kernel, operator, has_index):
        initial_raw_kernel = self._get_initial_raw_kernel(kernel, has_index)
        raw_kernel_old = self.raw_kernel

        def raw_kernel(original_X, current_X):
            if operator == KERNEL_ADD:
                return raw_kernel_old(original_X, current_X) + initial_raw_kernel(original_X, current_X)
            elif operator == KERNEL_MULTIPLY:
                return raw_kernel_old(original_X, current_X) * initial_raw_kernel(original_X, current_X)

        self.raw_kernel = raw_kernel

    def _get_index_string(self, has_index):
        return '_' if has_index else ''

    def _update_kernel_string(self, kernel, operator, has_index):
        name = kernel.__name__
        index_string = self._get_index_string(has_index)
        self.kernel_string = '( {} ) {} {}{}'.format(self.kernel_string, operator,
                                                     name, index_string)

    def _get_initial_raw_kernel(self, kernel, has_index):
        def raw_kernel(original_X, current_X):
            X_dim = original_X.shape[1]
            Y_dim = current_X.shape[1] - X_dim
            if Y_dim == 0:
                return kernel(input_dim=X_dim, active_dims=list(range(X_dim)))
            else:
                if has_index:
                    return kernel(input_dim=X_dim, active_dims=list(range(X_dim)))
                else:
                    return kernel(input_dim=Y_dim, active_dims=list(range(X_dim, X_dim + Y_dim)))

        return raw_kernel

    def get_raw_kernel(self):
        return self.raw_kernel

    def get_kernel_string(self):
        return self.kernel_string

    def print(self):
        print(self.kernel_string)
