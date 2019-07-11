from gpflow.kernels import RBF, RationalQuadratic, Linear


def get_linear_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    Y_dim = current_X.shape[1] - X_dim
    k1 = RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
    if Y_dim > 0:
        k_linear = Linear(input_dim=Y_dim,
                          active_dims=list(range(X_dim, X_dim + Y_dim)))
        return k1 + k_linear
    return k1


def get_linear_input_dependent_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    Y_dim = current_X.shape[1] - X_dim
    k1 = RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
    if Y_dim > 0:
        k2 = RationalQuadratic(input_dim=X_dim, active_dims=list(range(X_dim)))
        k_linear = Linear(input_dim=Y_dim,
                          active_dims=list(range(X_dim, X_dim + Y_dim)))
        return k1 + k2 * k_linear
    return k1


def get_non_linear_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    Y_dim = current_X.shape[1] - X_dim
    k1 = RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
    if Y_dim > 0:
        k2 = RationalQuadratic(input_dim=Y_dim,
                               active_dims=list(range(X_dim, X_dim + Y_dim)))
        return k1 + k2
    return k1


def get_non_linear_input_dependent_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    Y_dim = current_X.shape[1] - X_dim
    k1 = RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
    k2 = RationalQuadratic(input_dim=X_dim + Y_dim,
                           active_dims=list(range(0, X_dim + Y_dim)))
    return k1 + k2


def full_RBF(original_X, current_X):
    X_dim = current_X.shape[1]
    return RBF(input_dim=X_dim, active_dims=list(range(X_dim)))
