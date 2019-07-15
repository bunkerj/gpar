import gpflow
import tensorflow as tf
from gpflow.decors import params_as_tensors
from gpflow.kernels import RBF, RationalQuadratic, Linear


class ExponentialDecay(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(input_dim=1, active_dims=[0])
        self.alpha = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.beta = gpflow.Param(0.5, transform=gpflow.transforms.positive)

    @params_as_tensors
    def K(self, X_raw, X2_raw=None):
        if X2_raw is None:
            X2_raw = X_raw
        X = tf.reshape(X_raw[:, 0], (-1, 1))
        X2 = tf.reshape(X2_raw[:, 0], (-1, 1))
        return tf.pow(self.beta, self.alpha) \
               / tf.pow((tf.add(tf.add(X, tf.transpose(X2)), self.beta)), self.alpha)

    @params_as_tensors
    def Kdiag(self, X_raw):
        X = tf.reshape(X_raw[:, 0], (-1, 1))
        return tf.pow(self.beta, self.alpha) \
               / tf.pow(tf.add(2 * tf.reshape(X, (-1,)), self.beta), self.alpha)


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


def get_exponential_decay_kernel(original_X, current_X):
    X_dim = original_X.shape[1]
    if X_dim != 1:
        raise Exception('Invalid original X dimension')
    Y_dim = current_X.shape[1] - X_dim
    k1 = ExponentialDecay()
    if Y_dim > 0:
        k2 = RBF(input_dim=Y_dim,
                 active_dims=list(range(X_dim, X_dim + Y_dim)))
        return k1 + k2
    return k1
