import gpflow
import tensorflow as tf
from abc import ABC, abstractmethod
from gpflow.transforms import positive
from gpflow.decors import params_as_tensors


class DerivativeKernel(ABC, gpflow.kernels.Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _base_kernel(self, X1, X2):
        pass

    def _base_kernel_d(self, X1, X2):
        return tf.gradients(self._base_kernel(X1, X2), X2)[0]

    def _base_kernel_dd(self, X1, X2):
        return tf.gradients(self._base_kernel_d(X1, X2), X1)[0]

    def _compute_gram_matrix(self, X1, X2, kernel):
        if X2 is None:
            X2 = X1
        base_mat = tf.cast(0 * tf.add(X1, tf.transpose(X2)), float)
        input1_mat = base_mat + X1
        input2_mat = base_mat + tf.transpose(X2)
        return tf.map_fn(lambda x: kernel(x[0], x[1]),
                         (input1_mat, input2_mat), dtype=float)

    @params_as_tensors
    def K(self, X1, X2=None):
        return self._compute_gram_matrix(X1, X2, self._base_kernel_dd)

    @params_as_tensors
    def K_s(self, X1, X2=None):
        return self._compute_gram_matrix(X1, X2, self._base_kernel_d)

    @params_as_tensors
    def K_ss(self, X1, X2=None):
        return self._compute_gram_matrix(X1, X2, self._base_kernel)

    @params_as_tensors
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))


class RBF_dd(DerivativeKernel):
    def __init__(self, input_dim=1, lengthscale=1.0, variance=1.0, active_dims=[0]):
        super().__init__(input_dim, active_dims)
        self.lengthscale = gpflow.Param(lengthscale, transform=positive)
        self.variance = gpflow.Param(variance, transform=positive)

    def _base_kernel(self, X1, X2):
        d = tf.subtract(X1, tf.transpose(X2)) ** 2
        return self.variance * tf.exp(-(0.5 / (self.lengthscale ** 2)) * d)
