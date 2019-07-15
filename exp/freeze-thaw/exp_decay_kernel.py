import gpflow
import tensorflow as tf
from gpflow.decors import params_as_tensors


class ExponentialDecay(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(input_dim=1, active_dims=[0])
        self.alpha = gpflow.Param(1.0, transform=gpflow.transforms.positive)
        self.beta = gpflow.Param(0.5, transform=gpflow.transforms.positive)

    @params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return tf.pow(self.beta, self.alpha) \
               / tf.pow((tf.add(tf.add(X, tf.transpose(X2)), self.beta)), self.alpha)

    @params_as_tensors
    def Kdiag(self, X):
        return tf.pow(self.beta, self.alpha) \
               / tf.pow(tf.add(2 * tf.reshape(X, (-1,)), self.beta), self.alpha)
