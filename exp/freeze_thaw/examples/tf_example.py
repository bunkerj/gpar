import numpy as np
import tensorflow as tf
from src.optimizer import Optimizer

tf.enable_eager_execution()


def rbf_kernel_generator(vs, ls):
    return lambda x, y: tf.exp(-tf.exp(vs) / (2 * tf.pow(tf.exp(ls), 2)) *
                               sum(tf.pow(x - y, 2)))


def exp_kernel_generator(alpha, beta):
    pass


def print_info(vs, ls, loss_func):
    print('vs: {}  ---  ls: {}'.format(vs.numpy(), ls.numpy()))
    print('Loss: {}'.format(loss_func(*args)))


def compute_gram_matrix(kernel, X1, X2=None):
    if X2 is None:
        X2 = X1
    kern_outputs = []
    for x in X1:
        for y in X2:
            kern_outputs.append(kernel(x, y))
    return tf.reshape(kern_outputs, (tf.shape(X1)[0], tf.shape(X2)[0]))


def loss_func(x, y, kern):
    cov_mat = compute_gram_matrix(kern, x, y)
    return -(-tf.log(tf.linalg.det(cov_mat)) -
             tf.matmul(tf.matmul(tf.transpose(y), cov_mat), y))


vs = tf.Variable(np.log(2), dtype=float)
ls = tf.Variable(np.log(2), dtype=float)
params = [vs, ls]

kern = rbf_kernel_generator(vs, ls)
x = tf.reshape(tf.range(1, 5, dtype=float), (-1, 1))
y = x
args = (x, y, kern)

print_info(vs, ls, loss_func)
optimizer = Optimizer(learning_rate=0.1, n_epochs=10)
optimizer.minimize_loss(loss_func, params, args)
print_info(vs, ls, loss_func)
