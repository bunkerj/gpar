import numpy as np
import tensorflow as tf
from scipy.integrate import nquad


def f_numpy(x, y, z):
    return np.exp(np.sum([1 / 3 * x, 1 / 5 * y, 1 / 7 * z]))


def f_ddd_numpy(x, y, z):
    return (1 / 105) * f_numpy(x, y, z)


def g_numpy(x, y):
    return np.exp(np.sum([1 / 3 * x, 1 / 5 * y]))


def g_dd_numpy(x, y):
    return (1 / 15) * g_numpy(x, y)


def f(x, y, z):
    return tf.exp(tf.reduce_sum([1 / 3 * x, 1 / 5 * y, 1 / 7 * z]))


def f_d(x, y, z):
    return tf.gradients(f(x, y, z), [x])[0]


def f_dd(x, y, z):
    return tf.gradients(f_d(x, y, z), [y])[0]


def f_ddd(x, y, z):
    return tf.gradients(f_dd(x, y, z), [z])[0]


def f_ddd_test(x, y, z):
    return tf.gradients(f(x, y, z), [x, y, z])


b = [5, 11, 17]
a = [3, 7, 13]

actual_approx_result = nquad(f_ddd_numpy, [[a[0], b[0]], [a[1], b[1]], [a[2], b[2]]])[0]
closed_form_result = f_numpy(b[0], b[1], b[2]) \
                     - f_numpy(a[0], b[1], b[2]) \
                     - f_numpy(b[0], a[1], b[2]) \
                     + f_numpy(a[0], a[1], b[2]) \
                     - f_numpy(b[0], b[1], a[2]) \
                     + f_numpy(a[0], b[1], a[2]) \
                     + f_numpy(b[0], a[1], a[2]) \
                     - f_numpy(a[0], a[1], a[2])

print('Actual Approx Result: {}'.format(actual_approx_result))
print('Closed Form Result: {}'.format(closed_form_result))
