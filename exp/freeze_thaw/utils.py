import scipy
import tensorflow as tf
import numpy as np


def hyp_to_key(hyp):
    return '_'.join(str(v) for v in hyp)


def key_to_hyp(key):
    return tuple(int(v) for v in key.split('_'))


def get_O(count_list):
    arrays = [np.ones((count, 1)) for count in count_list]
    O_raw = scipy.linalg.block_diag(*arrays)
    return tf.constant(O_raw)
