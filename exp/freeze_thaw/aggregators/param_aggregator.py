import scipy
import numpy as np
import tensorflow as tf
from exp.freeze_thaw.utils import key_to_hyp, get_block_diag_matrix


class ParamAggregator:
    def __init__(self, loss_count_dict):
        curve_count = len(loss_count_dict)
        self.loss_count_dict = loss_count_dict
        self.global_means = self._initialize_global_means(curve_count)
        self.global_kernel_param_list = self._initialize_global_param_list()
        self.local_kernel_param_list = self._initialize_local_param_list(curve_count)

    def get_global_means(self):
        return self.global_means

    def get_global_kernel_param_list(self):
        return self.global_kernel_param_list

    def get_local_kernel_param_list(self):
        # Make sure list is flattened before returning
        return self._flatten_list(self.local_kernel_param_list)

    def get_O(self):
        arrays = [np.ones((count, 1)) for count in self.loss_count_dict.values()]
        O_raw = scipy.linalg.block_diag(*arrays)
        return tf.convert_to_tensor(O_raw, dtype=float)

    def get_K_t(self):
        K_t_mats = []
        for i, count in enumerate(self.loss_count_dict.values()):
            alpha, beta, diag_noise = self.local_kernel_param_list[i]
            kernel = self._exp_kernel_generator(alpha, beta)
            index_mat = tf.reshape(tf.range(1, count + 1, dtype=float), (-1, 1))
            K_t_mat = self._compute_gram_matrix(kernel, index_mat) \
                      + tf.eye(count) * (diag_noise + 1e-3)
            K_t_mats.append(K_t_mat)
        arr = get_block_diag_matrix(K_t_mats)
        return arr

    def get_K_x(self):
        index_matrix = self._get_index_matrix()
        kernel = self._rbf_kernel_generator(*self.global_kernel_param_list)
        K_x = self._compute_gram_matrix(kernel, index_matrix)
        return K_x

    def get_global_posterior(self, y):
        """Return Gaussian mean and variance."""
        K_x = self.get_K_x()
        K_t = self.get_K_t()
        K_t_inv = np.linalg.inv(K_t)

        m = self.get_global_means()
        O = self.get_O()

        T1 = np.matmul(tf.transpose(O), K_t_inv)
        t = y - np.matmul(O, m)
        L = np.matmul(T1, t)
        L_inv = np.linalg.inv(L)
        G = np.matmul(T1, O)

        C = K_x - np.matmul(np.matmul(K_x, np.linalg.inv(K_x + L_inv)), K_x)
        mu = m + np.matmul(C, G)

        return mu, C

    def get_global_posterior_predictive(self):
        """Return Gaussian mean and variance."""
        pass

    def get_local_posterior_predictive(self):
        """Return Gaussian mean and variance."""
        # Use two schemes depending on if there are existing observations
        pass

    def _get_index_matrix(self):
        """
        Returns an MxN matrix where each row corresponds
        to a hyperparam config.
        """
        index_rows = self._get_index_rows()
        index_matrix = np.concatenate(index_rows, axis=0)
        return tf.convert_to_tensor(index_matrix, dtype=float)

    def _get_index_rows(self):
        """
        Returns a list where each element is a 1xN vector
        corresponding to a hyperparam config.
        """
        return [np.array(key_to_hyp(idx)).reshape((1, -1))
                for idx in self.loss_count_dict.keys()]

    def _flatten_list(self, arr):
        # TODO: make this a recursive utils function
        flattened_list = []
        for v in arr:
            flattened_list += v
        return flattened_list

    def _initialize_global_means(self, curve_count):
        return tf.Variable(tf.ones((curve_count, 1)))

    def _initialize_global_param_list(self):
        """For global RBF: vs, ls."""
        return [tf.Variable(np.random.lognormal()),
                tf.Variable(np.random.lognormal())]

    def _sample_local_param(self):
        return [tf.Variable(np.random.lognormal()),
                tf.Variable(np.random.lognormal()),
                tf.Variable(np.random.lognormal())]

    def _initialize_local_param_list(self, curve_count):
        """For each curve: alpha, beta, diagonal_noise."""
        return [self._sample_local_param() for _ in range(curve_count)]

    def _rbf_kernel_generator(self, vs, ls):
        return lambda x, y: tf.exp(-tf.exp(vs) / (2. * tf.pow(tf.exp(ls), 2.)) *
                                   sum(tf.pow(x - y, 2.)))

    def _exp_kernel_generator(self, alpha, beta):
        return lambda x, y: tf.pow(beta, alpha) / tf.pow(x + y + beta, alpha)

    def _compute_gram_matrix(self, kernel, X1, X2=None):
        if X2 is None:
            X2 = X1
        kern_outputs = []
        for x in X1:
            for y in X2:
                kern_outputs.append(kernel(x, y))
        return tf.reshape(kern_outputs, (tf.shape(X1)[0], tf.shape(X2)[0]))
