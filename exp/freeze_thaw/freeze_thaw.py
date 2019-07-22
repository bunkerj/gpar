import numpy as np
from matplotlib import pyplot as plt
from src_utils import get_bounded_samples
from optimizer import Optimizer
from exp.freeze_thaw.utils import get_O
from exp.freeze_thaw.aggregators.model_aggregator import ModelAggregator
from exp.freeze_thaw.aggregators.param_aggregator import ParamAggregator


class FreezeThaw:
    def __init__(self, hyp_bounds_list, b_old, b_new,
                 max_epoch=100, init_epochs=10):
        self.hyp_bounds_list = hyp_bounds_list
        self.b_old = b_old
        self.b_new = b_new
        self.max_epoch = max_epoch
        self.init_epochs = init_epochs
        self.ma = None
        self.pa = None
        self.optimizer = Optimizer(learning_rate=0.1, n_epochs=10)

    def run(self):
        self._train_initial_models()
        self._update_hyperparameters()
        for _ in range(self.max_epoch):
            key_basket = self._construct_key_basket()
            self._train_model_from_basket(key_basket)
            self._update_hyperparameters()

    def plot(self):
        """Plot all losses."""
        losses = self.ma.get_all_losses()
        for key in losses:
            loss = losses[key]
            x = self._get_suitable_indices(loss)
            plt.plot(x, loss, label=key)

    def _train_initial_models(self):
        hyp_list = get_bounded_samples(self.hyp_bounds_list, self.b_old)
        self.ma = ModelAggregator(hyp_list)
        self.ma.train_all_models(self.init_epochs)

    def _construct_key_basket(self):
        """
        Use EI to compute this basket.
        Return: list of model keys.
        """
        return self.ma.get_all_keys()

    def _select_model_to_train(self, key_basket):
        """
        Use ES to compute the model to train.
        Return: key of model to train.
        """
        rand_index = np.random.randint(0, len(key_basket))
        return key_basket[rand_index]

    def _train_model_from_basket(self, basket):
        model_key = self._select_model_to_train(basket)
        print('Training model {}'.format(model_key))
        self.ma.train_model_given_key(model_key, 1)

    def _get_log_likelihood(self, y, O, m):
        """
        y: (y_1, y_2, ... , y_n)^T
        m: (m_1, m_2, ... , m_n)^T
        K_t: blockdiag(K_t1, K_t2, ... , K_tn)
        K_x: standard cov matrix (using RBF)
        O: blockdiag(1_1, 1_2, ... , 1_n)
        """
        K_t = self.pa.get_K_t()
        K_x = self.pa.get_K_x()
        return 3 + 2  # TODO: Construct loss expression.

    def _print_loss(self, args):
        print('Loss: {}'.format(self._get_log_likelihood(*args)))

    def _update_hyperparameters(self):
        print('Updating the hyperparameters...')
        if self.pa is None:
            self.pa = ParamAggregator(self.ma.get_curve_count())

        y = self.ma.get_stacked_losses()
        O = get_O(self.ma.get_loss_count_per_curve())
        m = self.pa.get_global_means()
        global_kernel_param_list = self.pa.get_global_kernel_param_list()
        local_kernel_param_list = self.pa.get_local_kernel_param_list()

        params = [m] + global_kernel_param_list + local_kernel_param_list
        args = (y, O, m)

        print('Loss Before: {}'.format(self._get_log_likelihood(*args)))
        self.optimizer.minimize_loss(self._get_log_likelihood, params, args)
        print('Loss Before: {}'.format(self._get_log_likelihood(*args)))

        a = 10  # TODO: Check if changes were performed.

    def _get_suitable_indices(self, arr):
        return np.arange(1, arr.shape[0] + 1).reshape((-1, 1))
