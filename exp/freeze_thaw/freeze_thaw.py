import numpy as np
from matplotlib import pyplot as plt
from src_utils import get_bounded_samples
from exp.freeze_thaw.model_aggregator import ModelAggregator


class FreezeThaw:
    def __init__(self, hyp_bounds_list, b_old, b_new,
                 max_epoch=100, init_epochs=10):
        self.hyp_bounds_list = hyp_bounds_list
        self.b_old = b_old
        self.b_new = b_new
        self.max_epoch = max_epoch
        self.init_epochs = init_epochs
        self.model_aggregator = None

    def _train_initial_models(self):
        hyp_list = get_bounded_samples(self.hyp_bounds_list, self.b_old)
        self.model_aggregator = ModelAggregator(hyp_list)
        self.model_aggregator.train_all_models(self.init_epochs)

    def _construct_key_basket(self):
        """
        Use EI to compute this basket.
        Return: list of model keys.
        """
        return self.model_aggregator.get_all_keys()

    def select_model_to_train(self, key_basket):
        """
        Use ES to compute the model to train.
        Return: key of model to train.
        """
        rand_index = np.random.randint(0, len(key_basket))
        return key_basket[rand_index]

    def _train_model_from_basket(self, basket):
        model_key = self.select_model_to_train(basket)
        print('Training model {}'.format(model_key))
        self.model_aggregator.train_model_given_key(model_key, 1)

    def _update_hyperparameters(self):
        print('Updating the hyperparameters...')

    def run(self):
        self._train_initial_models()
        for _ in range(self.max_epoch):
            key_basket = self._construct_key_basket()
            self._train_model_from_basket(key_basket)
            self._update_hyperparameters()

    def _get_suitable_indices(self, arr):
        return np.arange(1, arr.shape[0] + 1).reshape((-1, 1))

    def plot(self):
        """Plot all losses."""
        losses = self.model_aggregator.get_all_losses()
        for key in losses:
            loss = losses[key]
            x = self._get_suitable_indices(loss)
            plt.plot(x, loss, label=key)
