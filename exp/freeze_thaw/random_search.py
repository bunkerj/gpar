import numpy as np
from time import time
from matplotlib import pyplot as plt
from src_utils import get_bounded_samples, stack_all_columns
from exp.freeze_thaw.model_aggregator import ModelAggregator


class RandomSearch:
    def __init__(self, hyp_bounds_list, n_samples, n_epochs):
        self.hyp_bounds_list = hyp_bounds_list
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.min_losses = np.array([]).reshape((-1, 1))
        self.training_time = 0

    def _get_min_loss(self, losses):
        return np.min(losses, axis=1).reshape((-1, 1))

    def run(self):
        hyp_list = get_bounded_samples(self.hyp_bounds_list, self.n_samples)
        model_aggregator = ModelAggregator(hyp_list)

        start_time = time()
        model_aggregator.train_all_models(self.n_epochs)
        self.training_time = time() - start_time

        losses = model_aggregator.get_all_losses()
        self.min_losses = self._get_min_loss(stack_all_columns(losses))

    def plot(self):
        epochs = np.arange(1, self.n_epochs + 1) \
            .reshape((-1, 1)).astype(float)
        plt.plot(epochs, self.min_losses, label='Random Search')
