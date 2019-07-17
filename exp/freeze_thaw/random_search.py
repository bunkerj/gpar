import numpy as np
from time import time
from matplotlib import pyplot as plt
from src_utils import sample_from_bounds
from exp.freeze_thaw.model_aggregator import ModelAggregator


class RandomSearch:
    def __init__(self, hyp_ranges, n_samples, n_epochs):
        self.hyp_bounds_list = hyp_ranges
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.min_losses = np.array([]).reshape((-1, 1))
        self.training_time = 0

    def _get_hyp_configs(self):
        return [sample_from_bounds(self.hyp_bounds_list) for _ in range(self.n_samples)]

    def _get_min_loss(self, losses):
        return np.min(losses, axis=1).reshape((-1, 1))

    def run(self):
        hyp_configs = self._get_hyp_configs()
        model_aggregator = ModelAggregator(hyp_configs, self.n_epochs)

        start_time = time()
        model_aggregator.train_all_models()
        self.training_time = time() - start_time

        losses = model_aggregator.get_all_losses()
        self.min_losses = self._get_min_loss(losses)

    def plot(self):
        epochs = np.arange(1, self.n_epochs + 1) \
            .reshape((-1, 1)).astype(float)
        plt.plot(epochs, self.min_losses, label='Random Search')
