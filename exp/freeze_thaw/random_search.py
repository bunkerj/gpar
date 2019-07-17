import numpy as np
from time import time
from matplotlib import pyplot as plt
from exp.freeze_thaw.model_aggregator import ModelAggregator


class RandomSearch:
    def __init__(self, hyp_ranges, n_samples, n_epochs):
        self.hyp_ranges = hyp_ranges
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.min_losses = np.array([]).reshape((-1, 1))
        self.training_time = 0

    def _sample_hyp_config(self):
        hyp_config = []
        for hyp_range in self.hyp_ranges:
            high = hyp_range[1]
            low = hyp_range[0]
            hyp_config.append(np.random.randint(low, high + 1))
        return tuple(hyp_config)

    def _get_hyp_configs(self):
        return [self._sample_hyp_config() for _ in range(self.n_samples)]

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
