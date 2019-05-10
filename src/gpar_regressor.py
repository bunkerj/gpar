import GPy
import numpy as np
from matplotlib import pyplot as plt


class GPARRegression(GPy.models.GPRegression):
    def __init__(self, X, Y, *args, **kwargs):
        super().__init__(X, Y, *args, **kwargs)

    def get_plot_statistics(self, X_new):
        pred = self.predict(X_new)
        mean, var = pred
        std = np.sqrt(var)
        ub = mean + 2 * std
        lb = mean - 2 * std
        return mean, ub, lb

    def plot(self, X_new, figure_id=1):
        plt.figure(figure_id)
        mean, ub, lb = self.get_plot_statistics(X_new)
        plt.plot(X_new, mean)
        plt.scatter(self.X, self.Y, color='b', marker='x')
        plt.fill_between(
            X_new.flatten(),
            lb.flatten(),
            ub.flatten(),
            alpha=0.2,
            edgecolor='b')
        plt.grid(True)
