import pickle
import numpy as np
from exp.likelihood_std.constants import DATA_PATH
from matplotlib import pyplot as plt

with open(DATA_PATH, 'rb') as file:
    data_dict = pickle.load(file)

n_restarts_list = list(data_dict.keys())
log_likelihood_vars = [np.std(values) for values in data_dict.values()]

plt.plot(n_restarts_list, log_likelihood_vars)
plt.title('Total Log-likelihood Std. vs Num. Restarts')
plt.ylabel('Total Log-likelihood Std.')
plt.xlabel('Num. Restarts')
plt.grid()

plt.show()
