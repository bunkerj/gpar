from exp.freeze_thaw.random_search import RandomSearch
from matplotlib import pyplot as plt

n_samples = 20
n_epochs = 10
param_ranges = [[1, 10], [5, 500]]

random_search = RandomSearch(param_ranges, n_samples, n_epochs)
random_search.run()
random_search.plot()

plt.title('Lowest Loss over Epochs')
plt.ylabel('Lowest Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid()
plt.show()
