from exp.freeze_thaw.random_search import RandomSearch
from exp.freeze_thaw.freeze_thaw import FreezeThaw
from matplotlib import pyplot as plt

n_samples = 20
n_epochs = 10
param_ranges = [[1, 10], [5, 500]]

freeze_thaw = FreezeThaw(param_ranges)
freeze_thaw.run()
freeze_thaw.plot()

random_search = RandomSearch(param_ranges, n_samples, n_epochs)
random_search.run()
random_search.plot()

plt.title('Lowest Loss over Epochs')
plt.ylabel('Lowest Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid()
plt.show()
