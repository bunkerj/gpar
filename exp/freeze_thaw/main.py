from exp.freeze_thaw.random_search import RandomSearch
from exp.freeze_thaw.freeze_thaw import FreezeThaw
from matplotlib import pyplot as plt

N_SAMPLES = 3
N_EPOCHS = 2
bounds_list = [[1, 10], [5, 500]]
B_OLD = 10
B_NEW = 3

freeze_thaw = FreezeThaw(bounds_list, B_OLD, B_NEW)
freeze_thaw.run()
freeze_thaw.plot()

random_search = RandomSearch(bounds_list, N_SAMPLES, N_EPOCHS)
random_search.run()
random_search.plot()

plt.title('Lowest Loss over Epochs')
plt.ylabel('Lowest Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid()
plt.show()
