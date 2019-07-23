import tensorflow as tf
from exp.freeze_thaw.random_search import RandomSearch
from exp.freeze_thaw.freeze_thaw import FreezeThaw
from matplotlib import pyplot as plt

tf.enable_eager_execution()

N_SAMPLES = 3
N_EPOCHS = 2
bounds_list = [[1, 10], [5, 500]]
B_OLD = 5
B_NEW = 1

freeze_thaw = FreezeThaw(bounds_list, B_OLD, B_NEW, init_epochs=2, max_epoch=0)
freeze_thaw.run()
freeze_thaw.plot_min_losses()

random_search = RandomSearch(bounds_list, N_SAMPLES, N_EPOCHS)
random_search.run()
random_search.plot_min_losses()

plt.title('Lowest Loss over Epochs')
plt.ylabel('Lowest Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.grid()

plt.show()
