import pickle
from exp.bayesian_quadrature.constants import RESULTS_DIR
from matplotlib import pyplot as plt

ROOT_NAME = 'k4'
OBS = [5, 15, 30]
NAMES = ['{}_{}.pickle'.format(ROOT_NAME, obs) for obs in OBS]

for name in NAMES:
    with open(RESULTS_DIR + name, 'rb') as file:
        results = pickle.load(file)
        index, means, truth = (results[key] for key in ['index', 'means', 'truth'])
    n_obs = name.split('.')[0].split('_')[-1]
    plt.plot(index, means, label='{} Obs'.format(n_obs))

plt.title('Prediction Means for {} vs # MC Samples'.format(ROOT_NAME))
plt.axhline(truth, label='Truth', color='r')
plt.xlabel('# MC Samples')
plt.ylabel('Prediction Means')
plt.legend(loc='upper right')
plt.grid()

plt.show()
