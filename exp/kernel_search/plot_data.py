import pickle
from exp.kernel_search.constants import DATA_PATH
from matplotlib import pyplot as plt

BOUND = 10

with open(DATA_PATH, 'rb') as file:
    likelihood_data = pickle.load(file)

n = len(likelihood_data)
sorted_data = {k: v for k, v in
               sorted(likelihood_data.items(), key=lambda x: x[1])}

for idx, key in enumerate(list(sorted_data)):
    if BOUND <= idx < n - BOUND:
        del sorted_data[key]

print(sorted_data)

plt.barh(list(sorted_data.keys()),
         list(sorted_data.values()),
         align='center')
plt.title('Kernel Log-likelihoods for Kernel Search')
plt.xlabel('Log-likelihood')
plt.grid(axis='x')
plt.show()
