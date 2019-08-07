import pickle
from exp.kernel_search.constants import DATA_PATH

with open(DATA_PATH, 'rb') as file:
    likelihood_data = pickle.load(file)

print(likelihood_data)
