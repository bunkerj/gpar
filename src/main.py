import GPy
import numpy as np
from utils import *
from synthetic_data_functions import *
from matplotlib import pyplot as plt
from gpar_regressor import GPARRegression
from visualiser import Visualiser
from kernels import *

np.random.seed(17)

# Output stream of interest
NUM_RESTARTS = 5
KERNEL_FUNCTION = get_non_linear_input_dependent_kernel

# Construct synthetic observations
n = 50
X = np.linspace(0, 1, n).reshape((n, 1))
noisy_functions = (y1_noisy, y2_noisy, y3_noisy)
Y = map_and_stack_outputs(noisy_functions, X)

# Construct true outputs
n_new = 1000
X_new = np.linspace(0, 1, n_new).reshape((n_new, 1))
synthetic_functions = (y1, y2, y3)
Y_true = map_and_stack_outputs(synthetic_functions, X_new)

# Get predictions from GPAR
gpar_model = GPARRegression(X, Y, KERNEL_FUNCTION, num_restarts=NUM_RESTARTS)

# Display results
visualiser = Visualiser(gpar_model, X, Y, X_new, Y_true)
visualiser.plot_mse_values(0)
visualiser.plot_all_outputs(1)
plt.show()
