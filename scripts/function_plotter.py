import numpy as np
from kernels import get_non_linear_input_dependent_kernel
from src_utils import map_and_stack_outputs, slice_column
from synthetic_functions import synthetic_functions
from matplotlib import pyplot as plt

KERNEL_FUNCTION = get_non_linear_input_dependent_kernel
FUNCTIONS = synthetic_functions
N_COLS = 3
N_ROWS = 1
START = 0
END = 1

if N_COLS * N_ROWS < len(FUNCTIONS):
    raise Exception('Dimensions are too small: {} < {}'
                    .format(N_COLS * N_ROWS, len(FUNCTIONS)))

# Construct true outputs
n = 1000
X = np.linspace(START, END, n).reshape((n, 1))
Y = map_and_stack_outputs(FUNCTIONS, X)

# Plot all outputs
for idx in range(len(FUNCTIONS)):
    plt.subplot(N_ROWS, N_COLS, idx + 1)
    plt.plot(X, slice_column(Y, idx))
    plt.title('Y{}'.format(idx + 1))

plt.show()
