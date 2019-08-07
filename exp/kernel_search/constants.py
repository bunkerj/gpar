import GPy

KERNEL_ADD = '+'
KERNEL_MULTIPLY = '*'

BASE_KERNELS = (
    GPy.kern.RBF,
    GPy.kern.RatQuad,
    GPy.kern.Linear,
)

KERNEL_OPERATORS = (
    KERNEL_ADD,
    KERNEL_MULTIPLY,
)

INCLUDE_INDEX = (
    True,
    False,
)

BASE_KERNEL_CONFIGS = tuple((kernel, op, has_index)
                            for kernel in BASE_KERNELS
                            for op in KERNEL_OPERATORS
                            for has_index in INCLUDE_INDEX)

INITIAL_BASE_KERNEL_CONFIGS = tuple((kernel, has_index)
                                    for kernel in BASE_KERNELS
                                    for has_index in INCLUDE_INDEX)
