import gpflow
import numpy as np
from matplotlib import pyplot as plt
from exp_decay_kernel import ExponentialDecay
from utils import mse
from model_aggregator import ModelAggregator

N_OBS = 5
N_EPOCHS = 10

HYP_LIST = [
    (1, 32),
    (2, 64),
    (3, 128),
    (2, 256)
]

# Train all models
model_aggregator = ModelAggregator(HYP_LIST, N_EPOCHS)
model_aggregator.train_all_models()
losses = model_aggregator.get_all_losses()
Y_true = losses[:, 2]

# Get epochs for truth
X_true = np.arange(1, Y_true.shape[0] + 1).reshape((-1, 1)).astype(float)

# Get observations
X_obs = X_true[0:N_OBS].reshape((-1, 1))
Y_obs = Y_true[0:N_OBS].reshape((-1, 1))

# Build model
k = ExponentialDecay()
m = gpflow.models.GPR(X_obs, Y_obs, kern=k, mean_function=None)

# Train model
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

# Get test points
X_test = X_true

# Get predictions and samples
mean, var = m.predict_f(X_test)

# Verify how well the predictions were performed
print('MSE: {}'.format(mse(mean, Y_true)))

plt.scatter(X_obs, Y_obs, s=10, marker='x')
plt.plot(X_test, mean, label='GP Mean')
plt.fill_between(X_test[:, 0],
                 mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                 mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                 color='C0', alpha=0.2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.plot(X_true, Y_true, label='Truth')
plt.legend(loc='upper right')
plt.grid()
plt.show()
