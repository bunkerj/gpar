import GPy
import numpy as np
from matplotlib import pyplot as plt

# Generate synthetic visualization for poster presentation

np.random.seed(17)

START = 0
END = 6

# Hyperparameter space
plt.subplot(2, 1, 2)

x_obs_arr = [0.9, 1, 1.1, 3.1, 3.3, 3.4, 4, 4.05, 4.08, 4.1, 5]
n_obs = len(x_obs_arr)
x_obs = np.array(x_obs_arr).reshape((n_obs, 1))
y_obs = np.sin(x_obs) + 5 + np.random.normal(0, 0.1, size=(n_obs, 1))
kernel = GPy.kern.RBF(1)
m = GPy.models.GPRegression(x_obs, y_obs, kernel)
m.optimize_restarts(num_restarts=35)

n_pred = 1000
x_pred = np.linspace(START, END, n_pred).reshape((n_pred, 1))
means, variances = m.predict(x_pred)

pts_input = [0.2, 1.8, 3.1, 4.7]
pts = list(map(lambda x: round(x * n_pred / (END - START)), pts_input))

plt.plot(x_pred, means)
ub = means + 2 * np.sqrt(variances)
lb = means - 2 * np.sqrt(variances)
plt.fill_between(
    x_pred.flatten(),
    lb.flatten(),
    ub.flatten(),
    alpha=0.2,
    edgecolor='b')

colors = ['r', 'b', 'k', 'g']
for i in range(len(pts_input)):
    color = colors[i % len(colors)]
    ymax = ub[pts[i]]
    ymin = lb[pts[i]]
    plt.scatter(x_pred[pts[i]], means[pts[i]], color=color,
                marker='o', s=20, label='Observations')
    plt.vlines(x_pred[pts[i]], color=color, ymax=ymax, ymin=ymin)

plt.xticks([])
plt.yticks([])
plt.xlabel('Hyperparameters')
plt.ylabel('Asymptotic Loss')
plt.title('Posterior Prediction of Asymptotes')

# Training Loss
plt.subplot(2, 1, 1)

training_time_weights = 1 / variances[pts]
training_percentages = training_time_weights \
                       / max(training_time_weights)

for i in range(len(pts_input)):
    label = 'Model {}'.format(i + 1)
    n = round(100 * float(training_percentages[i]))
    end_epoch = max(round(10 * float(training_percentages[i])), 1)
    variance = variances[pts[i]]
    x = np.linspace(0, end_epoch, n).reshape((n, 1))
    asymptote = means[pts[i]]
    color = colors[i % len(colors)]
    y = np.exp(-(1 / asymptote) * x) \
        + np.random.normal(0, 0.005, size=(n, 1))
    plt.plot(x, y, color=color, label=label)

plt.xticks([])
plt.yticks([])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Losses During Training')
plt.legend(loc='upper right')

plt.show()
