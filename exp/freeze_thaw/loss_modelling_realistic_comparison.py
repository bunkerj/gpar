import gpflow
from src.evaluation import smse
from matplotlib import pyplot as plt
from exp.freeze_thaw.aggregators.model_aggregator import ModelAggregator
from src.kernels import get_exponential_decay_kernel
from src.src_utils import sample_from_bounds, sort_by_value_len, concat_right_column
from exp.freeze_thaw.utils import get_observations_from_truth, get_index_values

N_MODELS = 6
N_MAX_EPOCHS = 25
N_OBS = 20
KERNEL_FUNCTION = get_exponential_decay_kernel


def get_trained_gp_model(X_base, X_curr, Y):
    kernel = KERNEL_FUNCTION(X_base, X_curr)
    m = gpflow.models.GPR(X_curr, Y, kernel)
    gpflow.train.ScipyOptimizer().minimize(m)
    return m


# Construct all models
bounds_list = [[1, 6], [5, 200]]
hyp_list = [sample_from_bounds(bounds_list) for _ in range(N_MODELS)]

# Train all models
model_aggregator = ModelAggregator(hyp_list)
model_aggregator.train_all_models(N_MAX_EPOCHS)
losses = model_aggregator.get_all_losses()

# Get epochs for truth
X_true = get_index_values(N_MAX_EPOCHS)

# Get observations
X_obs = X_true[:N_MAX_EPOCHS]
Y_obs = sort_by_value_len(get_observations_from_truth(losses))

x_full_gpar = get_index_values(N_MAX_EPOCHS)
x_obs_gpar = None

for idx, loss_key in enumerate(Y_obs):
    obs_loss = Y_obs[loss_key]
    true_loss = losses[loss_key]
    x_obs_base = get_index_values(len(obs_loss))
    x_obs_gpar = x_full_gpar[:len(obs_loss), :].copy()

    # Train GPs
    m_igp = get_trained_gp_model(x_obs_base, x_obs_base, obs_loss)
    m_gpar = get_trained_gp_model(x_obs_base, x_obs_gpar, obs_loss)
    print(m_gpar.compute_log_likelihood())

    # Get Predictions
    pred_igp = m_igp.predict_f(X_true)[0]
    pred_gpar = m_gpar.predict_f(x_full_gpar)[0]

    # Get Error Predictions
    smse_igp = smse(true_loss, pred_igp)
    smse_gpar = smse(true_loss, pred_gpar)

    # Plot modelling performance
    plt.figure(0)
    plt.subplots_adjust(hspace=0.32)
    plt.subplot(2, 3, idx + 1)
    plt.title('Model {}'.format(idx + 1))
    plt.plot(X_true, pred_igp, label='IGP')
    plt.plot(X_true, pred_gpar, label='GPAR')
    plt.plot(X_true, true_loss, label='Truth')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.scatter(x_obs_base, obs_loss, s=20, marker='x',
                color='b', label='Observations')
    plt.grid()

    if idx + 1 == 6:
        plt.legend(loc='upper right')

    # plot error estimation
    plt.figure(1)
    plt.subplot(2, 3, idx + 1)
    plt.bar([0, 1], [smse_igp, smse_gpar], tick_label=['IGP', 'GPAR'])
    plt.title('Model {} SMSE'.format(idx + 1))

    # Update cumulative GPAR index
    x_full_gpar = concat_right_column(x_full_gpar, pred_gpar)

# plt.subplot_tool()
plt.show()
