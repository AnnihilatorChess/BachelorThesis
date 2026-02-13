import numpy as np
import matplotlib.pyplot as plt


baseline_fno_full_rollout = np.load(r"turbulent_radiative_layer_2D-fno-FNO-0.001\2\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_100\full_RMSE_rollout.npy")
cond_t_cool_push_fno_full_rollout = np.load(r"turbulent_radiative_layer_2D-fno-FNO-0.001\10\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_100\full_RMSE_rollout.npy")



def plot_rollout_losses(losses, title="Rollout Loss", xlabel="Rollout step", ylabel="Loss"):
    """
    Plot rollout losses over time.

    Parameters
    ----------
    losses : array-like
        1D array of rollout losses (e.g. shape (97,))
    title : str
        Plot title
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    """
    losses = np.asarray(losses)

    if losses.ndim != 1:
        raise ValueError("losses must be a 1D array")

    steps = np.arange(len(losses))

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(baseline_fno_full_rollout.mean())
print(cond_t_cool_push_fno_full_rollout.mean())

plot_rollout_losses(baseline_fno_full_rollout, title="Baseline FNO Rollout Loss")
plot_rollout_losses(cond_t_cool_push_fno_full_rollout, title="Cond t_cool Push FNO Rollout Loss")

