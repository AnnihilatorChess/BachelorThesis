import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os



plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # Standard NeurIPS body font
    "font.size": 11,                    # NeurIPS standard is roughly 10-11pt
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,               # Set to True if you have a local TeX distribution
})

# -- FNO
# baseline validation run FNO
baseline_fno_full_rollout = np.load(r"turbulent_radiative_layer_2D-fno-FNO-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")
best_fno_full_rollout = np.load(r"turbulent_radiative_layer_2D-fno_film-FNOFiLM-0.001\14\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")

# - UNET Convnext
# baseline validation run Convnext
baseline_conv_next_full_rollout = np.load(r"turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.001\8\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")
best_unet_convnext_full_rollout = np.load(r"turbulent_radiative_layer_2D-unet_convnext_film-UNetConvNextFiLM-0.001\7\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")

# -- UNET CLASSIC
# baseline validation run Unet classic
baseline_unet_classic_full_rollout = np.load(r"turbulent_radiative_layer_2D-unet_classic-UNetClassic-0.001\5\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")
# validate FiLM t_cool push classic
best_unet_classic_full_rollout = np.load(r"turbulent_radiative_layer_2D-unet_classic_film-UNetClassicFiLM-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy")



# cond_t_cool_push_fno_full_rollout = np.load(r"turbulent_radiative_layer_2D-fno-FNO-0.001\10\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_1100\full_RMSE_rollout.npy")

if True:
    print(f"FNO mean: {best_fno_full_rollout.mean()}")
    print(f"ConvNeXt mean: {best_unet_convnext_full_rollout.mean()}")
    print(f"Unet Classic mean: {best_unet_classic_full_rollout.mean()}")

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

if False:
    plot_rollout_losses(best_fno_full_rollout)
    plot_rollout_losses(best_unet_convnext_full_rollout)

if False:
    print(f"Baseline FNO mean: {baseline_fno_full_rollout.mean()}")
    print(f"Baseline ConvNeXt mean: {baseline_conv_next_full_rollout.mean()}")
    print(f"Baseline Unet Classic mean: {baseline_unet_classic_full_rollout.mean()}")
    #print(cond_t_cool_push_fno_full_rollout.mean())

    plot_rollout_losses(baseline_fno_full_rollout, title="Baseline FNO Rollout Loss")
    plot_rollout_losses(baseline_conv_next_full_rollout, title="Baseline Convnext Rollout Loss")
    plot_rollout_losses(baseline_unet_classic_full_rollout, title="Baseline Unet Classic Rollout Loss")
#plot_rollout_losses(cond_t_cool_push_fno_full_rollout, title="Cond t_cool Push FNO Rollout Loss")




def plot_one_step_vrmse(model_files, dataset_name="turbulent_radiative_layer_2D"):
    """
    Generates a grouped bar chart comparing one-step VRMSE for different models.

    Args:
        model_files (dict): A dictionary where keys are Model Names (str)
                            and values are paths to their corresponding .json files.
                            Example: {'FNO': 'path/to/fno.json', 'U-Net': 'path/to/unet.json'}
        dataset_name (str): The name of the dataset used in the wandb keys.
                            Default is 'turbulent_radiative_layer_2D'.
    """
    field_map = {
        'P': 'pressure',
        r'$V_y$': 'velocity_y',
        r'$V_x$': 'velocity_x',
        r'$\rho$': 'density'
    }
    data = []
    # 1. Iterate through each model and its file
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)

            # 2. Extract specific metrics for this model
            for label, field_key in field_map.items():
                full_key = f"valid_{dataset_name}/{field_key}_VRMSE_T=all"
                if full_key in metrics:
                    val = metrics[full_key]
                    data.append({
                        "Model": model_name,
                        "Field": label,
                        "VRMSE": val
                    })
                else:
                    print(f"Warning: Key '{full_key}' not found for model '{model_name}'")

        except FileNotFoundError:
            print(f"Error: File not found for model '{model_name}' at {file_path}")

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(
        data=df,
        x="Field",
        y="VRMSE",
        hue="Model",
        width=0.5,
        #palette="viridis",
        edgecolor="black"
    )

    # 5. Formatting
    plt.title(f"One-Step VRMSE by Field and Model", fontsize=14)
    plt.ylabel("")
    plt.xlabel("")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if False:
    model_paths = {
        "Baseline FNO": "data/baseline_fno_wandb-summary.json",
        "Baseline U-Net_Classic": "data/baseline_unet_classic_wandb-summary.json",
        "Baseline U-Net_ConvNeXt": "data/baseline_unet_convnext_wandb-summary.json"
    }
    plot_one_step_vrmse(model_paths)


def plot_rollout_mean_vrmse(model_paths_map):
    """
    Calculates the mean of the full rollout VRMSE for specific physical fields
    and plots them in a grouped bar chart comparing models.

    Args:
        model_paths_map (dict): Dictionary where keys are Model Names and values
                                are paths to the 'full_VRMSE_rollout.npy' file.
                                The function assumes field-specific files
                                (e.g., 'density_VRMSE_rollout.npy') exist in
                                the same directory.
    """
    # Map label to the filename prefix used in The Well
    field_map = {
        'P': 'pressure',
        r'$V_y$': 'velocity_y',
        r'$V_x$': 'velocity_x',
        r'$\rho$': 'density'
    }

    data = []

    for model_name, full_path in model_paths_map.items():
        # Get the directory containing the .npy files
        base_dir = os.path.dirname(full_path)

        for label, field_name in field_map.items():
            # Construct the filename: e.g., "density_VRMSE_rollout.npy"
            filename = f"{field_name}_VRMSE_rollout.npy"
            file_path = os.path.join(base_dir, filename)

            if os.path.exists(file_path):
                try:
                    # Load the 1D array of losses (shape: [Rollout_Steps])
                    rollout_array = np.load(file_path)

                    # Calculate the mean over the entire rollout
                    mean_loss = rollout_array.mean()

                    data.append({
                        "Model": model_name,
                        "Field": label,
                        "Mean Rollout VRMSE": mean_loss
                    })
                except Exception as e:
                    print(f"Error loading {filename} for {model_name}: {e}")
            else:
                print(f"Warning: File not found for {model_name}: {filename}")

    if not data:
        print("No data found to plot.")
        return

    # Create DataFrame for Seaborn
    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Field",
        y="Mean Rollout VRMSE",
        hue="Model",
        width=0.5,
        edgecolor="black"
    )

    plt.title("Mean Full Rollout VRMSE by Field and Model", fontsize=14)
    plt.ylabel("", fontsize=12)
    plt.xlabel("")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if False:

    paths = {
        "Best FNO": r"turbulent_radiative_layer_2D-fno_film-FNOFiLM-0.001\14\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",
        "Best U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic_film-UNetClassicFiLM-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",
        "Best U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext_film-UNetConvNextFiLM-0.001\7\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",

    }

    plot_rollout_mean_vrmse(paths)

if False:
    # --- Usage Example ---
    # Pass the PATHS (strings), not the loaded arrays
    paths = {
        "Baseline FNO": r"turbulent_radiative_layer_2D-fno-FNO-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",
        "Baseline U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic-UNetClassic-0.001\5\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",
        "Baseline U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.001\8\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0\full_VRMSE_rollout.npy",

    }

    plot_rollout_mean_vrmse(paths)


def plot_spectral_rollouts(model_paths_map):
    """
    Creates a 3-panel line plot stacked vertically showing the rollout evolution
    of Spectral NMSE for High, Mid, and Low frequency bins (top to bottom).
    Includes a global horizontal legend positioned under the main title.
    """
    # Map axes to bins in reverse order: top=High (Bin 2), middle=Mid (Bin 1), bottom=Low (Bin 0)
    bin_indices = [2, 1, 0]
    y_labels = ['High', 'Mid', 'Low']

    # 1. Setup Plot: 3 rows, 1 column for vertical stacking
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Set the main figure title, positioned higher to accommodate the legend row
    fig.suptitle("Spectral NMSE per frequency bin", fontsize=16, y=0.97)

    # 2. Define colors and linestyles
    def get_model_color_linestyle(model_name):
        name_lower = model_name.lower()
        if 'fno' in name_lower:
            return 'tab:blue', 'solid'
        elif 'classic' in name_lower:
            return 'tab:orange', 'dashed'
        elif 'convnext' in name_lower:
            return 'tab:green', 'dashdot'
        return '#000000', 'solid'

    # Lists to store line objects for the global legend
    lines = []
    labels = []

    # 3. Iterate through the axes and map them to the correct frequency bins
    for i, ax in enumerate(axes):
        bin_idx = bin_indices[i]
        label = y_labels[i]

        # Iterate through each Model (Line)
        for model_name, base_path in model_paths_map.items():
            # Load the 'full' error data representing the average across all fields
            filename = f"full_spectral_error_nmse_per_bin_{bin_idx}_rollout.npy"
            file_path = os.path.join(base_path, filename)

            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    color, linestyle = get_model_color_linestyle(model_name)

                    line, = ax.plot(
                        data,
                        label=model_name,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.5,
                        alpha=0.9
                    )

                    # Collect legend objects once from the first plot
                    if i == 0:
                        lines.append(line)
                        labels.append(model_name)

                except Exception as e:
                    print(f"Error loading {filename} for {model_name}: {e}")
            else:
                print(f"Warning: File not found: {file_path}")

        # 4. Formatting per subplot
        ax.set_ylabel(label, fontsize=14)
        ax.grid(False)

        # Make axis ticks less frequent
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # 5. Global Legend row under the main title
    # fig.legend(
    #     lines,
    #     labels,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 0.94),
    #     ncol=len(labels),
    #     fontsize=12,
    #     frameon=True
    # )

    # Set common x-label "T" on the bottom plot
    axes[2].set_xlabel("T", fontsize=14)

    # Adjust layout to accommodate suptitle and legend row
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if True:
    paths = {
        "Best FNO": r"turbulent_radiative_layer_2D-fno_film-FNOFiLM-0.001\14\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic_film-UNetClassicFiLM-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext_film-UNetConvNextFiLM-0.001\7\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",

    }
    plot_spectral_rollouts(paths)

if False:
    paths = {
        "Baseline FNO": r"turbulent_radiative_layer_2D-fno-FNO-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic-UNetClassic-0.001\5\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.001\8\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0"
    }

    plot_spectral_rollouts(paths)


def plot_spectral_rollouts_both(model_paths_map):
    # Mapping axes to bins: High (Top), Mid, Low (Bottom)
    bin_indices = [2, 1, 0]
    y_labels = ['High', 'Mid', 'Low']

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle("Spectral NMSE: Baseline vs. Improved Stability", fontsize=16, fontweight='bold', y=0.97)

    def get_style(model_name):
        name_lower = model_name.lower()
        # Architecture Color
        if 'fno' in name_lower:
            color = 'tab:blue'
        elif 'classic' in name_lower:
            color = 'tab:orange'
        elif 'convnext' in name_lower:
            color = 'tab:green'
        else:
            color = 'black'

        # Version Style: Baseline is dashed, Improved is solid
        linestyle = 'dashed' if 'baseline' in name_lower else 'solid'
        # Improved models get a thicker line for emphasis
        linewidth = 2.0 if 'baseline' in name_lower else 3.0

        return color, linestyle, linewidth

    lines, labels = [], []

    for i, ax in enumerate(axes):
        bin_idx = bin_indices[i]
        for model_name, base_path in model_paths_map.items():
            filename = f"full_spectral_error_nmse_per_bin_{bin_idx}_rollout.npy"
            file_path = os.path.join(base_path, filename)

            if os.path.exists(file_path):
                data = np.load(file_path)
                color, linestyle, lw = get_style(model_name)
                line, = ax.plot(data, color=color, linestyle=linestyle, linewidth=lw, alpha=0.9)

                if i == 0:
                    lines.append(line)
                    labels.append(model_name)

        ax.set_ylabel(y_labels[i], fontsize=14, fontweight='bold')
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Single horizontal legend
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               ncol=3, fontsize=10, frameon=False)

    axes[2].set_xlabel("Rollout Steps (T)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()


if False:
    paths = {
        "Best FNO": r"turbulent_radiative_layer_2D-fno_film-FNOFiLM-0.001\14\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic_film-UNetClassicFiLM-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext_film-UNetConvNextFiLM-0.001\7\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline FNO": r"turbulent_radiative_layer_2D-fno-FNO-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic-UNetClassic-0.001\5\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.001\8\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0"
    }
    plot_spectral_rollouts_both(paths)


def plot_rollout_mean_vrmse(model_paths_map):
    """
    Plots the mean full rollout VRMSE using a grouped bar chart with paired saturated colors.
    """
    field_map = {
        'P': 'pressure',
        r'$V_y$': 'velocity_y',
        r'$V_x$': 'velocity_x',
        r'$\rho$': 'density'
    }

    data = []

    # Identify architecture families for color grouping
    for model_name, base_dir in model_paths_map.items():

        for label, field_name in field_map.items():
            filename = f"{field_name}_VRMSE_rollout.npy"
            file_path = os.path.join(base_dir, filename)

            if os.path.exists(file_path):
                try:
                    rollout_array = np.load(file_path)
                    mean_loss = rollout_array.mean()

                    data.append({
                        "Model": model_name,
                        "Field": label,
                        "Mean Rollout VRMSE": mean_loss
                    })
                except Exception as e:
                    print(f"Error loading {filename} for {model_name}: {e}")

    df = pd.DataFrame(data)

    # 1. Define the Paired Saturated Palette
    # You must ensure the order of models in model_paths_map matches this palette,
    # or use a mapping dictionary as shown below.
    # [FNO-Base, FNO-Imp, Classic-Base, Classic-Imp, Conv-Base, Conv-Imp]
    paired_colors = {
        "Baseline FNO": "#9ecae1",  # Light Blue
        "Best FNO": "#084594",  # Dark Blue
        "Baseline U-Net_Classic": "#fdae6b",  # Light Orange
        "Best U-Net_Classic": "#d94801",  # Dark Orange
        "Baseline U-Net_ConvNeXt": "#a1d99b",  # Light Green
        "Best U-Net_ConvNeXt": "#006d2c"  # Dark Green
    }

    plt.figure(figsize=(12, 7))

    # 2. Plotting with manual color mapping
    ax = sns.barplot(
        data=df,
        x="Field",
        y="Mean Rollout VRMSE",
        hue="Model",
        palette=paired_colors,
        edgecolor="black",
        width=0.8
    )

    # 3. Styling and Legend row under title
    plt.title("Mean Full Rollout VRMSE: Architecture & Stability Comparison",
              fontsize=15, fontweight='bold', pad=25)
    plt.ylabel("Mean VRMSE")
    plt.xlabel("")

    # Place legend as horizontal row directly under the title
    plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center',
               ncol=3, frameon=False, fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# Example Usage
if False:
    # Ensure keys match the 'paired_colors' dictionary exactly
    paths = {
        "Baseline FNO": r"turbulent_radiative_layer_2D-fno-FNO-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best FNO": r"turbulent_radiative_layer_2D-fno_film-FNOFiLM-0.001\14\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic-UNetClassic-0.001\5\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_Classic": r"turbulent_radiative_layer_2D-unet_classic_film-UNetClassicFiLM-0.001\13\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Baseline U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext-UNetConvNext-0.001\8\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
        "Best U-Net_ConvNeXt": r"turbulent_radiative_layer_2D-unet_convnext_film-UNetConvNextFiLM-0.001\7\viz\turbulent_radiative_layer_2D\rollout_losses\epoch_0",
    }
    plot_rollout_mean_vrmse(paths)
