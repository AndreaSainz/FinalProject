"""
Visualization utilities for CT reconstruction experiments.

This module provides helper functions for plotting training curves,
comparing model outputs against ground truth images, and visualizing
reconstructions from various CT algorithms.

Includes:
    - Side-by-side image comparisons (output vs. ground truth)
    - Epoch-based visual tracking of reconstructions
    - Metric plotting with optional test/reference overlay
    - Visualization of outputs from multiple reconstruction pipelines

These functions are intended for use in model evaluation, debugging,
and presentation of CT reconstruction results.

Example:
    >>> from ct_reconstruction.utils.visualization import show_example, plot_metric
    >>> show_example(output, target)
    >>> plot_metric(range(50), {"train": train_loss, "val": val_loss}, "Loss", "Epoch", "MSE")
"""

import matplotlib.pyplot as plt
import torch
import seaborn as sns

# Define a colorblind-friendly palette (Color Universal Design by Okabe & Ito)
CUD_COLORS = [
    "#000000",  # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
]

# Apply consistent style and font sizes for accessibility
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
})


def show_example(output_img, ground_truth):
    """
    Displays a side-by-side comparison of a model reconstruction and its ground truth image.

    All images are visualized using a grayscale colormap with intensity values fixed in the range [0, 1].

    Args:
        output_img (torch.Tensor): Reconstructed image tensor of shape (H, W) or (1, H, W).
        ground_truth (torch.Tensor): Ground truth image tensor of the same shape.

    Returns:
        None. Displays the images using matplotlib.

    Example:
        >>> show_example(output_img, ground_truth)
    """

    # creating one figure with two imagenes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(output_img.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Reconstruction')
    axes[1].imshow(ground_truth.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_example_epoch(output_img, ground_truth, epoch, save_path=None):
    """
    Displays and optionally saves side-by-side images of the reconstruction and ground truth at a specific epoch.

    All images are visualized using a grayscale colormap with intensity values fixed in the range [0, 1].

    Args:
        output_img (torch.Tensor): Reconstructed image tensor.
        ground_truth (torch.Tensor): Ground truth image tensor.
        epoch (int): Epoch number, used for titles and filenames.
        save_path (str, optional): If provided, saves the figure to '{save_path}_{epoch}.png'.

    Returns:
        None
    """
    # creating one figure with two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(output_img.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Reconstructed image in epoch {epoch}")
    axes[1].imshow(ground_truth.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Ground Truth in epoch {epoch}")
    plt.tight_layout()
    plt.axis('off')
    
    if save_path:
        plt.savefig(f"{save_path}_{epoch}.png")
    
    plt.close()


def plot_metric(x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
    """
    Plots training and validation metrics over time (e.g., across epochs).

    If a test/reference value is provided, it is plotted as a dashed horizontal line for comparison.
    This function is intended for tracking metrics like loss, PSNR, or SSIM during training.

    Note: This function does not apply image normalization but is useful for scalar metric visualization.

    Args:
        x (list or range): X-axis values, typically epoch indices.
        y_dict (dict): Dictionary of metric series to plot. Keys are labels; values are lists of floats.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        test_value (float, optional): Optional horizontal line representing test/reference performance.
        save_path (str, optional): If provided, saves the plot to this path.

    Example:
        >>> plot_metric(range(50), {"train": train_loss, "val": val_loss}, "Loss", "Epoch", "MSE")
    """
    
    plt.figure()

    # go through the dictionary to plot the data
    for i, (label, y) in enumerate(y_dict.items()):
        plt.plot(x, y, label=label, color=CUD_COLORS[i % len(CUD_COLORS)])

    if test_value is not None:
        plt.axhline(y=test_value, color=CUD_COLORS[6], linestyle='--', label='Test')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_learned_filter(filter_module, angle_idx=None, angle=None, title=None, save_path="figures/"):
    """
    Plots the learned frequency-domain filter from a LearnableFilter module.

    Supports both shared filters and per-angle filters. If per-angle and angle_idx is not provided,
    plots the mean filter across all angles.

    Args:
        filter_module (LearnableFilter): Instance of the LearnableFilter class.
        angle_idx (int, optional): Index of the angle to visualize (used in per-angle mode).
        angle (int, optional): Angle label for title purposes.
        title (str, optional): Custom title for the plot.
        save_path (str): Base path for saving the plotted figure.
    """

    weights = filter_module.weights

    if filter_module.per_angle:
        filter_type = "FilterII"
        if angle_idx is not None:
            weights = weights[angle_idx]
            default_title = f"Filter II for Angle Index {angle}"
            file_suffix = f"{filter_type}_{angle}"
        else:
            # Compute mean filter across all angles
            weights = weights.mean(dim=0)
            default_title = "Mean Filter II over all angles"
            file_suffix = f"{filter_type}_mean"
    else:
        filter_type = "FilterI"
        default_title = "Shared Filter I (all angles)"
        file_suffix = f"{filter_type}"

    # Prepare weights for plotting
    if weights.ndim > 1:
        weights = weights.squeeze()
    weights = weights.detach()

    D = len(weights)
    freqs = torch.fft.fftshift(torch.fft.fftfreq(D))
    filter_shifted = torch.fft.fftshift(weights)

    freqs_np = freqs.cpu().numpy()
    filter_np = filter_shifted.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.fill_between(freqs_np, 0, filter_np, color="steelblue")
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title(title if title else default_title)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}_{file_suffix}.png")
    plt.close()

def plot_different_reconstructions(model_type, sample, recon_dict, output_img, ground_truth, save_path=None):
    """
    Saves a set of images comparing model outputs, ground truth, and classical CT reconstructions.

    All visualizations are generated using a fixed grayscale scale [0, 1] to ensure consistent brightness
    and contrast across models and methods.

    Args:
        model_type (str): Name of the model used (e.g., "DBP", "UNet").
        sample (int): Index of the current sample (used in file naming).
        recon_dict (dict): Dictionary of reconstructions from classical methods. Keys are method names.
        output_img (torch.Tensor): Model prediction tensor of shape (H, W) or (1, H, W).
        ground_truth (torch.Tensor): Ground truth tensor.
        save_path (str, optional): Base path where the figures will be saved.
    """

    #model reconstructed image
    plt.figure()
    plt.imshow(output_img.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"Reconstructed Image with {model_type} model")
    plt.tight_layout()
    plt.savefig(f"{save_path}_{sample}_model_prediction.png")
    plt.close()

    # ground truth image
    plt.figure()
    plt.imshow(ground_truth.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    plt.title(f"Ground Truth Image")
    plt.tight_layout()
    plt.savefig(f"{save_path}_{sample}_ground_truth.png")
    plt.close()

    # other reconstructed ct images (classical methods)
    for key in recon_dict:
        plt.figure()
        plt.imshow(recon_dict[key].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.title(f"Reconstructed Image with {key} algorithm")
        plt.tight_layout()
        plt.savefig(f"{save_path}_{sample}_{key}.png")
        plt.close()


def plot_learned_filter(filter_module, angle_idx=None, angle=None, title=None, save_path="figures/"):
    """
    Plots the learned frequency-domain filter from a LearnableFilter module.

    Supports both shared filters and per-angle filters. If per-angle and angle_idx is not provided,
    plots the mean filter across all angles.

    Args:
        filter_module (LearnableFilter): Instance of the LearnableFilter class.
        angle_idx (int, optional): Index of the angle to visualize (used in per-angle mode).
        angle (int, optional): Angle label for title purposes.
        title (str, optional): Custom title for the plot.
        save_path (str): Base path for saving the plotted figure.
    """

    weights = filter_module.weights

    if filter_module.per_angle:
        filter_type = "FilterII"
        if angle_idx is not None:
            weights = weights[angle_idx]
            default_title = f"Filter II for Angle Index {angle}"
            file_suffix = f"{filter_type}_{angle}"
        else:
            # Compute mean filter across all angles
            weights = weights.mean(dim=0)
            default_title = "Mean Filter II over all angles"
            file_suffix = f"{filter_type}_mean"
    else:
        filter_type = "FilterI"
        default_title = "Shared Filter I (all angles)"
        file_suffix = f"{filter_type}"

    # Prepare weights for plotting
    if weights.ndim > 1:
        weights = weights.squeeze()
    weights = weights.detach()

    D = len(weights)
    freqs = torch.fft.fftshift(torch.fft.fftfreq(D))
    filter_shifted = torch.fft.fftshift(weights)

    freqs_np = freqs.cpu().numpy()
    filter_np = filter_shifted.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.fill_between(freqs_np, 0, filter_np, color=CUD_COLORS[5])
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title(title if title else default_title)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.xticks([]) 
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_path}_{file_suffix}.png")
    plt.close()


def plot_results_distributions(path, df_psnr, df_ssim, df_mse):
    """
    Plots and saves violin plots comparing reconstruction metrics across methods.

    Args:
        path (str): File path prefix for saving plots.
        df_psnr (pd.DataFrame): DataFrame with columns ["Algorithm", "PSNR"].
        df_ssim (pd.DataFrame): DataFrame with columns ["Algorithm", "SSIM"].
        df_mse (pd.DataFrame): DataFrame with columns ["Algorithm", "MSE"].

    Saves:
        Three PNG files: _psnr_distribution.png, _ssim_distribution.png, _mse_distribution.png
    """

    metric_dfs = {
        "PSNR": df_psnr.rename(columns={"Algorithm": "Method", "PSNR": "Value"}),
        "SSIM": df_ssim.rename(columns={"Algorithm": "Method", "SSIM": "Value"}),
        "MSE": df_mse.rename(columns={"Algorithm": "Method", "MSE": "Value"}),
    }

    for metric, df in metric_dfs.items():
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Method", y="Value", data=df, inner="box", palette="Set2")
        plt.title(f"{metric} Distribution by Method")
        plt.ylabel(metric)
        plt.xlabel("Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{path}_{metric.lower()}_distribution.png")
        plt.close()