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

def show_example(output_img, ground_truth):
    """
    Displays a side-by-side comparison of a model reconstruction and its ground truth image.

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
    plt.tight_layout()
    plt.show()

def show_example_epoch(output_img, ground_truth, epoch, save_path=None):
    """
    Displays and optionally saves side-by-side images of the reconstruction and ground truth at a specific epoch.

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
    
    if save_path:
        plt.savefig(f"{save_path}_{epoch}.png")
    
    plt.close()


def plot_metric(x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
    """
    Plots training and validation metrics over time (e.g., across epochs).

    Args:
        x (list or range): X-axis values, typically epoch indices.
        y_dict (dict): Dictionary of metric series to plot. Keys are labels; values are lists of floats.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        test_value (float, optional): Optional horizontal line representing test/reference performance.
        save_path (str, optional): If provided, saves the plot to this path.

    Returns:
        None

    Example:
        >>> plot_metric(range(50), {"train": train_loss, "val": val_loss}, "Loss", "Epoch", "MSE")
    """
    
    plt.figure()

    # go through the dictionary to plot the data
    for label, y in y_dict.items():
        plt.plot(x, y, label=label)

    if test_value is not None:
        plt.axhline(y=test_value, color='red', linestyle='--', label='Test')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_different_reconstructions(model_type, sample, recon_dict, output_img, ground_truth, save_path=None):
    """
    Saves a set of images comparing model outputs, ground truth, and classical CT reconstructions.

    Args:
        model_type (str): Name of the model used (e.g., "DBP", "UNet").
        sample (int): Index of the current sample (used in file naming).
        recon_dict (dict): Dictionary of reconstructions from classical methods. Keys are method names.
        output_img (torch.Tensor): Model prediction tensor of shape (H, W) or (1, H, W).
        ground_truth (torch.Tensor): Ground truth tensor.
        save_path (str, optional): Base path where the figures will be saved.

    Returns:
        None
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


def plot_learned_filter(weights: torch.Tensor, title="(a) Filter I", save_path="filter_plot.png"):
    """
    Grafica el filtro aprendido en el dominio de la frecuencia con simetría centrada en cero.

    Args:
        weights (torch.Tensor): Tensor del filtro de forma (D,) o (1, D).
        title (str): Título de la gráfica.
    """

    # Asegurarse de que el tensor es 1D
    if weights.ndim > 1:
        weights = weights.squeeze()
        
    weights = weights.detach()

    # Generar eje de frecuencia simétrico
    D = len(weights)
    freqs = torch.fft.fftshift(torch.fft.fftfreq(D))
    filter_shifted = torch.fft.fftshift(weights)

    # Convertir a NumPy para graficar
    freqs_np = freqs.cpu().numpy()
    filter_np = filter_shifted.cpu().numpy()

    # Graficar y guardar
    plt.figure(figsize=(6, 4))
    plt.fill_between(freqs_np, 0, filter_np, color="steelblue")
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Cerrar para evitar mostrarla en notebooks