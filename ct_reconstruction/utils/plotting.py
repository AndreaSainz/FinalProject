"""
Visualization utilities for CT reconstruction models.

Includes functions to:
    - Display side-by-side image comparisons (model output vs. ground truth).
    - Plot training/validation metrics over epochs.
    - Visualize reconstructions from multiple classical CT algorithms.
    - Optionally overlay test metrics as horizontal reference lines.

These tools are used to visually assess model performance during and after training.
"""

import matplotlib.pyplot as plt
import torch

def show_example(output_img, ground_truth):
    """
    Displays a side-by-side comparison of the model's reconstruction and the ground truth image.

    Args:
        output_img (torch.Tensor): Reconstructed image from the model. Shape: (1, H, W) or (H, W).
        ground_truth (torch.Tensor): Ground truth image for comparison. Same shape as output_img.
    """

    # creating one figure with two imagenes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(output_img.squeeze().cpu(), cmap='gray')
    axes[0].set_title('Reconstruction')
    axes[1].imshow(ground_truth.squeeze().cpu(), cmap='gray')
    axes[1].set_title('Ground Truth')
    plt.tight_layout()
    plt.show()

def show_example_epoch(output_img, ground_truth, epoch, save_path=None):
    """
    Displays a side-by-side comparison of the model's reconstruction and the ground truth image.

    Args:
        output_img (torch.Tensor): Reconstructed image from the model.
        ground_truth (torch.Tensor): Ground truth image.
        epoch (int): Epoch index.
        save_path (str, optional): Base path to save the figure (without extension).
    """
    # creating one figure with two images
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(output_img.squeeze().detach().cpu(), cmap='gray')
    axes[0].set_title(f"Reconstructed image in epoch {epoch}")
    axes[1].imshow(ground_truth.squeeze().detach().cpu(), cmap='gray')
    axes[1].set_title(f"Ground Truth in epoch {epoch}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_{epoch}.png")
    
    plt.close()


def plot_metric(x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
    """
    Plots training/validation metrics over a sequence (e.g., epochs).

    Args:
        x (list or range): Values for the x-axis (e.g., epoch indices).
        y_dict (dict): Dictionary of curves to plot. Keys are labels, values are y-values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        test_value (float, optional): If provided, adds a horizontal line to indicate a test/reference value.
        save_path (str, optional): If provided, saves the figure to this path.
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
    Saves plots comparing model output, ground truth, and multiple CT reconstructions.

    Args:
        model_type (str): Name of the trained model used (e.g., "UNet", "ResNet").
        sample (int): Sample index used in filenames.
        recon_dict (dict): Dictionary of reconstructions from various algorithms.
        output_img (torch.Tensor): Model's predicted reconstruction.
        ground_truth (torch.Tensor): Ground truth image.
        save_path (str, optional): Base path to save the output images. Filenames will be suffixed with sample/algorithm.
    """

    #model reconstructed image
    plt.figure()
    plt.imshow(output_img.squeeze().cpu(), cmap='gray')
    plt.title(f"Reconstructed Image with {model_type} model")
    plt.tight_layout()
    plt.savefig(f"{save_path}_{sample}_model_prediction.png")
    plt.close()

    # ground truth image
    plt.figure()
    plt.imshow(ground_truth.squeeze().cpu(), cmap='gray')
    plt.title(f"Ground Truth Image")
    plt.tight_layout()
    plt.savefig(f"{save_path}_{sample}_ground_truth.png")
    plt.close()

    # other reconstructed ct images (classical methods)
    for key in recon_dict:
        plt.figure()
        plt.imshow(recon_dict[key].squeeze().cpu(), cmap='gray')
        plt.title(f"Reconstructed Image with {key} algorithm")
        plt.tight_layout()
        plt.savefig(f"{save_path}_{sample}_{key}.png")
        plt.close()


