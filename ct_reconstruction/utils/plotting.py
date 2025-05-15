"""
Visualization utilities for CT reconstruction models.

Includes functions to:
    - Display side-by-side image comparisons (model output vs. ground truth).
    - Plot training/validation metrics over epochs.
    - Optionally overlay test metrics as horizontal reference lines.

These tools are used to visually assess model performance during and after training.
"""

import matplotlib.pyplot as plt
import torch

def show_example(output_img, ground_truth):
    """
    Displays a side-by-side comparison of the model's reconstruction and the ground truth.

    Args:
        output_img (torch.Tensor): Reconstructed image from the model. Shape: (1, H, W) or (H, W).
        ground_truth (torch.Tensor): Corresponding ground truth image. Same shape as output.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(output_img.squeeze().cpu(), cmap='gray')
    axes[0].set_title('Reconstruction')
    axes[1].imshow(ground_truth.squeeze().cpu(), cmap='gray')
    axes[1].set_title('Ground Truth')
    plt.tight_layout()
    plt.show()


def plot_metric(x, y_dict, title, xlabel, ylabel, test_value=None, save_path=None):
    """
    Plots one or more training/validation metrics over time (e.g., epochs).

    Optionally includes a horizontal line to represent a test set reference value.

    Args:
        x (list or range): Values for the x-axis (e.g., list of epoch indices).
        y_dict (dict): Dictionary of curves to plot. Keys are labels, values are lists of y-values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        test_value (float, optional): Horizontal line indicating test set metric value.
        save_path (str, optional): If specified, saves the plot to the given path.
    """
    
    plt.figure()
    for label, y in y_dict.items():
        plt.plot(x, y, label=label)

    if test_value is not None:
        plt.axhline(y=test_value, color='red', linestyle='--', label='Test')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()