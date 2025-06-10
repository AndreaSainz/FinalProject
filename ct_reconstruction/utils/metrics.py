"""
Utility functions for computing image quality metrics used in CT reconstruction.

This module includes functions to compute standard evaluation metrics for
reconstructed images, comparing them against ground truth references.

Metrics implemented:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)

These metrics are commonly used in image reconstruction tasks to evaluate
the fidelity of neural network outputs.

"""


from pytorch_msssim import ssim
import math
import torch
from torch.nn import MSELoss


def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) from a given MSE value.

    PSNR is a logarithmic metric that quantifies the difference between two images,
    typically used to evaluate reconstruction quality. Higher PSNR values indicate
    better image fidelity.

    Args:
        mse (float or torch.Tensor): Mean squared error between the images.
        max_val (float, optional): Maximum possible pixel value (default: 1.0).

    Returns:
        float: PSNR value in decibels (dB). Returns infinity if MSE is zero.

    Example:
        >>> mse_value = 0.001
        >>> psnr = compute_psnr(mse_value)
    """
    # if mse is a tensor, extract the value
    if isinstance(mse, torch.Tensor):
        mse = mse.item()
    
    # if MSE is zero, return infinite PSNR (perfect match)
    if mse == 0:
        return float('inf')
    
    # calculate PSNR using the standard formula
    psnr = 10 * math.log10(max_val ** 2 / mse)

    return psnr


def compute_psnr_results(pred, target, max_val=1.0):
    """
    Computes the PSNR between a predicted and a ground truth image.

    This function internally calculates the MSE and then applies the PSNR formula.
    Input tensors must have the same shape.

    Args:
        pred (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W).
        target (torch.Tensor): Ground truth image tensor of the same shape.
        max_val (float, optional): Maximum possible pixel value (default: 1.0).

    Returns:
        float: PSNR value in decibels (dB).

    Example:
        >>> psnr = compute_psnr_results(pred, target)
    """

    # initilize the L2 loss function
    mse_fn = MSELoss()

    #calculate the L2 value for the ground truth and prediction
    mse = mse_fn(pred, target).item()

    if mse == 0:
        return float('inf')

    psnr = 10 * math.log10(max_val ** 2 / mse)

    return psnr


def compute_ssim(reconstructed, ground_truth, data_range):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    SSIM is a perceptual metric that evaluates image similarity based on luminance,
    contrast, and structural information. Values close to 1.0 indicate high similarity.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W) or (B, H, W).
        ground_truth (torch.Tensor): Ground truth image tensor of shape (B, C, H, W) or (B, H, W).
        data_range (float): The dynamic range of the input data (e.g., 1.0 or 255).

    Returns:
        float: SSIM value in the range [-1, 1], where 1.0 indicates perfect similarity.

    Example:
        >>> ssim_score = compute_ssim(reconstructed, ground_truth, data_range=1.0)
    """
    
    # Ensure input tensors are 4D: [B, C, H, W]
    if reconstructed.ndim == 3:
        reconstructed = reconstructed.unsqueeze(1)  # Add channel dimension
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.unsqueeze(1)
        
    # both inputs must be shape [B, C, H, W]
    return ssim(reconstructed, ground_truth, data_range).item()


def compute_mse(pred, target):
    return ((pred - target) ** 2).mean().item()