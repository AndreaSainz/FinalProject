"""
Utility functions for computing image quality metrics used in CT reconstruction.

Includes:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)

These metrics are commonly used to evaluate the similarity between reconstructed
images and ground truth references in image reconstruction tasks.

Dependencies:
    - pytorch_msssim: For SSIM calculation.
    - torch
    - math
"""


from pytorch_msssim import ssim
import math
import torch
from torch.nn import MSELoss


def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a logarithmic metric that compares the ratio between the maximum possible
    pixel value and the mean squared error (MSE) between a reconstructed and a reference image.

    Args:
        mse (float or torch.Tensor): Mean squared error between reconstructed and reference images.
        max_val (float, optional): Maximum possible pixel value (default: 1.0).

    Returns:
        float: PSNR value. Returns infinity if MSE is zero.
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

    mse_fn = MSELoss()
    
    mse = mse_fn(pred, target).item()

    if mse == 0:
        return float('inf')

    psnr = 10 * math.log10(max_val ** 2 / mse)

    return psnr


def compute_ssim(reconstructed, ground_truth, data_range):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    SSIM is a perceptual metric that measures image similarity, considering luminance,
    contrast, and structure. Values close to 1.0 indicate high similarity.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor of shape (B, C, H, W).
        ground_truth (torch.Tensor): Ground truth image tensor of shape (B, C, H, W).

    Returns:
        float: SSIM value between -1 and 1. A value of 1 indicates perfect similarity.
    """
    # both inputs must be shape [B, C, H, W]
    return ssim(reconstructed, ground_truth, data_range).item()
