import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim


def compute_psnr(mse, max_val=1.0):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        mse (torch.Tensor or float): The mean squared error between reconstructed and reference images.
        max_val (float, optional): The maximum possible pixel value of the images (default: 1.0).

    Returns:
        float: PSNR value in decibels (dB). Returns infinity if MSE is zero.
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

def compute_ssim(reconstructed, reference):
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        reconstructed (torch.Tensor): The reconstructed or predicted image tensor with shape [Batch size, Channels, Height, Weight].
        reference (torch.Tensor): The ground truth or reference image tensor with shape [B, C, H, W].

    Returns:
        float: SSIM value between -1 and 1, where 1 means perfect similarity.
    """
    # both inputs must be shape [B, C, H, W]
    return ssim(reconstructed, reference).item()