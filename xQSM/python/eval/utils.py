import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from math import exp, log10, sqrt
from scipy.ndimage import gaussian_filter

def psnr(img1, img2):
    mse = np.mean((img1 - img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img2)
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True, volume = False):
    # print(img1.size())
    if volume:
        (_, channel, _, _, _) = img1.size()
    else:
        (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def zero_padding(field, factor=8):
    """
    PyTorch implementation of zero-padding to make dimensions divisible by factor.
    Args:
        field: Input tensor (D, H, W)
        factor: Divisor requirement (default=8)
    Returns:
        padded_field: Padded tensor
        pos: Padding positions [(start_D, end_D), ...] for each dimension
    """
    field = field.squeeze(0)
    im_size = torch.tensor(field.shape)
    # print(f"Thois is the im_size {im_size}")
    up_size = torch.ceil(im_size.float() / factor) * factor
    up_size = up_size.long()  # Convert to integer
    # print(f"This is the up_size {up_size}")

    # Calculate padding amounts (half before, half after)
    pad_total = up_size - im_size
    # print(f"pad_total {pad_total}")
    pos_init = torch.ceil(pad_total.float() / 2).long()
    pos_end = pos_init + im_size
    
    # Create padded tensor
    padded_field = torch.zeros(tuple(up_size), dtype=field.dtype, device=field.device)
    # print(f"This is the pos_init {pos_init}")
    # print(f"This is the pos_end {pos_end}")
    # print(f"This is the padded_field {padded_field.shape}")
    # Fill the original data in the center
    padded_field[
        pos_init[0]:pos_end[0],
        pos_init[1]:pos_end[1],
        pos_init[2]:pos_end[2]
    ] = field
    
    # Store padding positions
    pos = torch.stack([pos_init, pos_end], dim=1)  # Shape: [3, 2]
    
    return padded_field.unsqueeze(0), pos
    
def zero_removing(padded_tensor, pos):
    """
    ZeroRemoving: inverse function of ZeroPadding;
    """
    # print(f"Zero Removing")
    # print(f"This is the padded_tensor shape {padded_tensor.shape}")
    pos_init = pos[:, 0]
    pos_end = pos[:, 1]
    # print(f"This is the pos_init {pos_init}")
    # print(f"This is the pos_end {pos_end}")
    Field = padded_tensor[pos_init[0]:pos_end[0], pos_init[1]:pos_end[1], pos_init[2]:pos_end[2]]
    # print(f"This is the field shape {Field.shape}")
    return Field

# Calculate xSIM

def compute_xsim(img1, img2, mask=None, sw=None):
    """
    Python Reimplementation of the MATLAB code provided by the Carlo Milovic.
    Compute xSIM between two 3D images.
    Args:
        img1: Reference image
        img2: Target image
        mask: Mask image
        sw: Gaussian window size
    Returns:
        mean_xssim: Mean SSIM with the mask applied
        mean_ssim: Mean SSIM without the mask applied
        ssim_map: SSIM map
    """
    
    if img1.shape != img2.shape:
        raise ValueError("Reference and Target images do not have matching dimensions!")
    
    s = img1.shape
    
    # Argument parsing
    if sw is None:
        sw = np.array([3, 3, 3])
    else:
        sw = np.array(sw)
    
    if np.any(np.array(s) < sw):
        raise ValueError("Gaussian window is larger than the image!")
    
    if mask is None:
        mask = (img2 != 0)
    
    K = np.array([0.01, 0.001])
    L = 1.0
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    
    # Convert to float for Gaussian filtering
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Calculate truncate parameter to match MATLAB's FilterSize

    # Truncate Calculation: truncate_val = sw[0] / 1.5 ensures that the kernel size matches MATLAB's FilterSize = 2*sw + 1
    # Mathematical Relationship:
    # SciPy: kernel_size = 2 * round(truncate * sigma) + 1
    # MATLAB: FilterSize = 2*sw + 1
    # To match: 2 * round(truncate * 1.5) + 1 = 2*sw + 1 → truncate = sw / 1.5
    truncate_val = sw[0] / 1.5  # Use first dimension for calculation
    
    # Compute Gaussian filtered images with exact window size matching MATLAB
    mu1 = gaussian_filter(img1_float, sigma=1.5, mode='nearest', truncate=truncate_val)
    mu2 = gaussian_filter(img2_float, sigma=1.5, mode='nearest', truncate=truncate_val)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(img1_float*img1_float, sigma=1.5, mode='nearest', truncate=truncate_val) - mu1_sq
    sigma2_sq = gaussian_filter(img2_float*img2_float, sigma=1.5, mode='nearest', truncate=truncate_val) - mu2_sq
    sigma12 = gaussian_filter(img1_float*img2_float, sigma=1.5, mode='nearest', truncate=truncate_val) - mu1_mu2
    
    # Compute SSIM map
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    mean_ssim = ssim_map.mean()
    
    # Compute mean SSIM over the mask
    mean_xssim = np.sum(ssim_map * mask) / np.sum(mask)
    
    return mean_xssim, mean_ssim, ssim_map