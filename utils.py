import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Default RP length (avoid magic numbers); set to 512 for the current mini-dataset.
DEFAULT_RP_SIZE = 512
# Turn on to crop the tail of the RP (e.g., noisy/padded ending).
CROP_RP_TAIL = False
# Retention ratio when cropping (6/7 by default).
CUTOFF_RATIO = 6 / 7

def detect_ship(binary_mask, aber=False):
    """
    Detects ship regions using erosion and dilation on a binary mask.
    Uses first rising edge (1) and last falling edge (-1) to define start and end.
    """
    if binary_mask.ndim == 1:
        binary_mask = binary_mask.reshape(1, -1)

    binary_mask = binary_mask.float()
    detected = apply_dilation_erosion(binary_mask)
    
    # Compute changes: rising and falling edges
    diff = torch.diff(detected.int(), dim=1)
    changes = torch.zeros_like(detected, dtype=torch.int32)
    changes[:, 1:] = diff

    starts = torch.full((detected.shape[0],), -1, dtype=torch.int32)
    ends = torch.full((detected.shape[0],), -1, dtype=torch.int32)

    for i in range(changes.shape[0]):
        change_i = changes[i]
        rising_edges = torch.where(change_i == 1)[0]
        falling_edges = torch.where(change_i == -1)[0]

        if len(rising_edges) > 0 and len(falling_edges) > 0:
            starts[i] = rising_edges[0].item()
            ends[i] = falling_edges[-1].item()
        else:
            # fallback: max value in smoothed
            max_idx = torch.argmax(binary_mask[i])
            starts[i] = max(0, max_idx - 1)
            ends[i] = min(binary_mask.shape[1] - 1, max_idx + 1)

    if aber:
        ships_pos = [list(range(starts[i], ends[i])) for i in range(len(starts))]
        return ships_pos
    else:
        lengths = ends - starts
        return lengths, starts, ends

def apply_dilation_erosion(x, kernel_size=15):
    """Perform 1D dilation then erosion (opening) to smooth a binary-ish mask."""
    # Dilation
    dilation = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
    dilated = dilation(x)

    # Erosion (via negative trick)
    eroded = -dilation(-dilated)
    return eroded

def uniform_filter_1d(signal, kernel_size=11):
    """Boxcar smoothing of a 1D tensor with configurable kernel size."""
    kernel = torch.ones(kernel_size)/kernel_size
    kernel = kernel.view(1, 1, -1).to(signal.device)
    smoothed_signal = F.conv1d(signal.unsqueeze(1), kernel, padding=int(kernel_size//2)).squeeze(1)
    return smoothed_signal

def get_df_RP_length(df, tresh=0.25, return_first_last=False, kernel_size=11):
    """
    Computes ship length using uniform filter-based detection.
    Uses first 1 and last -1 edge in the smoothed binary mask.
    """
    if isinstance(df, pd.DataFrame):
        global selectRP  # assuming it's set externally
        values = df[selectRP].values
        signal = torch.tensor(values, dtype=torch.float32)
    else:
        signal = df.float() if isinstance(df, torch.Tensor) else torch.tensor(df, dtype=torch.float32)

    smoothed = uniform_filter_1d(signal, kernel_size=kernel_size)

    # Compute threshold per signal
    if smoothed.ndim == 2:
        tresh_vals = tresh * torch.max(smoothed, dim=1, keepdim=True)[0]
        binary_mask = smoothed > tresh_vals
    else:
        tresh_val = tresh * torch.max(smoothed)
        binary_mask = (smoothed > tresh_val).unsqueeze(0)

    lengths, starts, ends = detect_ship(binary_mask)

    if not return_first_last:
        return lengths if smoothed.ndim != 1 else lengths[0]
    else:
        return (lengths, starts, ends) if smoothed.ndim != 1 else (lengths[0], starts[0], ends[0])

def get_expected_len(length, width, va):
    """Projected ship length on the radar line-of-sight given yaw angle va."""
    return abs(np.cos(va))*length + abs(np.sin(va))*width


def gaussian_filter_1d(signal, kernel_size=17, sigma=1.5):
    """
    Applies a 1D Gaussian filter using a differentiable convolution.
    """
    # Create Gaussian kernel
    kernel = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gaussian_kernel = torch.exp(- (kernel ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize

    # Convert to 3D kernel for 1D convolution (batch, channels, width)
    gaussian_kernel = gaussian_kernel.view(1, 1, -1).to(signal.device)
    
    # Apply convolution (keep input shape)
    smoothed_signal = F.conv1d(signal.unsqueeze(1), gaussian_kernel, padding=int(kernel_size // 2)).squeeze(1)
    return smoothed_signal

def mfn_decomposition_2D(RP, sigma, kernel_size=17):
    """Compute MFN (mask–feature-noise) decomposition on batch of RP signals."""
    RP = RP.squeeze()
    if RP.ndim == 1:
        RP = RP.unsqueeze(0)
    num_signals, signal_length = RP.shape

    # Optional cropping only for detection (avoid missing targets on noisy tails)
    if CROP_RP_TAIL:
        mask_det = RP.new_ones(signal_length)
        cutoff = min(int(DEFAULT_RP_SIZE * CUTOFF_RATIO), signal_length)
        mask_det[cutoff:] = 0.
        RP_det = RP * mask_det
    else:
        RP_det = RP

    # Get signal lengths and boundaries for all signals
    lrp, first, last = get_df_RP_length(RP_det, tresh=0.5, return_first_last=True)

    # Create indices as a 2D matrix: shape (num_signals, signal_length)
    indices = torch.tile(torch.arange(signal_length), (num_signals, 1))  # Shape (num_signals, signal_length)

    # --- Compute m component (Mean inside first:last) ---
    in_range = (indices >= first[:, None]) & (indices < last[:, None])  # Boolean mask
    means = torch.sum(RP * in_range, axis=1) / (torch.sum(in_range, axis=1)+1e-2)  # Compute mean only in range

    lpf = torch.zeros_like(RP)
    lpf[in_range] = torch.repeat_interleave(means[:, None], signal_length, axis=1)[in_range]  # Assign mean where in range

    # --- Compute mask ---
    mask = torch.ones_like(RP)

    # Left side mask
    left_mask = indices < first[:, None]  # Boolean mask for left side
    mask[left_mask] = torch.exp(2*(indices - first[:, None]) / (lrp[:, None] / 3))[left_mask]

    # Right side mask
    right_mask = indices >= last[:, None]  # Boolean mask for right side
    mask[right_mask] = torch.exp(2*(last[:, None] - indices) / (lrp[:, None] / 3))[right_mask]
    
    # --- Compute f component ---
    f_comp = gaussian_filter_1d(RP, kernel_size,  sigma) * mask

    # --- Compute n component ---
    n_comp = RP - f_comp

    return lpf, f_comp, n_comp

def f_mse(fx, mx, fy, my):
    """MSE and cosine similarity between two filtered components with masks."""
    if fy.shape[0] > fx.shape[0]:
        fx, mx = fx.repeat(fy.shape[0], 1), mx.repeat(fy.shape[0], 1)
    elif fy.shape[0] < fx.shape[0]:
        fy, my = fy.repeat(fx.shape[0], 1), my.repeat(fx.shape[0], 1)
    assert fx.shape[0] == fy.shape[0] and mx.shape[0] == fx.shape[0]
    num_signals = fx.shape[0]
    mse_matrix = torch.zeros((num_signals))
    cosine_matrix = torch.zeros((num_signals))
    for i in range(num_signals):
        if (mx[i]>0).sum() > (my[i]>0).sum(): # mi wider than mj, enlarge mj with its max
            mi = mx[i]
            mj = my[i]
            mj[mi>0] = mj.max()                
        else:
            mj = my[i]
            mi = mx[i]
            mi[mj>0] = mi.max()
        if (mi>0).sum() == 0:
            fmi, lmi = 0, fx.shape[1]
        else:
            fmi, lmi = torch.argwhere(mi>0)[0][0],  torch.argwhere(mi>0)[-1][0]+1
        mse_matrix[i] = torch.mean((fx[i] - fy[i]) ** 2)/(((mx[i]>0).sum()+(my[i]>0).sum())/2)
        cosine_matrix[i] = cosine_similarity((fx[i]-mi)[fmi:lmi].reshape(1, -1), (fy[i]-mj)[fmi:lmi].reshape(1, -1)).item()
    return mse_matrix, cosine_matrix
