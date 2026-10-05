"""Module for base_models -> three_dimensions -> point_clouds -> loger -> utils -> geometry.py functionality."""

import numpy as np
import torch
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.utils.geometry import (
    se3_inverse as se3_inverse,
    get_pixel as get_pixel,
    depthmap_to_absolute_camera_coordinates as depthmap_to_absolute_camera_coordinates,
    depthmap_to_camera_coordinates as depthmap_to_camera_coordinates,
    homogenize_points as homogenize_points,
    get_gt_warp as get_gt_warp,
    warp_kpts as warp_kpts,
    geotrf as geotrf,
    inv as inv,
    opencv_camera_to_plucker as opencv_camera_to_plucker,
    depth_edge as depth_edge,
)

def robust_scale_estimation(ratios: torch.Tensor, trim_ratio: float = 0.25) -> torch.Tensor:
    """
    Compute a robust mean of ratios by trimming the top and bottom trim_ratio fraction.
    Args:
        ratios: (B, N) tensor of ratios
        trim_ratio: fraction to trim from each end (0.0 to 0.5)
    Returns:
        (B,) tensor of robust means
    """
    B, N = ratios.shape
    if N == 0:
        return torch.ones(B, device=ratios.device, dtype=ratios.dtype)
    
    # Sort ratios along the last dimension
    sorted_ratios, _ = torch.sort(ratios, dim=-1)
    
    # Determine indices to keep
    trim_cnt = int(N * trim_ratio)
    start_idx = trim_cnt
    end_idx = N - trim_cnt
    
    if start_idx >= end_idx:
        # Fallback to median if trimming removes everything (shouldn't happen with reasonable N and trim_ratio < 0.5)
        return sorted_ratios[:, N // 2]
        
    # Slice and compute mean
    valid_ratios = sorted_ratios[:, start_idx:end_idx]
    return valid_ratios.mean(dim=-1)
