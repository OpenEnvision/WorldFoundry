# Adapted from Tencent Hunyuan-GameCraft-1.0, under the Tencent Hunyuan Community License.
from typing import List, Union

import torch

from worldfoundry.core.attention import get_1d_rotary_pos_embed, get_meshgrid_nd


def get_nd_rotary_pos_embed_new(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    concat_dict={},
):
    """
    Generates multi-dimensional Rotary Position Embeddings (RoPE).

    Creates position embeddings for n-dimensional spaces by generating a meshgrid
    of positions and applying 1D rotary embeddings to each dimension, then combining them.

    Args:
        rope_dim_list (list): List of embedding dimensions for each axis
        start: Starting dimensions for generating the meshgrid
        *args: Additional arguments for meshgrid generation
        theta (float): Base theta parameter for RoPE frequency calculation
        use_real (bool): If True, returns separate cosine and sine embeddings
        theta_rescale_factor: Rescaling factor(s) for theta (per dimension)
        interpolation_factor: Interpolation factor(s) for position scaling (per dimension)
        concat_dict: Dictionary for special concatenation modes (e.g., time-based extensions)

    Returns:
        tuple or tensor: Cosine and sine embeddings if use_real=True, combined embedding otherwise
    """
    # Generate n-dimensional meshgrid of positions (shape: [dim, *sizes])
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    # Handle special concatenation modes (e.g., adding time-based bias)
    if concat_dict:
        if concat_dict["mode"] == "timecat":
            # Add bias as first element in first dimension
            bias = grid[:, :1].clone()
            bias[0] = concat_dict["bias"] * torch.ones_like(bias[0])
            grid = torch.cat([bias, grid], dim=1)
        elif concat_dict["mode"] == "timecat-w":
            # Add biased first element with spatial offset
            bias = grid[:, :1].clone()
            bias[0] = concat_dict["bias"] * torch.ones_like(bias[0])
            bias[2] += start[-1]  # Spatial offset reference: OminiControl implementation
            grid = torch.cat([bias, grid], dim=1)

    # Normalize theta rescale factors to list format (per dimension)
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), (
        "Length of theta_rescale_factor must match number of dimensions"
    )

    # Normalize interpolation factors to list format (per dimension)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), (
        "Length of interpolation_factor must match number of dimensions"
    )

    # Generate 1D rotary embeddings for each dimension and combine
    embs = []
    for i in range(len(rope_dim_list)):
        # Flatten grid dimension and generate embeddings
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),  # Flatten to 1D positions
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    # Combine embeddings from all dimensions
    if use_real:
        # Return separate cosine and sine components
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        # Return combined embedding
        return torch.cat(embs, dim=1)
