# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> utils -> math.py functionality."""

import torch


def generate_permute_matrix(dim, num, keep_first=True, gpu_id=0):
    """Generate permute matrix.

    Args:
        dim: The dim.
        num: The num.
        keep_first: The keep first.
        gpu_id: The gpu id.
    """
    all_matrix = []
    for idx in range(num):
        random_matrix = torch.eye(dim, device=torch.device("cuda", gpu_id))
        if keep_first:
            fg = random_matrix[1:][torch.randperm(dim - 1)]
            random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        else:
            random_matrix = random_matrix[torch.randperm(dim)]
        all_matrix.append(random_matrix)
    return torch.stack(all_matrix, dim=0)


def truncated_normal_(tensor, mean=0, std=0.02):
    """Truncated normal.

    Args:
        tensor: The tensor.
        mean: The mean.
        std: The std.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
