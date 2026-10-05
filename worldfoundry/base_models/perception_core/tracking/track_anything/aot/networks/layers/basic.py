# This file includes code originally from the Segment and Track Anything repository:
# https://github.com/z-x-yang/Segment-and-Track-Anything
# Licensed under the AGPL-3.0 License. See repository-root THIRD-PARTY-NOTICES for details.

"""Module for base_models -> perception_core -> tracking -> track_anything -> aot -> networks -> layers -> basic.py functionality."""

import torch
import torch.nn.functional as F
from torch import nn


class GroupNorm1D(nn.Module):
    """Group norm d implementation."""
    def __init__(self, indim, groups=8):
        """Init.

        Args:
            indim: The indim.
            groups: The groups.
        """
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class GNActDWConv2d(nn.Module):
    """Gn act dw conv d implementation."""
    def __init__(self, indim, gn_groups=32):
        """Init.

        Args:
            indim: The indim.
            gn_groups: The gn groups.
        """
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, indim)
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)

    def forward(self, x, size_2d):
        """Forward.

        Args:
            x: The x.
            size_2d: The size 2d.
        """
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class DWConv2d(nn.Module):
    """Dw conv d implementation."""
    def __init__(self, indim, dropout=0.1):
        """Init.

        Args:
            indim: The indim.
            dropout: The dropout.
        """
        super().__init__()
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x, size_2d):
        """Forward.

        Args:
            x: The x.
            size_2d: The size 2d.
        """
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class ScaleOffset(nn.Module):
    """Scale offset implementation."""
    def __init__(self, indim):
        """Init.

        Args:
            indim: The indim.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(indim))
        # torch.nn.init.normal_(self.gamma, std=0.02)
        self.beta = nn.Parameter(torch.zeros(indim))

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        if len(x.size()) == 3:
            return x * self.gamma + self.beta
        else:
            return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class ConvGN(nn.Module):
    """Conv gn implementation."""
    def __init__(self, indim, outdim, kernel_size, gn_groups=8):
        """Init.

        Args:
            indim: The indim.
            outdim: The outdim.
            kernel_size: The kernel size.
            gn_groups: The gn groups.
        """
        super().__init__()
        self.conv = nn.Conv2d(indim, outdim, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return self.gn(self.conv(x))


def seq_to_2d(tensor, size_2d):
    """Seq to 2d.

    Args:
        tensor: The tensor.
        size_2d: The size 2d.
    """
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop path.

    Args:
        x: The x.
        drop_prob: The drop prob.
        training: The training.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (
        x.shape[0],
        x.shape[1],
    ) + (1,) * (x.ndim - 2)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def mask_out(x, y, mask_rate=0.15, training=False):
    """Mask out.

    Args:
        x: The x.
        y: The y.
        mask_rate: The mask rate.
        training: The training.
    """
    if mask_rate == 0.0 or not training:
        return x

    keep_prob = 1 - mask_rate
    shape = (
        x.shape[0],
        x.shape[1],
    ) + (1,) * (x.ndim - 2)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x * random_tensor + y * (1 - random_tensor)

    return output


class DropPath(nn.Module):
    """Drop path implementation."""
    def __init__(self, drop_prob=None, batch_dim=0):
        """Init.

        Args:
            drop_prob: The drop prob.
            batch_dim: The batch dim.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        """Drop path.

        Args:
            x: The x.
            drop_prob: The drop prob.
        """
        if drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DropOutLogit(nn.Module):
    """Drop out logit implementation."""
    def __init__(self, drop_prob=None):
        """Init.

        Args:
            drop_prob: The drop prob.
        """
        super(DropOutLogit, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        return self.drop_logit(x, self.drop_prob)

    def drop_logit(self, x, drop_prob):
        """Drop logit.

        Args:
            x: The x.
            drop_prob: The drop prob.
        """
        if drop_prob == 0.0 or not self.training:
            return x
        random_tensor = drop_prob + torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        mask = random_tensor * 1e8 if (x.dtype == torch.float32) else random_tensor * 1e4
        output = x - mask
        return output
