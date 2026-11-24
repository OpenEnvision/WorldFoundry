# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Activation functions
# -----------------------------------------------------------------------------


def activate_head_gs(out, activation="norm_exp", conf_activation="expp1", conf_dim=None):
    """
    Process network output to extract GS params and density values.
    Density could be view-dependent as SH coefficient


    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    conf_dim = 1 if conf_dim is None else conf_dim
    xyz = fmap[:, :, :, :-conf_dim]
    conf = fmap[:, :, :, -1] if conf_dim == 1 else fmap[:, :, :, -conf_dim:]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    elif conf_activation == "linear":
        conf_out = conf
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


# -----------------------------------------------------------------------------
# Other utilities
# -----------------------------------------------------------------------------


class Permute(nn.Module):
    """nn.Module wrapper around Tensor.permute for cleaner nn.Sequential usage."""

    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x.permute(*self.dims)


# Inspired by https://github.com/microsoft/moge


# -----------------------------------------------------------------------------
# Interpolation (safe interpolation, avoid INT_MAX overflow)
# -----------------------------------------------------------------------------
def custom_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Safe interpolation implementation to avoid INT_MAX overflow in torch.nn.functional.interpolate.
    """
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks]
        return torch.cat(outs, dim=0).contiguous()

    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
