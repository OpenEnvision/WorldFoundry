# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Frame-wise convolution and resampling preserve image-checkpoint tensor layouts."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.animatediff.resnet import (
    Downsample3D,
    InflatedConv3d,
    InflatedGroupNorm,
    Upsample3D,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inflated_conv_loads_conv2d_weights_and_preserves_frames(dtype: torch.dtype) -> None:
    reference = torch.nn.Conv2d(4, 6, 3, padding=1).to(dtype)
    layer = InflatedConv3d(4, 6, 3, padding=1).to(dtype)
    layer.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 4, 3, 5, 7).to(dtype)
    frames = x.permute(0, 2, 1, 3, 4).reshape(6, 4, 5, 7)
    with torch.no_grad():
        expected = reference(frames).reshape(2, 3, 6, 5, 7).permute(0, 2, 1, 3, 4)
        actual = layer(x)
    assert layer.weight.ndim == 4
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_inflated_group_norm_normalizes_each_frame() -> None:
    reference = torch.nn.GroupNorm(2, 4)
    layer = InflatedGroupNorm(2, 4)
    layer.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 4, 3, 5, 7)
    frames = x.permute(0, 2, 1, 3, 4).reshape(6, 4, 5, 7)
    with torch.no_grad():
        expected = reference(frames).reshape(2, 3, 4, 5, 7).permute(0, 2, 1, 3, 4)
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_resamplers_keep_temporal_axis_and_spatial_checkpoint_keys(dtype: torch.dtype) -> None:
    up = Upsample3D(4, use_conv=True, out_channels=6).to(dtype)
    down = Downsample3D(6, use_conv=True, out_channels=4).to(dtype)
    x = torch.randn(1, 4, 3, 4, 6).to(dtype)
    with torch.no_grad():
        enlarged = up(x)
        restored = down(enlarged)
        explicit = up(x, output_size=(3, 7, 9))
    assert enlarged.shape == (1, 6, 3, 8, 12)
    assert restored.shape == x.shape
    assert explicit.shape == (1, 6, 3, 7, 9)
    assert explicit.dtype == dtype
    assert set(up.state_dict()) == {"conv.weight", "conv.bias"}
    assert set(down.state_dict()) == {"conv.weight", "conv.bias"}
