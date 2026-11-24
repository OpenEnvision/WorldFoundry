# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pi3X convolutional heads reuse MoGe residuals with unchanged native keys."""

from typing import Literal

import pytest
import torch
from torch import nn

from worldfoundry.base_models.three_dimensions.depth.moge.model.v1 import ResidualConvBlock
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers import conv_head

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "silu", "elu"])
@pytest.mark.parametrize("norm", ["group_norm", "layer_norm"])
def test_pi3_residual_binding_retains_moge_parameters(
    activation: Literal["relu", "leaky_relu", "silu", "elu"], norm: Literal["group_norm", "layer_norm"]
) -> None:
    assert conv_head.ResidualConvBlock is ResidualConvBlock
    block = conv_head.ResidualConvBlock(32, 64, 64, activation=activation, norm=norm)
    assert block.layers[3].num_groups == (2 if norm == "group_norm" else 1)
    assert "layers.2.weight" in block.state_dict()
    assert "skip_connection.weight" in block.state_dict()
    output = block(torch.randn(2, 32, 6, 8))
    assert output.shape == (2, 64, 6, 8)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("using_uv", [False, True])
def test_conv_head_keeps_projection_and_patch_geometry(using_uv: bool) -> None:
    head = conv_head.ConvHead(
        num_features=1,
        dim_in=8,
        dim_out=[3, 1],
        dim_proj=32,
        dim_upsample=[32],
        projects=nn.Linear(8, 32),
        using_uv=using_uv,
    ).eval()
    with torch.no_grad():
        points, confidence = head(torch.randn(2, 6, 8), patch_h=2, patch_w=3)
    assert points.shape == (2, 3, 28, 42)
    assert confidence.shape == (2, 1, 28, 42)
    assert "projects.weight" in head.state_dict()
    assert "upsample_blocks.0.1.layers.2.weight" in head.state_dict()
