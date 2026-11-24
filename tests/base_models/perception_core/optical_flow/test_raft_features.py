# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""RAFT feature leaves retain native normalization, aliasing and encoder layouts."""

import importlib

import pytest
import torch

from worldfoundry.base_models.perception_core.optical_flow.raft_features import (
    BottleneckBlock,
    ResidualBlock,
    SmallEncoder,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("block_type", [ResidualBlock, BottleneckBlock])
@pytest.mark.parametrize("norm", ["group", "batch", "instance", "none"])
def test_residual_downsample_keeps_the_registered_norm_alias(
    block_type: type[ResidualBlock] | type[BottleneckBlock], norm: str
) -> None:
    block = block_type(32, 64, norm_fn=norm, stride=2)
    assert block.downsample is not None
    assert block.downsample[1] is getattr(block, "norm3" if block_type is ResidualBlock else "norm4")
    output = block(torch.randn(2, 32, 8, 10))
    assert output.shape == (2, 64, 4, 5)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "module",
    [
        "worldfoundry.base_models.perception_core.frame_interpolation.amt.networks.blocks.feat_enc",
        "worldfoundry.base_models.perception_core.optical_flow.flowformerplusplus.core.FlowFormer.PerCostFormer3.cnn",
    ],
)
def test_flow_encoders_bind_the_canonical_residual_block(module: str) -> None:
    assert importlib.import_module(module).ResidualBlock is ResidualBlock


def test_amt_small_encoder_retains_paired_image_batching() -> None:
    module = importlib.import_module("worldfoundry.base_models.perception_core.frame_interpolation.amt.networks.AMT-S")
    assert module.SmallEncoder is SmallEncoder
    encoder = module.SmallEncoder(output_dim=16, norm_fn="instance").eval()
    image = torch.randn(1, 3, 32, 40)
    with torch.no_grad():
        first, second = encoder([image, image])
    assert first.shape == (1, 16, 4, 5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
