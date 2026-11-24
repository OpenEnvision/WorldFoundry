# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Canonical vision leaves preserve source initialization and mask arithmetic."""

from __future__ import annotations

import pytest
import torch
from timm.layers import DropPath, drop_path
from torch import nn

from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.construction import (
    BlockChunk,
    init_weights_vit_timm,
)
from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.models.components.layers import (
    SwiGLUFFN,
    SwiGLUFFNFused,
)
from worldfoundry.core.nn.blocks import layers
from worldfoundry.core.nn.blocks.stochastic_depth import UniformDropPath, drop_path_uniform
from worldfoundry.core.nn.blocks.tuples import ntuple

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("probability", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("training", [False, True])
def test_uniform_drop_path_keeps_source_mask_and_division_order(probability: float, training: bool) -> None:
    x = torch.linspace(-1.0, 1.0, 96).reshape(8, 3, 4)
    torch.manual_seed(23)
    if probability == 0 or not training:
        expected = x
    else:
        keep = 1 - probability
        mask = (keep + torch.rand((8, 1, 1))).floor_()
        expected = x.div(keep) * mask
    torch.manual_seed(23)
    torch.testing.assert_close(drop_path_uniform(x, probability, training), expected, rtol=0, atol=0)
    torch.manual_seed(23)
    torch.testing.assert_close(UniformDropPath(probability).train(training)(x), expected, rtol=0, atol=0)


def test_bernoulli_and_uniform_drop_paths_remain_distinct() -> None:
    x = torch.ones(64, 2, 4)
    torch.manual_seed(17)
    bernoulli = DropPath(0.4).train()(x)
    torch.manual_seed(17)
    torch.testing.assert_close(drop_path(x, 0.4, True), bernoulli, rtol=0, atol=0)
    torch.manual_seed(17)
    assert not torch.equal(drop_path_uniform(x, 0.4, True), bernoulli)


def test_stream3r_imports_the_packed_swiglu_owners() -> None:
    assert SwiGLUFFN is layers.SwiGLUFFN
    assert SwiGLUFFNFused is layers.SwiGLUFFNFused
    plain = SwiGLUFFN(8, 20)
    fused = SwiGLUFFNFused(8, 20)
    assert list(plain.state_dict()) == ["w12.weight", "w12.bias", "w3.weight", "w3.bias"]
    assert plain.w12.weight.shape == (40, 8)
    assert fused.w12.weight.shape == (32, 8)


def test_dino_initializer_does_not_reset_non_linear_modules() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Conv1d(2, 2, 1))
    with torch.no_grad():
        model[1].weight.fill_(3)
        model[2].weight.fill_(7)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.apply(init_weights_vit_timm)
    assert not torch.equal(model[0].weight, before["0.weight"])
    assert torch.count_nonzero(model[0].bias) == 0
    for key in ["1.weight", "1.bias", "2.weight", "2.bias"]:
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)


def test_block_chunk_preserves_flat_modulelist_checkpoint_indices() -> None:
    chunk = BlockChunk([nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)])
    assert list(chunk.state_dict()) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(chunk(x), chunk[2](chunk[1](chunk[0](x))), rtol=0, atol=0)


@pytest.mark.parametrize("value", [[1, 2], (1, 2), range(2)])
def test_ntuple_retains_iterable_configuration_objects(value: object) -> None:
    assert ntuple(2)(value) is value
    assert ntuple(2)(3) == (3, 3)
    assert ntuple(2)("xy") == ("xy", "xy")
