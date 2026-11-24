# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""VGGT shells preserve each reconstruction model's attention and cache bindings."""

from __future__ import annotations

import importlib

import pytest
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.attention import Attention as VGGTAttention
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    NestedTensorBlock as VGGTNestedTensorBlock,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "family",
    [
        "hunyuan_mirror.models.layers",
        "hyworldmirror_2p0.models.layers",
        "stream3r.models.components.layers",
        "infinite_vggt.layers",
        "vggt.vggt.variants.fantasy_world.layers",
    ],
)
def test_nested_interface_binds_family_constructor_and_forward(family: str) -> None:
    module = importlib.import_module("worldfoundry.base_models.three_dimensions.point_clouds." + family + ".block")
    torch.manual_seed(5)
    nested = module.NestedTensorBlock(32, 4, mlp_ratio=2, qk_norm=True).eval()
    torch.manual_seed(5)
    plain = module.Block(32, 4, mlp_ratio=2, qk_norm=True).eval()
    assert nested.forward.__func__ is VGGTNestedTensorBlock.forward
    assert type(nested.attn) is type(plain.attn)
    assert list(nested.state_dict()) == list(plain.state_dict())
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(nested(x), plain(x), rtol=0, atol=0)
    with pytest.raises(AssertionError, match="xFormers is required"):
        nested([x])


def test_mirror_keeps_sdpa_when_its_stored_fused_flag_is_false() -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.layers.attention import Attention

    mirror = Attention(32, 4, qk_norm=True, fused_attn=False).eval()
    canonical = VGGTAttention(32, 4, qk_norm=True, fused_attn=True).eval()
    canonical.load_state_dict(mirror.state_dict(), strict=True)
    assert mirror.fused_attn is False
    torch.manual_seed(9)
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(mirror(x), canonical(x), rtol=0, atol=0)


def test_stream_block_accumulates_key_values_without_replacing_the_shell() -> None:
    from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.models.components.layers.block import Block

    block = Block(32, 4, qk_norm=True).eval()
    x = torch.randn(2, 3, 32)
    first, cache = block(x, kv_cache=[None, None])
    second, cache = block(x + 1, kv_cache=cache)
    assert first.shape == second.shape == x.shape
    assert [value.shape for value in cache] == [(2, 4, 6, 8), (2, 4, 6, 8)]
    restored, fresh = block(x, kv_cache=[None, None])
    torch.testing.assert_close(restored, first, rtol=0, atol=0)
    assert [value.shape for value in fresh] == [(2, 4, 3, 8), (2, 4, 3, 8)]
