# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""VGGT-Omega reuses DINOv3 while retaining its query/key normalization graph."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.perception_core.general_perception.dinov3.layers.attention import SelfAttention
from worldfoundry.base_models.perception_core.general_perception.dinov3.layers.block import SelfAttentionBlock
from worldfoundry.base_models.perception_core.general_perception.dinov3.layers.ffn_layers import ListMlp
from worldfoundry.base_models.perception_core.general_perception.dinov3.layers.rope_position_encoding import (
    RopePositionEmbedding,
)
from worldfoundry.base_models.perception_core.general_perception.dinov3.variants.vggt_omega import (
    SelfAttentionBlock as OmegaBlock,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("qk_norm", [False, True])
def test_attention_normalizes_queries_and_keys_before_rotary_embedding(qk_norm: bool) -> None:
    torch.manual_seed(11)
    attention = SelfAttention(32, 4, qkv_bias=True, use_qk_norm=qk_norm).eval()
    rotary = RopePositionEmbedding(32, num_heads=4, dtype=torch.float32).eval()(H=2, W=3)
    x = torch.randn(2, 8, 32)
    q, k, v = attention.qkv(x).reshape(2, 8, 3, 4, 8).unbind(2)
    q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
    if qk_norm:
        assert isinstance(attention.q_norm, nn.LayerNorm)
        assert isinstance(attention.k_norm, nn.LayerNorm)
        q, k = attention.q_norm(q), attention.k_norm(k)
        assert list(attention.state_dict())[:4] == ["q_norm.weight", "q_norm.bias", "k_norm.weight", "k_norm.bias"]
    else:
        assert not any("_norm." in key for key in attention.state_dict())
    q, k = attention.apply_rope(q, k, rotary)
    expected = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 8, 32)
    expected = attention.proj(expected)
    torch.testing.assert_close(attention(x, rope=rotary), expected, rtol=0, atol=0)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_omega_block_reuses_dino_forward_and_keeps_strict_keys(qk_norm: bool) -> None:
    assert OmegaBlock.forward is SelfAttentionBlock.forward
    block = OmegaBlock(32, 4, use_qk_norm=qk_norm, init_values=0.1).eval()
    restored = OmegaBlock(32, 4, use_qk_norm=qk_norm, init_values=0.1).eval()
    restored.load_state_dict(block.state_dict(), strict=True)
    assert isinstance(block.mlp, ListMlp)
    assert ("attn.q_norm.weight" in block.state_dict()) is qk_norm
    x = [torch.randn(2, 6, 32), torch.randn(1, 4, 32)]
    torch.testing.assert_close(restored(x), block(x), rtol=0, atol=0)


def test_list_mlp_preserves_projection_names_and_variable_token_shapes() -> None:
    mlp = ListMlp(8, 12, 6).eval()
    x = [torch.randn(2, 3, 8), torch.randn(1, 4, 8)]
    actual = mlp.forward_list(x)
    assert [value.shape for value in actual] == [(2, 3, 6), (1, 4, 6)]
    flattened = torch.cat([value.flatten(0, -2) for value in x])
    expected = mlp(flattened)
    torch.testing.assert_close(torch.cat([value.flatten(0, -2) for value in actual]), expected, rtol=0, atol=0)
    assert list(mlp.state_dict()) == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
