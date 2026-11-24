# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for shared ViT QKV attention paths."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from worldfoundry.core.nn.transformer.vit_qkv import QKNormRopeSelfAttention, QKVSelfAttention

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("attention_cls", [QKVSelfAttention, QKNormRopeSelfAttention])
def test_eager_qkv_matches_scaled_query_matmuls(attention_cls: type[QKVSelfAttention]) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        attention = attention_cls(32, num_heads=4, qkv_bias=True).eval()
        attention.fused_attn = False
        x = torch.randn(2, 7, 32)
        qkv = attention.qkv(x).reshape(2, 7, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * attention.scale, qkv[1], qkv[2]
        expected = attention.proj(((q @ k.transpose(-2, -1)).softmax(dim=-1) @ v).transpose(1, 2).reshape(2, 7, 32))

        torch.testing.assert_close(attention(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(attention._forward(x), expected, rtol=0, atol=0)


def test_qkv_default_retains_sdpa_and_mask() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        attention = QKVSelfAttention(32, num_heads=4).eval()
        assert attention.fused_attn
        x = torch.randn(2, 7, 32)
        mask: Tensor = torch.ones(7, 7, dtype=torch.bool).tril()
        q, k, v = attention.qkv(x).reshape(2, 7, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
        expected = attention.proj(F.scaled_dot_product_attention(q, k, v, mask).transpose(1, 2).reshape(2, 7, 32))

        torch.testing.assert_close(attention(x, mask), expected, rtol=0, atol=0)
