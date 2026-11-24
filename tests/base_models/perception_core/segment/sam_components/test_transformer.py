# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SAM generations share a two-way shell without sharing kernel or MLP key layouts."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
from torch import nn

from worldfoundry.base_models.perception_core.segment.sam_components.transformer import (
    Attention,
    Sam1Attention,
    Sam1TwoWayTransformer,
    Sam3Attention,
    Sam3TwoWayTransformer,
    TwoWayTransformer,
)
from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.core.nn.blocks.layers import SamHeadMLP, SamMLPBlock

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cls", [Attention, Sam1Attention, Sam3Attention])
def test_attention_keeps_its_generation_kernel(cls: type[Attention]) -> None:
    torch.manual_seed(5)
    model = cls(32, 4, downsample_rate=2).eval()
    q, k, v = torch.randn(2, 3, 32), torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    projected = [
        layer(x).reshape(2, x.shape[1], 4, 4).transpose(1, 2)
        for layer, x in zip([model.q_proj, model.k_proj, model.v_proj], [q, k, v], strict=True)
    ]
    pq, pk, pv = projected
    if cls is Sam1Attention:
        expected = ((pq @ pk.transpose(-2, -1)) / math.sqrt(4)).softmax(-1) @ pv
    else:
        expected = scaled_dot_product_attention(pq, pk, pv, dropout_p=0.0)
    expected = model.out_proj(expected.transpose(1, 2).reshape(2, 3, 16))
    torch.testing.assert_close(model(q, k, v), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("cls", "mlp_cls", "first_weight"),
    [
        (TwoWayTransformer, SamHeadMLP, "layers.0.mlp.layers.0.weight"),
        (Sam1TwoWayTransformer, SamMLPBlock, "layers.0.mlp.lin1.weight"),
        (Sam3TwoWayTransformer, SamMLPBlock, "layers.0.mlp.lin1.weight"),
    ],
)
def test_two_way_shell_preserves_generation_keys(
    cls: type[TwoWayTransformer], mlp_cls: type[nn.Module], first_weight: str
) -> None:
    assert cls.forward is TwoWayTransformer.forward
    model = cls(2, 32, 4, 48).eval()
    restored = cls(2, 32, 4, 48).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(model.layers[0].mlp, mlp_cls)
    assert first_weight in model.state_dict()
    assert list(restored.state_dict()) == list(model.state_dict())
    inputs = (torch.randn(2, 32, 2, 3), torch.randn(2, 32, 2, 3), torch.randn(2, 5, 32))
    actual = model(*inputs)
    assert [value.shape for value in actual] == [(2, 5, 32), (2, 6, 32)]
    torch.testing.assert_close(restored(*inputs), actual, rtol=0, atol=0)


def test_shared_sam_imports_do_not_initialize_model_families_or_hydra() -> None:
    code = """
import sys
from worldfoundry.base_models.perception_core.segment.sam_components import (
    mask_decoder, mask_ops, memory, prompt_encoder, transformer
)
assert 'hydra' not in sys.modules
assert not any('segment.' + family in name for family in ['sam2', 'sam3', 'sam_v1'] for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
