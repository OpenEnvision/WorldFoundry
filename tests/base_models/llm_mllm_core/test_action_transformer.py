# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared ACT containers keep per-layer weights and intermediate normalization order."""

import pytest
import torch
from torch import nn

from worldfoundry.base_models.llm_mllm_core.action.transformer import (
    TransformerDecoder,
    TransformerEncoder,
)
from worldfoundry.synthesis.action_generation.tinyvla import transformer

pytestmark = pytest.mark.unit


def test_transformer_encoder_copies_layers_without_sharing_parameters() -> None:
    assert transformer.TransformerEncoder is TransformerEncoder
    layer = transformer.TransformerEncoderLayer(8, 2, 16, dropout=0.0)
    norm = nn.LayerNorm(8)
    model = TransformerEncoder(layer, 2, norm).eval()
    assert model.layers[0].self_attn.in_proj_weight is not model.layers[1].self_attn.in_proj_weight
    torch.testing.assert_close(
        model.layers[0].self_attn.in_proj_weight, model.layers[1].self_attn.in_proj_weight, atol=0, rtol=0
    )
    assert "layers.0.self_attn.in_proj_weight" in model.state_dict()
    assert "norm.weight" in model.state_dict()
    src, pos = torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    with torch.no_grad():
        expected = norm(model.layers[1](model.layers[0](src, pos=pos), pos=pos))
        torch.testing.assert_close(model(src, pos=pos), expected, atol=0, rtol=0)


@pytest.mark.parametrize("return_intermediate", [False, True])
def test_transformer_decoder_preserves_intermediate_layer_stack(return_intermediate: bool) -> None:
    assert transformer.TransformerDecoder is TransformerDecoder
    layer = transformer.TransformerDecoderLayer(8, 2, 16, dropout=0.0)
    norm = nn.LayerNorm(8)
    model = TransformerDecoder(layer, 2, norm, return_intermediate).eval()
    tgt, memory = torch.randn(3, 2, 8), torch.randn(5, 2, 8)
    with torch.no_grad():
        first = model.layers[0](tgt, memory)
        second = norm(model.layers[1](first, memory))
        expected = torch.stack([norm(first), second]) if return_intermediate else second.unsqueeze(0)
        torch.testing.assert_close(model(tgt, memory), expected, atol=0, rtol=0)
