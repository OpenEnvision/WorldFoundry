# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.gr00t.modeling.dit import AdaLayerNorm, TimestepEncoder
from worldfoundry.synthesis.action_generation.mem0 import cross_attention_dit
from worldfoundry.synthesis.action_generation.spirit_v15 import modeling as spirit

pytestmark = pytest.mark.unit


def test_action_consumers_reuse_gr00t_conditioning_leaves() -> None:
    assert cross_attention_dit.AdaLayerNorm is spirit.AdaLayerNorm is AdaLayerNorm
    assert cross_attention_dit.TimestepEncoder is spirit.TimestepEncoder is TimestepEncoder


@pytest.mark.parametrize("affine", [False, True])
def test_ada_layer_norm_preserves_affine_slots_and_stored_chunk_option(affine: bool) -> None:
    layer = AdaLayerNorm(8, norm_elementwise_affine=affine, chunk_dim=1)
    assert layer.chunk_dim == 1
    expected = ["linear.weight", "linear.bias"] + (["norm.weight", "norm.bias"] if affine else [])
    assert list(layer.state_dict()) == expected
    x, temb = torch.randn(2, 3, 8), torch.randn(2, 8)
    scale, shift = layer.linear(layer.silu(temb)).chunk(2, dim=1)
    torch.testing.assert_close(layer(x, temb), layer.norm(x) * (1 + scale[:, None]) + shift[:, None], rtol=0, atol=0)


def test_timestep_encoder_uses_parameter_dtype_not_constructor_hint() -> None:
    layer = TimestepEncoder(8, compute_dtype=torch.bfloat16)
    assert next(layer.parameters()).dtype == torch.float32
    assert list(layer.state_dict()) == [
        "timestep_embedder.linear_1.weight",
        "timestep_embedder.linear_1.bias",
        "timestep_embedder.linear_2.weight",
        "timestep_embedder.linear_2.bias",
    ]
    layer.to(torch.bfloat16)
    assert layer(torch.tensor([1, 2])).dtype == torch.bfloat16
