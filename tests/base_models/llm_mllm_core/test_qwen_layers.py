# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen leaves preserve checkpoint names and low-precision boundaries."""

import pytest
import torch
from transformers import PretrainedConfig

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import (
    QwenMLP,
    QwenRMSNorm,
    get_unpad_data,
    repeat_kv,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("affine_dtype", [torch.float32, torch.bfloat16])
def test_qwen_rms_norm_casts_before_affine(dtype: torch.dtype, affine_dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    norm = QwenRMSNorm(8, eps=1e-5).to(affine_dtype)
    state = {"weight": torch.randn(8).to(affine_dtype)}
    norm.load_state_dict(state, strict=True)
    x = torch.randn(2, 3, 8).to(dtype)
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = state["weight"] * normalized.to(dtype)
    actual = norm(x)
    assert actual.dtype == expected.dtype
    assert list(norm.state_dict()) == ["weight"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("intermediate_size", [None, 12])
def test_qwen_mlp_keeps_gated_projection_checkpoint_names(bias: bool, intermediate_size: int | None) -> None:
    config = PretrainedConfig(hidden_size=8, intermediate_size=20, hidden_act="silu")
    model = QwenMLP(config, bias=bias, intermediate_size=intermediate_size).eval()
    width = 20 if intermediate_size is None else intermediate_size
    expected_shapes = {
        "gate_proj.weight": (width, 8),
        "up_proj.weight": (width, 8),
        "down_proj.weight": (8, width),
    }
    if bias:
        expected_shapes.update({"gate_proj.bias": (width,), "up_proj.bias": (width,), "down_proj.bias": (8,)})
    assert config.intermediate_size == 20
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected_shapes
    x = torch.randn(2, 3, 8)
    expected = model.down_proj(torch.nn.functional.silu(model.gate_proj(x)) * model.up_proj(x))
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_qwen_repeat_kv_keeps_head_order(repetitions: int) -> None:
    x = torch.arange(48).reshape(1, 2, 3, 8)
    torch.testing.assert_close(repeat_kv(x, repetitions), x.repeat_interleave(repetitions, dim=1), atol=0, rtol=0)
    if repetitions == 1:
        assert repeat_kv(x, repetitions) is x


def test_get_unpad_data_keeps_empty_rows_and_int32_offsets() -> None:
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    indices, offsets, maximum = get_unpad_data(mask)
    torch.testing.assert_close(indices, torch.tensor([1, 2, 3, 8, 9]), atol=0, rtol=0)
    torch.testing.assert_close(offsets, torch.tensor([0, 3, 3, 5], dtype=torch.int32), atol=0, rtol=0)
    assert maximum == 3


@pytest.mark.parametrize("bias", [False, True])
def test_hunyuan_vl_mot_decoder_uses_qwen_mlp_for_both_modalities(bias: bool) -> None:
    from worldfoundry.synthesis.action_generation.hy_embodied_vla.configuration_hunyuan_vl_mot import (
        HunYuanVLMoTTextConfig,
    )
    from worldfoundry.synthesis.action_generation.hy_embodied_vla.modeling_hunyuan_vl_mot import (
        HunYuanVLMoTDecoderLayer,
    )

    config = HunYuanVLMoTTextConfig(
        hidden_size=16, intermediate_size=24, num_attention_heads=2, num_key_value_heads=1, mlp_bias=bias
    )
    layer = HunYuanVLMoTDecoderLayer(config, layer_idx=0)
    for mlp in (layer.mlp, layer.mlp_v):
        assert type(mlp) is QwenMLP
        assert (mlp.gate_proj.bias is not None) == bias
        assert mlp.gate_proj.weight.shape == (24, 16)
