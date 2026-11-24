# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared vision leaves retain model-specific projection and normalization layouts."""

from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.vision import (
    Qwen2_5_VisionPatchEmbed,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    apply_rotary_pos_emb_vision,
)

pytestmark = pytest.mark.unit


def test_qwen_vision_patch_bias_is_checkpoint_specific() -> None:
    qwen25 = Qwen2_5_VisionPatchEmbed(patch_size=2, temporal_patch_size=2, in_channels=3, embed_dim=16)
    config = PretrainedConfig(patch_size=2, temporal_patch_size=2, in_channels=3, hidden_size=16)
    qwen3 = Qwen3VLVisionPatchEmbed(config)
    assert set(qwen25.state_dict()) == {"proj.weight"}
    assert set(qwen3.state_dict()) == {"proj.weight", "proj.bias"}
    assert qwen25(torch.randn(4, 24)).shape == qwen3(torch.randn(4, 24)).shape == (4, 16)


@pytest.mark.parametrize("postshuffle", [False, True])
def test_qwen3_patch_merger_keeps_norm_width(postshuffle: bool) -> None:
    config = PretrainedConfig(hidden_size=16, spatial_merge_size=2, out_hidden_size=12)
    merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=postshuffle)
    assert merger.state_dict()["norm.weight"].shape == ((64,) if postshuffle else (16,))
    assert merger(torch.randn(8, 16)).shape == (2, 12)


def test_qwen_vision_rotary_restores_independent_input_dtypes() -> None:
    q = torch.randn(5, 2, 8).bfloat16()
    k = torch.randn(5, 2, 8).half()
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    actual = apply_rotary_pos_emb_vision(q, k, cos, sin)
    for x, output in zip((q, k), actual, strict=True):
        x_float = x.float()
        rotated = torch.cat((-x_float[..., 4:], x_float[..., :4]), dim=-1)
        expected = (x_float * cos[:, None] + rotated * sin[:, None]).to(x.dtype)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_go1_loads_shared_internvit_leaves_without_changing_its_encoder(tmp_path: Path) -> None:
    from worldfoundry.base_models.llm_mllm_core.mllm.intern_vit.model import InternMLP, InternVisionEmbeddings
    from worldfoundry.synthesis.action_generation.go1.configuration_intern_vit import InternVisionConfig
    from worldfoundry.synthesis.action_generation.go1.modeling_intern_vit import (
        InternVisionEncoder,
        InternVisionModel,
    )

    config = InternVisionConfig(
        image_size=8,
        patch_size=2,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        use_flash_attn=False,
    )
    model = InternVisionModel(config).eval()
    assert type(model.embeddings) is InternVisionEmbeddings
    assert type(model.encoder) is InternVisionEncoder
    assert type(model.encoder.layers[0].mlp) is InternMLP
    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = InternVisionModel.from_pretrained(tmp_path).eval()
    assert model.state_dict().keys() == reloaded.state_dict().keys()
    pixels = torch.randn(1, 3, 12, 12)
    with torch.no_grad():
        torch.testing.assert_close(model(pixels).last_hidden_state, reloaded(pixels).last_hidden_state, atol=0, rtol=0)


def test_qwen2_5_vision_block_uses_the_gated_qwen_owner() -> None:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

    from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP
    from worldfoundry.synthesis.action_generation.lingbot_vla.modeling.qwen25_vl import Qwen2_5_VLVisionBlock

    config = Qwen2_5_VLVisionConfig(hidden_size=16, intermediate_size=32, num_heads=2)
    config._attn_implementation = "eager"
    block = Qwen2_5_VLVisionBlock(config)
    assert type(block.mlp) is QwenMLP
    assert {key for key in block.mlp.state_dict() if key.endswith("bias")} == {
        "gate_proj.bias",
        "up_proj.bias",
        "down_proj.bias",
    }
