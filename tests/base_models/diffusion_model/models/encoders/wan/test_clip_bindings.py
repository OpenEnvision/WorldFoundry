# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Wan CLIP bindings preserve checkpoint graphs and family precision/kernel policies."""

import importlib
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.encoders.wan import clip
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import (
    action_clip,
    linear_clip,
    liveworld_clip,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import geometry_attention

pytestmark = pytest.mark.unit

MODULES = [clip, action_clip, linear_clip, liveworld_clip]


@pytest.mark.parametrize("module", MODULES, ids=["canonical", "action", "linear", "liveworld"])
def test_direct_embedding_materialization_preserves_family_dtype(module: Any) -> None:
    vision = module.VisionTransformer(
        image_size=8, patch_size=4, dim=8, num_heads=2, num_layers=0, pre_norm=False
    ).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = vision(torch.randn(2, 3, 8, 8))
    assert output.dtype == (torch.bfloat16 if module is clip else torch.float32)
    assert isinstance(vision, clip.VisionTransformer)


@pytest.mark.parametrize("module", [clip, action_clip, liveworld_clip], ids=["canonical", "action", "liveworld"])
def test_layer_norm_keeps_activation_or_weight_dtype_policy(module: Any) -> None:
    norm = module.LayerNorm(8).bfloat16()
    value = torch.randn(2, 3, 8)
    output = norm(value)
    if module is liveworld_clip:
        expected = F.layer_norm(value.bfloat16(), (8,), norm.weight, norm.bias, norm.eps)
        assert output.dtype == torch.bfloat16
    else:
        expected = F.layer_norm(value.float(), (8,), norm.weight.float(), norm.bias.float(), norm.eps)
        assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_linear_clip_retains_mixed_affine_dtype_error() -> None:
    with pytest.raises(RuntimeError, match="dtype"):
        linear_clip.LayerNorm(8).bfloat16()(torch.randn(2, 3, 8))


def test_liveworld_norm_requires_its_affine_weight() -> None:
    with pytest.raises(AttributeError, match="dtype"):
        liveworld_clip.LayerNorm(8, elementwise_affine=False)(torch.randn(2, 3, 8))


def test_attention_shells_bind_family_backends() -> None:
    assert clip.SelfAttention(8, 2)._attention_fn is None
    assert action_clip.SelfAttention(8, 2)._attention_fn is action_clip.flash_attention
    assert linear_clip.SelfAttention(8, 2)._attention_fn is linear_clip.flash_attention
    live_block = liveworld_clip.AttentionBlock(8, 2, 2)
    assert type(live_block.attn) is clip.SelfAttention
    assert live_block.attn._attention_fn is None
    assert liveworld_clip.AttentionPool(8, 2, 2)._attention_fn is geometry_attention.flash_attention


def test_liveworld_pool_keeps_the_fixed_fa2_call() -> None:
    pool = liveworld_clip.AttentionPool(8, 2, 2).eval()
    calls: list[dict[str, Any]] = []

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    pool._attention_fn = attention
    assert pool(torch.randn(2, 5, 8)).shape == (2, 8)
    assert calls == [{"version": 2}]


@pytest.mark.parametrize("module", MODULES[1:], ids=["action", "linear", "liveworld"])
def test_xlm_roberta_clip_reuses_the_native_checkpoint_graph(module: Any) -> None:
    kwargs: dict[str, Any] = {
        "embed_dim": 8,
        "image_size": 8,
        "patch_size": 4,
        "vision_dim": 16,
        "vision_heads": 4,
        "vision_layers": 2,
        "vocab_size": 25,
        "max_text_len": 12,
        "text_dim": 16,
        "text_heads": 4,
        "text_layers": 2,
        "text_dropout": 0,
    }
    torch.manual_seed(23)
    expected = clip.XLMRobertaCLIP(**kwargs)
    torch.manual_seed(23)
    actual = module.XLMRobertaCLIP(**kwargs)
    assert isinstance(actual, clip.XLMRobertaCLIP)
    assert isinstance(actual.visual, clip.VisionTransformer)
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_liveworld_public_factory_resolves_the_canonical_variant() -> None:
    module = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.wan.modules.clip"
    )
    assert module.XLMRobertaCLIP is liveworld_clip.XLMRobertaCLIP
    assert "XLMRobertaCLIP" in module.__all__
