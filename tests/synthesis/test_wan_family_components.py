# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Family loaders retain Wan checkpoint layouts while sharing the codec graph."""

from __future__ import annotations

import importlib

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import model as wan_vae
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import WanTextEncoder

pytestmark = pytest.mark.unit

VAE_MODULES = (
    "worldfoundry.synthesis.action_generation.ahawam.wan_video_vae",
    "worldfoundry.synthesis.action_generation.fastwam.wan_video_vae",
    "worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.vaes.wan_video_vae",
    "worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.models.wan_video_vae",
)


@pytest.mark.parametrize("module_name", VAE_MODULES)
@pytest.mark.parametrize(
    "wrapper_name,model_class", [("WanVideoVAE", wan_vae.VideoVAE_), ("WanVideoVAE38", wan_vae.VideoVAE38_)]
)
def test_family_vae_builds_canonical_checkpoint_graph(module_name: str, wrapper_name: str, model_class: type) -> None:
    module = importlib.import_module(module_name)
    with torch.device("meta"):
        wrapper = getattr(module, wrapper_name)()
        canonical = getattr(wan_vae, wrapper_name)()
    assert type(wrapper.model) is model_class
    assert wrapper.state_dict().keys() == canonical.state_dict().keys()
    for key, tensor in wrapper.state_dict().items():
        assert tensor.shape == canonical.state_dict()[key].shape, key


def test_ahawam_loader_builds_canonical_text_encoder() -> None:
    from worldfoundry.synthesis.action_generation.ahawam import modeling

    assert modeling.WanTextEncoder is WanTextEncoder


def test_worldcam_offload_classes_match_canonical_codec() -> None:
    from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.pipelines import wan_video_new

    assert wan_video_new.CausalConv3d is wan_vae.CausalConv3d
    assert wan_video_new.RMS_norm is wan_vae.RMS_norm
    assert wan_video_new.Upsample is wan_vae.Upsample


@pytest.mark.parametrize("family", ["ahawam", "fastwam"])
def test_wam_video_and_action_graphs_bind_family_attention(family: str) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.wam import (
        NativeDtypeHead,
        WamDiTBlock,
    )
    from worldfoundry.core.nn import RMSNorm

    module = importlib.import_module(f"worldfoundry.synthesis.action_generation.{family}.wan_video_dit")
    action_module = importlib.import_module(f"worldfoundry.synthesis.action_generation.{family}.action_dit")
    shared = {
        "hidden_dim": 24,
        "ffn_dim": 48,
        "text_dim": 12,
        "freq_dim": 16,
        "eps": 1e-6,
        "num_heads": 2,
        "attn_head_dim": 8,
        "num_layers": 1,
    }
    if family == "ahawam":
        shared.update(rope_max_length=32, rope_theta=10000.0)
    video = module.WanVideoDiT(
        **shared, in_dim=4, out_dim=4, patch_size=(1, 2, 2), has_image_input=False, seperated_timestep=True
    ).eval()
    action = action_module.ActionDiT(**shared, action_dim=7).eval()
    for model in [video, action]:
        block = model.blocks[0]
        assert type(block) is WamDiTBlock
        assert block.self_attn.attention_fn is module.flash_attention
        assert block.self_attn.rope_fn is module.rope_apply
        assert block.cross_attn.attention_fn is module.flash_attention
        assert type(block.self_attn.norm_q) is RMSNorm
        assert block.self_attn.q.weight.shape == (16, 24)
        incompatible = model.load_state_dict(model.state_dict(), strict=True)
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
    assert type(video.head) is NativeDtypeHead
    x = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 3, 12)
    mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        expected = video(x, timestep, context, context_mask=mask, fuse_vae_embedding_in_latents=True)
        video(x + 1, timestep, context + 1, context_mask=mask, fuse_vae_embedding_in_latents=True)
        actual = video(x, timestep, context, context_mask=mask, fuse_vae_embedding_in_latents=True)
        action_output = action(torch.randn(1, 4, 7), timestep, context, context_mask=mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == x.shape
    assert action_output.shape == (1, 4, 7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wam_head_preserves_native_dtype_modulation(dtype: torch.dtype) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.wam import NativeDtypeHead

    head = NativeDtypeHead(16, 4, (1, 2, 2), 1e-6).to(dtype).eval()
    x = torch.randn(2, 3, 16, dtype=dtype)
    timestep = torch.randn_like(x)
    shift, scale = (head.modulation.unsqueeze(0) + timestep.unsqueeze(2)).chunk(2, dim=2)
    expected = head.head(head.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
    torch.testing.assert_close(head(x, timestep), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wam_gate_preserves_unfused_residual(dtype: torch.dtype) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.wam import TorchGateModule

    x = torch.randn(2, 3, 16, dtype=dtype)
    gate = torch.randn(2, 1, 16, dtype=dtype)
    residual = torch.randn_like(x)
    actual = TorchGateModule()(x, gate, residual)
    torch.testing.assert_close(actual, x + gate * residual, atol=0, rtol=0)


@pytest.mark.parametrize(
    "available,expected",
    [
        ((True, True, True), "flash_attention_3"),
        ((False, True, True), "flash_attention_2"),
        ((False, False, True), "sage_attention"),
    ],
)
def test_worldcam_preserves_explicit_provider_priority(
    monkeypatch: pytest.MonkeyPatch, available: tuple[bool, bool, bool], expected: str
) -> None:
    from functools import partial

    from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.models import wan_video_dit

    called = []

    def record(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        called.append(name)
        assert kwargs == {
            "q_pattern": "b s (n d)",
            "k_pattern": "b s (n d)",
            "v_pattern": "b s (n d)",
            "out_pattern": "b s (n d)",
            "dims": {"n": 2},
        }
        return q

    for flag, enabled in zip(
        ["FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"], available, strict=True
    ):
        monkeypatch.setattr(wan_video_dit, flag, enabled)
    for provider in ["flash_attention_3", "flash_attention_2", "sage_attention"]:
        monkeypatch.setattr(wan_video_dit, provider, partial(record, provider))
    q = torch.randn(1, 4, 16)
    assert wan_video_dit.flash_attention(q, q, q, num_heads=2) is q
    assert called == [expected]


@pytest.mark.parametrize("compatibility_mode", [False, True])
def test_worldcam_keeps_exact_sdpa_fallback(monkeypatch: pytest.MonkeyPatch, compatibility_mode: bool) -> None:
    from worldfoundry.core.attention import scaled_dot_product_attention
    from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.models import wan_video_dit

    for flag in ["FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"]:
        monkeypatch.setattr(wan_video_dit, flag, compatibility_mode)
    q, k, v = (torch.randn(1, 4, 16) for _ in range(3))
    actual = wan_video_dit.flash_attention(q, k, v, num_heads=2, compatibility_mode=compatibility_mode)
    split = [value.reshape(1, 4, 2, 8).transpose(1, 2) for value in [q, k, v]]
    expected = scaled_dot_product_attention(*split).transpose(1, 2).reshape(1, 4, 16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
