# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Wan codec leaves retain cache slots, normalization and family wrapper dispatch."""

import importlib

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import model, reference_22

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module_name",
    [
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_22",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.action_21",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.camera_21",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.geometry_bridge",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.linear",
    ],
)
def test_wan_variants_use_canonical_leaf_types(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.CausalConv3d is model.CausalConv3d
    assert module.RMS_norm is model.RMS_norm
    assert module.Upsample is model.Upsample


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_magi_resample_preserves_first_and_cached_chunk_layout(dtype: torch.dtype) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.magi2 import video_vae

    assert video_vae.Resample is reference_22.Resample
    assert video_vae.RMSNorm is model.RMS_norm
    assert video_vae.CausalConv3d is model.CausalConv3d
    resample = video_vae.Resample(4, "upsample3d").to(dtype)
    value = torch.randn(1, 4, 2, 3, 5, dtype=dtype)
    cache: list[torch.Tensor | str | None] = [None]
    first = resample(value, feat_cache=cache, feat_idx=[0])
    assert cache == ["Rep"]
    second = resample(value, feat_cache=cache, feat_idx=[0])
    assert first.shape == (1, 4, 2, 6, 10)
    assert second.shape == (1, 4, 4, 6, 10)
    assert first.dtype == second.dtype == dtype
    assert isinstance(cache[0], torch.Tensor)
    torch.testing.assert_close(cache[0], value, rtol=0, atol=0)


def test_resample_keeps_native_weight_replacement_initializers() -> None:
    resample = reference_22.Resample(4, "upsample3d")
    weight = resample.time_conv.weight
    resample.init_weight2(resample.time_conv)
    assert resample.time_conv.weight is not weight
    torch.testing.assert_close(resample.time_conv.weight[:4, :, -1, 0, 0], torch.eye(4), rtol=0, atol=0)
    torch.testing.assert_close(resample.time_conv.weight[4:, :, -1, 0, 0], torch.eye(4), rtol=0, atol=0)


@pytest.mark.parametrize(
    "module_name",
    [
        "worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.vaes.wan_video_vae",
        "worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.models.wan_video_vae",
    ],
)
def test_wan_video_vae38_reuses_construction_and_keeps_local_dispatch(module_name: str) -> None:
    module = importlib.import_module(module_name)
    vae = module.WanVideoVAE38(dim=4).eval()
    assert isinstance(vae, model.WanVideoVAE38)
    assert isinstance(vae, module.WanVideoVAE)
    assert vae.decode.__func__ is module.WanVideoVAE.decode
    assert vae.upsampling_factor == 16
    assert vae.z_dim == 48
    assert vae.mean.shape == vae.std.shape == (48,)
    assert not any(parameter.requires_grad for parameter in vae.parameters())
    assert all(key.startswith("model.") for key in vae.state_dict())
    with torch.no_grad():
        encoded = vae.encode(torch.randn(1, 3, 1, 32, 32), "cpu")
        decoded = vae.decode(encoded, "cpu")
    assert encoded.shape == (1, 48, 1, 2, 2)
    assert decoded.shape == (1, 3, 1, 32, 32)


def test_flash_world_image_embedding_keeps_native_precision_norms() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.image_embedding import (
        WanImageEmbedding,
    )
    from worldfoundry.base_models.three_dimensions.point_clouds.flash_world import transformer_wan

    projection = transformer_wan.WanImageEmbedding(8, 12, pos_embed_seq_len=6).bfloat16().eval()
    assert isinstance(projection, WanImageEmbedding)
    assert isinstance(projection.norm1, transformer_wan.FP32LayerNorm)
    assert isinstance(projection.norm2, transformer_wan.FP32LayerNorm)
    value = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    assert projection.pos_embed is not None
    expected = projection.norm2(projection.ff(projection.norm1(value.view(2, 6, 8) + projection.pos_embed)))
    torch.testing.assert_close(projection(value), expected, rtol=0, atol=0)
