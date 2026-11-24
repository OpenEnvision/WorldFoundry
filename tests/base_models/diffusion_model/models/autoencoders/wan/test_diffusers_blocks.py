# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Diffusers-layout Wan blocks preserve native caches and FlashWorld spatial bindings."""

import inspect
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import diffusers_blocks
from worldfoundry.base_models.three_dimensions.point_clouds.flash_world import autoencoder_kl_wan as flash
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules import (
    autoencoder_kl_wan as longcat,
)

pytestmark = pytest.mark.unit


def _spatial_runtime(module: nn.Module) -> None:
    """Apply FlashWorld's runtime temporal-kernel slicing to the reduced graph."""
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, flash.WanCausalConv3d):
                padding = cast(tuple[int, ...], child._padding)
                time_pad = padding[4]
                child.padding = (0, padding[2], padding[0])
                child._padding = (0, 0, 0, 0, 0, 0)
                child.weight = nn.Parameter(child.weight[:, :, time_pad:].clone())


@pytest.mark.parametrize(
    "name", ["WanResample", "WanResidualBlock", "WanAttentionBlock", "WanMidBlock", "WanUpBlock", "WanResidualUpBlock"]
)
def test_longcat_uses_canonical_diffusers_layout_blocks(name: str) -> None:
    assert getattr(longcat, name) is getattr(diffusers_blocks, name)


def test_flash_world_binds_its_mutating_blocks_and_single_frame_shortcut() -> None:
    mid = flash.WanMidBlock(4)
    assert isinstance(mid, diffusers_blocks.WanMidBlock)
    assert type(mid.resnets[0]) is flash.WanResidualBlock
    assert type(mid.attentions[0]) is flash.WanAttentionBlock
    up = flash.WanResidualUpBlock(8, 4, 1, up_flag=True, temperal_upsample=True)
    assert type(up.avg_shortcut) is flash.DupUp3D
    assert type(up.upsampler) is flash.WanResample
    assert type(up.upsampler.time_conv) is flash.WanCausalConv3d


@pytest.mark.parametrize("name", ["WanResample", "WanMidBlock", "WanUpBlock", "WanResidualUpBlock"])
def test_mutable_cache_index_defaults_remain_family_local(name: str) -> None:
    flash_default = inspect.signature(getattr(flash, name).forward).parameters["feat_idx"].default
    native_default = inspect.signature(getattr(longcat, name).forward).parameters["feat_idx"].default
    assert flash_default is not native_default


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_resample_advances_and_resets_temporal_cache(dtype: torch.dtype) -> None:
    resample = diffusers_blocks.WanResample(4, "upsample3d").to(dtype)
    value = torch.randn(1, 4, 1, 3, 5, dtype=dtype)
    cache: list[torch.Tensor | str | None] = [None]
    first = resample(value, cache, [0])
    assert cache == ["Rep"]
    second = resample(value, cache, [0])
    third = resample(value, cache, [0])
    assert first.shape == (1, 2, 1, 6, 10)
    assert second.shape == third.shape == (1, 2, 2, 6, 10)
    torch.testing.assert_close(cache[0], value.repeat(1, 1, 2, 1, 1), rtol=0, atol=0)
    torch.testing.assert_close(resample(value, [None], [0]), first, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rms_keeps_scaled_gamma_multiplication(dtype: torch.dtype) -> None:
    norm = flash.WanRMS_norm(6, bias=True).to(dtype)
    assert isinstance(norm.gamma, nn.Parameter)
    assert isinstance(norm.bias, nn.Parameter)
    with torch.no_grad():
        norm.gamma.uniform_(0.3, 1.7)
        norm.bias.uniform_(-0.5, 0.5)
    value = torch.randn(2, 6, 3, 4, dtype=dtype)
    expected = F.normalize(value, dim=1).mul_(norm.scale * norm.gamma).add_(norm.bias)
    torch.testing.assert_close(norm(value), expected, rtol=0, atol=0)


def test_flash_causal_adapter_keeps_its_spatial_runtime_contract() -> None:
    conv = flash.WanCausalConv3d(3, 4, 3, padding=1)
    value = torch.randn(1, 3, 1, 4, 6)
    assert conv(value).shape == (1, 4, 3, 6, 8)
    _spatial_runtime(conv)
    assert conv.weight.shape[2] == 1
    assert conv._padding == (0, 0, 0, 0, 0, 0)
    assert conv.padding == (0, 1, 1)
    assert conv(value).shape == (1, 4, 1, 4, 6)


def test_flash_residual_preserves_input_mutation() -> None:
    block = flash.WanResidualBlock(4, 4).eval()
    _spatial_runtime(block)
    value = torch.randn(1, 4, 1, 3, 5)
    before = value.clone()
    with torch.no_grad():
        result = block(value)
    assert result is value
    assert not torch.equal(value, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_longcat_reduced_vae_decodes_cached_video_and_clears_state(dtype: torch.dtype) -> None:
    vae = longcat.AutoencoderKLWan(base_dim=4, z_dim=4, num_res_blocks=1).to(dtype).eval()
    video = torch.randn(1, 3, 5, 16, 16, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(video).latent_dist.mode()
        decoded = vae.decode(latent).sample
    assert latent.shape == (1, 4, 2, 2, 2)
    assert decoded.shape == video.shape
    assert all(value is None for value in vae._feat_map)
    assert all(value is None for value in vae._enc_feat_map)


def test_flash_reduced_spatial_vae_keeps_single_image_layout() -> None:
    vae = flash.AutoencoderKLWan(
        base_dim=4, z_dim=4, num_res_blocks=1, is_residual=True, in_channels=12, out_channels=12, patch_size=2
    ).eval()
    _spatial_runtime(vae)
    image = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent).sample
    assert latent.shape == (1, 4, 1, 2, 2)
    assert decoded.shape == image.shape
