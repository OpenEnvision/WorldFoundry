# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Codec component identity, checkpoint loading and cached decoding contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import reference_21
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants import camera_21, taehv
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils import (
    taehv as inspatio_taehv,
)
from worldfoundry.synthesis.visual_generation.krea_realtime.krea_runtime.runtime_utils import vae_block3
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.wan.modules import vae

pytestmark = pytest.mark.unit


def test_liveworld_vae_checkpoint_and_cached_decode(tmp_path: Path) -> None:
    assert vae.Encoder3d is reference_21.Encoder3d
    assert vae.Decoder3d is reference_21.Decoder3d
    assert vae.count_conv3d is reference_21.count_conv3d
    config = {
        "dim": 4,
        "z_dim": 4,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True, True],
    }
    reference = reference_21.WanVAE_(**config).eval()
    checkpoint = tmp_path / "vae.pth"
    torch.save(reference.state_dict(), checkpoint)
    model = vae._video_vae(pretrained_path=checkpoint, **config).eval()
    assert model.state_dict().keys() == reference.state_dict().keys()
    assert len(model._feat_map) == reference_21.count_conv3d(model.decoder)
    latent = torch.randn(1, 4, 2, 2, 2)
    with torch.no_grad():
        expected = reference.decode(latent, [0.0, 1.0])
        decoded = model.decode(latent, [0.0, 1.0])
        model.clear_cache()
        cached = torch.cat([model.cached_decode(chunk, [0.0, 1.0]) for chunk in latent.split(1, dim=2)], dim=2)
    assert decoded.shape == (1, 3, 5, 16, 16)
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)
    torch.testing.assert_close(cached, expected, atol=0, rtol=0)


def test_krea_resample_retains_zero_tensor_cache() -> None:
    assert vae_block3.Resample is camera_21.Resample
    module = vae_block3.Resample(4, "upsample3d").eval()
    cache: list[torch.Tensor | None] = [None]
    with torch.no_grad():
        first = module(torch.randn(1, 4, 1, 3, 3), feat_cache=cache, feat_idx=[0])
        assert isinstance(cache[0], torch.Tensor)
        assert torch.count_nonzero(cache[0]) == 0
        following = module(torch.randn(1, 4, 1, 3, 3), feat_cache=cache, feat_idx=[0])
    assert first.shape == (1, 2, 1, 6, 6)
    assert following.shape == (1, 2, 2, 6, 6)


def test_inspatio_taehv_uses_canonical_memory_blocks() -> None:
    for name in ("conv", "Clamp", "MemBlock", "TPool", "TGrow"):
        assert getattr(inspatio_taehv, name) is getattr(taehv, name)
    model = torch.nn.Sequential(taehv.MemBlock(4, 4), taehv.TGrow(4, 2), taehv.MemBlock(4, 3)).eval()
    latent = torch.randn(1, 2, 4, 2, 2)
    with torch.no_grad():
        expected = inspatio_taehv.apply_model_with_memblocks_sequential(model, latent, False)
        queue = [inspatio_taehv.TWorkItem(frame, 0) for frame in latent.unbind(1)]
        memory = [None] * len(model)
        frames = []
        while queue:
            frame = inspatio_taehv.apply_model_with_memblocks_sequential_single_step(model, memory, queue)
            if frame is not None:
                frames.append(frame)
    torch.testing.assert_close(torch.cat(frames, dim=1), expected, atol=0, rtol=0)
