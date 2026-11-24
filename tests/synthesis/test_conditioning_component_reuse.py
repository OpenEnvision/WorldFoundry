# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared conditioning variants preserve checkpoint names and precision conventions."""

from __future__ import annotations

import importlib

import pytest
import torch
from diffusers.models.normalization import FP32LayerNorm

from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import FinalLayer
from worldfoundry.base_models.diffusion_model.models.networks.sana.variants.cpu_frequency import (
    CpuFrequencySizeEmbedder,
    CpuFrequencyTimestepEmbedder,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.image_embedding import WanImageEmbedding
from worldfoundry.synthesis.visual_generation.nwm import models as nwm
from worldfoundry.synthesis.visual_generation.open_oasis import dit as oasis

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module_name",
    [
        "worldfoundry.synthesis.visual_generation.uniworld_view.uniworld_view_runtime.model.uniview_transformer",
        "worldfoundry.synthesis.action_generation.giga_world_policy.transformer",
    ],
)
def test_wan_image_conditioning_loads_fp32_variant(module_name: str) -> None:
    runtime = importlib.import_module(module_name)
    assert runtime.WanImageEmbedding is WanImageEmbedding
    torch.manual_seed(8)
    source = runtime.WanTimeTextImageEmbedding(16, 8, 32, 12, image_embed_dim=8).eval().to(torch.bfloat16)
    restored = runtime.WanTimeTextImageEmbedding(16, 8, 32, 12, image_embed_dim=8).eval().to(torch.bfloat16)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert isinstance(restored.image_embedder.norm1, FP32LayerNorm)
    image_keys = {k.removeprefix("image_embedder.") for k in restored.state_dict() if k.startswith("image_embedder.")}
    assert image_keys == {
        "norm1.weight",
        "norm1.bias",
        "ff.net.0.proj.weight",
        "ff.net.0.proj.bias",
        "ff.net.2.weight",
        "ff.net.2.bias",
        "norm2.weight",
        "norm2.bias",
    }
    args = (torch.tensor([1.0, 3.0]), torch.randn(2, 4, 12).bfloat16(), torch.randn(2, 3, 8).bfloat16())
    with torch.no_grad():
        for expected, actual in zip(source(*args), restored(*args), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wan_image_first_last_frame_position_table_keeps_keys() -> None:
    torch.manual_seed(9)
    module = WanImageEmbedding(8, 16, pos_embed_seq_len=6).eval()
    assert module.pos_embed is not None
    module.pos_embed.data.normal_()
    x = torch.randn(2, 3, 8)
    expected = module.norm2(module.ff(module.norm1(x.reshape(1, 6, 8) + module.pos_embed)))
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    assert module.state_dict()["pos_embed"].shape == (1, 6, 8)


def test_open_oasis_timestep_projection_does_not_cast_frequency_inputs() -> None:
    assert oasis.TimestepEmbedder is CpuFrequencyTimestepEmbedder
    module = oasis.TimestepEmbedder(16, frequency_embedding_size=9).bfloat16().eval()
    t = torch.tensor([0.0, 1.5, 1000.0])
    frequencies = module.timestep_embedding(t, 9)
    assert frequencies.dtype == torch.float32
    assert frequencies.shape == (3, 9)
    assert torch.equal(frequencies[:, -1], torch.zeros(3))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(module(t), module.mlp(frequencies), rtol=0, atol=0)


def test_open_sora_size_projection_keeps_batch_repeat_and_affine_keys() -> None:
    torch.manual_seed(10)
    source = CpuFrequencySizeEmbedder(8).bfloat16().eval()
    restored = CpuFrequencySizeEmbedder(8).bfloat16().eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    assert set(source.state_dict()) == {"mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"}
    sizes = torch.tensor([[720.0, 1280.0]])
    with torch.no_grad():
        actual = restored(sizes, 2)
        torch.testing.assert_close(actual, source(sizes.repeat(2, 1), 2), rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 16)


def test_nwm_final_layer_is_canonical_sana() -> None:
    assert nwm.FinalLayer is FinalLayer
