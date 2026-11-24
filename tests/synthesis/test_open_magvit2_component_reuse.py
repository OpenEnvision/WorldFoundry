# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""DCR channel order and checkpoint contracts for the shared MAGVIT2 upsampler."""

from __future__ import annotations

import pytest
import torch
from timm.layers import DropPath

from worldfoundry.base_models.diffusion_model.models.autoencoders.magvit2.upsampling import Upsampler, depth_to_space
from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules import util
from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules.diffusionmodules import (
    improved_model,
)
from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules.transformer import (
    gpt,
)

pytestmark = pytest.mark.unit


def test_magvit2_depth_to_space_uses_dcr_channel_order() -> None:
    x = torch.arange(8).reshape(1, 8, 1, 1)
    expected = torch.tensor([[[[0, 2], [4, 6]], [[1, 3], [5, 7]]]])
    assert torch.equal(depth_to_space(x, 2), expected)
    assert not torch.equal(depth_to_space(x, 2), torch.nn.functional.pixel_shuffle(x, 2))


def test_open_magvit2_decoder_loads_canonical_upsampler_keys() -> None:
    assert improved_model.Upsampler is Upsampler
    torch.manual_seed(12)
    module = Upsampler(3, dim_out=17).eval()
    assert set(module.state_dict()) == {"conv1.weight", "conv1.bias"}
    assert module.conv1.out_channels == 12
    restored = improved_model.Upsampler(3).eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    x = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), module(x), rtol=0, atol=0)
    decoder = improved_model.Decoder(
        ch=32, out_ch=3, in_channels=3, num_res_blocks=1, z_channels=4, ch_mult=(1, 2), resolution=16
    )
    assert isinstance(decoder.up[1].upsample, Upsampler)


def test_open_magvit2_uses_timm_drop_path() -> None:
    assert gpt.DropPath is DropPath
    x = torch.randn(2, 4, 8)
    assert torch.equal(gpt.DropPath(0.5).eval()(x), x)


def test_open_magvit2_condition_encoders_need_no_abstract_wrapper() -> None:
    assert util.Labelator.__bases__ == (torch.nn.Module,)
    assert util.SOSProvider.__bases__ == (torch.nn.Module,)
    labels = torch.tensor([3, 207])
    encoder = util.Labelator(1000)
    assert not encoder.state_dict()
    output, _, quantized = encoder.encode(labels)
    assert torch.equal(output, labels[:, None])
    assert torch.equal(quantized[-1], labels[:, None])
    sos = util.SOSProvider(0, quantize_interface=False)
    assert torch.equal(sos.encode(labels), torch.zeros(2, 1, dtype=torch.long))
