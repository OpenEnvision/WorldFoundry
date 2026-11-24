# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""AdaWorld's diffusion leaves preserve their LVDM checkpoint and dispatch contracts."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_adaworld_geglu_keeps_feedforward_checkpoint_names() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.lvdm.attention import GEGLU
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.attention import FeedForward

    layer = FeedForward(8, glu=True, mult=2).eval()
    assert type(layer.net[0]) is GEGLU
    assert set(layer.state_dict()) == {"net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"}
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        projected, gate = layer.net[0].proj(x).chunk(2, dim=-1)
        expected = layer.net[2](projected * torch.nn.functional.gelu(gate))
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_adaworld_group_norm_and_decomposed_silu_use_lvdm_owner() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.lvdm.blocks import Normalize, nonlinearity
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules import attention
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules import model

    assert attention.Normalize is model.Normalize is Normalize
    assert model.nonlinearity is nonlinearity
    norm = attention.Normalize(32)
    assert set(norm.state_dict()) == {"weight", "bias"}
    assert norm.num_groups == 32
    assert norm.eps == 1e-6
    x = torch.randn(2, 32, 3, 3)
    torch.testing.assert_close(model.nonlinearity(x), x * torch.sigmoid(x), atol=0, rtol=0)


def test_adaworld_timestep_dispatch_recognizes_canonical_base() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.lvdm.unet3d import TimestepBlock
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules.openaimodel import (
        ResBlock,
        TimestepEmbedSequential,
    )

    residual = ResBlock(channels=32, emb_channels=16, dropout=0).eval()
    assert isinstance(residual, TimestepBlock)
    with torch.no_grad():
        for parameter in residual.parameters():
            parameter.normal_(0, 0.1)
        x, emb = torch.randn(2, 32, 3, 3), torch.randn(2, 16)
        expected = residual(x, emb)
        actual = TimestepEmbedSequential(residual)(x, emb)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
