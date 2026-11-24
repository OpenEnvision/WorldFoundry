# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Canonical Wan projections retain family normalization and initialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan import clip as canonical_clip
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants import liveworld_clip
from worldfoundry.base_models.diffusion_model.models.networks.wan import reference_21
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import camera_21
from worldfoundry.synthesis.visual_generation.infinite_world.infinite_world_runtime.models import dit_model
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.wan.modules import (
    clip,
    model_state_adapter,
)
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.networks import wan2pt1

pytestmark = pytest.mark.unit


def test_liveworld_and_infinite_world_use_canonical_wan_leaves() -> None:
    for module in (model_state_adapter, dit_model):
        assert module.WanRMSNorm is reference_21.WanRMSNorm
        assert module.sinusoidal_embedding_1d is reference_21.sinusoidal_embedding_1d
    assert model_state_adapter.MLPProj is camera_21.MLPProj
    assert model_state_adapter.WanLayerNorm is camera_21.WanLayerNorm
    assert model_state_adapter.rope_params is camera_21.rope_params
    assert dit_model.MLPProj is reference_21.MLPProj
    assert not Path(model_state_adapter.__file__).with_name("model.py").exists()


def test_lyra2_norm_inherits_canonical_math_and_resets_scale() -> None:
    assert wan2pt1.WanRMSNorm.forward is reference_21.WanRMSNorm.forward
    module = wan2pt1.WanRMSNorm(16)
    reference = reference_21.WanRMSNorm(16)
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight))
        reference.load_state_dict(module.state_dict(), strict=True)
        x = torch.randn(2, 3, 16)
        torch.testing.assert_close(module(x), reference(x), atol=0, rtol=0)
    weight = module.weight
    module.reset_parameters()
    assert module.weight is weight
    torch.testing.assert_close(module.weight, torch.ones_like(module.weight), atol=0, rtol=0)


def test_lyra2_projector_inherits_canonical_graph_and_resets_parameters() -> None:
    assert wan2pt1.MLPProj.forward is reference_21.MLPProj.forward
    module = wan2pt1.MLPProj(16, 8)
    reference = reference_21.MLPProj(16, 8)
    reference.load_state_dict(module.state_dict(), strict=True)
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        torch.testing.assert_close(module(x), reference(x), atol=0, rtol=0)
    parameters = dict(module.named_parameters())
    module.init_weights()
    assert all(module.state_dict()[name].shape == parameter.shape for name, parameter in parameters.items())
    assert all(dict(module.named_parameters())[name] is parameter for name, parameter in parameters.items())


def test_liveworld_clip_reuses_fa2_projections_and_preserves_weight_dtype_norm() -> None:
    assert clip.XLMRobertaCLIP is liveworld_clip.XLMRobertaCLIP
    assert liveworld_clip.SelfAttention is canonical_clip.SelfAttention
    norm = liveworld_clip.LayerNorm(16).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        actual = norm(x)
        expected = torch.nn.functional.layer_norm(
            x.to(dtype=norm.weight.dtype), norm.normalized_shape, norm.weight, norm.bias, norm.eps
        )
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
