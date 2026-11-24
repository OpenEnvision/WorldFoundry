# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""rCM reuses Wan layers without reintroducing FP32 interpolation buffers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import reference_21
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.rcm import Decoder3d
from worldfoundry.core.vram.initialization import init_weights_on_device
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.tokenizers.wan2pt1 import WanVAE_

pytestmark = pytest.mark.unit


def test_rcm_vae_uses_reference_graph_and_native_dtype_upsampling() -> None:
    with torch.device("meta"):
        model = WanVAE_(dim=8, z_dim=4, num_res_blocks=1)
        reference = reference_21.WanVAE_(dim=8, z_dim=4, num_res_blocks=1)
    assert type(model.encoder) is reference_21.Encoder3d
    assert type(model.decoder) is Decoder3d
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in model.state_dict().items():
        assert value.shape == reference.state_dict()[key].shape
    for layer in model.decoder.upsamples:
        if isinstance(layer, reference_21.Resample):
            assert type(layer.resample[0]) is nn.Upsample
            pixels = torch.randn(1, 8, 2, 2, dtype=torch.bfloat16)
            assert layer.resample[0](pixels).dtype == torch.bfloat16


@pytest.mark.parametrize("include_buffers", [False, True])
def test_canonical_weight_placement_restores_registration_hooks(include_buffers: bool) -> None:
    parameter_hook = nn.Module.register_parameter
    buffer_hook = nn.Module.register_buffer
    with init_weights_on_device(include_buffers=include_buffers):
        model = nn.Linear(3, 4)
        model.register_buffer("buffer", torch.ones(2))
    assert model.weight.device.type == "meta"
    assert model.buffer.device.type == ("meta" if include_buffers else "cpu")
    assert nn.Module.register_parameter is parameter_hook
    assert nn.Module.register_buffer is buffer_hook


@pytest.mark.parametrize("model_type", ["t2v", "i2v", "flf2v"])
def test_rcm_dit_constructs_canonical_projection_and_resettable_norm(model_type: str) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_21 import MLPProj
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.rcm import WanRMSNorm
    from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.networks.wan2pt1 import WanModel

    with torch.device("meta"):
        model = WanModel(
            model_type=model_type,
            in_dim=4,
            dim=48,
            ffn_dim=96,
            freq_dim=16,
            text_dim=12,
            out_dim=4,
            num_heads=2,
            num_layers=1,
        )
    assert type(model.blocks[0].self_attn.norm_q) is WanRMSNorm
    assert model.state_dict()["blocks.0.self_attn.norm_q.weight"].shape == (48,)
    if model_type != "t2v":
        assert type(model.img_emb) is MLPProj
        assert model.state_dict()["img_emb.proj.1.weight"].shape == (1280, 1280)
    if model_type == "flf2v":
        assert model.state_dict()["img_emb.emb_pos"].shape == (1, 514, 1280)


def test_rcm_resettable_norm_preserves_reference_forward() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_21 import (
        WanRMSNorm as ReferenceWanRMSNorm,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.rcm import WanRMSNorm

    norm = WanRMSNorm(16).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.zero_()
    norm.reset_parameters()
    reference = ReferenceWanRMSNorm(16).to(torch.bfloat16)
    value = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(norm(value), reference(value), atol=0, rtol=0)


def test_rcm_reuses_all_to_all_and_rotary_tensor_operations() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.gamma_world import context_parallel
    from worldfoundry.base_models.diffusion_model.models.networks.magi2 import preview_layers
    from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils import a2a_cp, rope

    assert a2a_cp.post_all2all is context_parallel.post_all2all
    assert a2a_cp.single_all_to_all is context_parallel.single_all_to_all
    assert rope.apply_rotary_emb_torch is preview_layers.apply_rotary_emb_torch
    value = torch.arange(96).reshape(2, 1, 3, 2, 8)
    actual = a2a_cp.post_all2all(True, 2)(value)
    assert actual.shape == (1, 6, 2, 8)
    torch.testing.assert_close(actual, value.reshape(1, 6, 2, 8), atol=0, rtol=0)
