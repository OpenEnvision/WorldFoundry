# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Helios history/norm components and Warp-as-History processor bindings."""

import importlib
import subprocess
import sys

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from worldfoundry.base_models.diffusion_model.models.networks.helios import history, normalization
from worldfoundry.base_models.diffusion_model.models.networks.helios.model import HeliosAttention

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "owner",
    [
        "worldfoundry.core.nn.blocks.volume",
        "worldfoundry.core.nn.diffusion.utils",
        "worldfoundry.base_models.diffusion_model.models.networks.helios.history",
        "worldfoundry.base_models.diffusion_model.models.networks.helios.normalization",
    ],
)
def test_helper_import_avoids_model_and_accelerator_initializers(owner: str) -> None:
    code = f"""
import importlib
import sys
import torch
# Diffusers may initialize Dynamo/Triton independently of the repository owner.
if {owner!r}.endswith('helios.normalization'):
    importlib.import_module('diffusers.models.normalization')
before = set(sys.modules)
importlib.import_module({owner!r})
assert 'worldfoundry.base_models.diffusion_model.models.networks.helios.model' not in sys.modules
assert not any(name == 'triton' or name.startswith('triton.') for name in set(sys.modules) - before)
assert not any(name in sys.modules for name in ('hydra', 'omegaconf', 'wandb'))
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_fp32_rms_replacement_preserves_parameters_and_native_output_dtype(dtype: torch.dtype) -> None:
    model = torch.nn.Sequential(torch.nn.RMSNorm(8, eps=1e-6), RMSNorm(8, eps=1e-5, bias=True)).to(dtype)
    original_parameters = tuple(model.parameters())
    original_keys = list(model.state_dict())
    assert normalization.replace_rmsnorm_with_fp32(model) is model
    assert normalization.replace_rmsnorm_with_fp32(model) is model
    assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
    assert list(model.state_dict()) == original_keys
    x = torch.randn(2, 3, 8, dtype=dtype)
    expected = x
    for layer in model:
        assert isinstance(layer, (torch.nn.RMSNorm, RMSNorm))
        assert isinstance(layer.weight, torch.Tensor)
        expected = torch.nn.functional.rms_norm(expected.float(), (8,), layer.weight.float(), layer.eps)
        bias = getattr(layer, "bias", None)
        if bias is not None:
            assert isinstance(bias, torch.Tensor)
            expected = expected + bias.float()
        expected = expected.to(dtype)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert model(x).dtype == dtype


@pytest.mark.parametrize("npu,modern,message", [(True, True, "NPU"), (False, False, "2.4")])
def test_fp32_rms_preserves_device_and_version_errors(
    monkeypatch: pytest.MonkeyPatch, npu: bool, modern: bool, message: str
) -> None:
    monkeypatch.setattr(normalization, "is_torch_npu_available", lambda: npu)
    monkeypatch.setattr(normalization, "is_torch_version", lambda *args: modern)
    with pytest.raises(ValueError, match=message):
        normalization.FP32RMSNorm(8, eps=1e-6)(torch.ones(1, 8))


def test_history_masks_pool_in_float32_and_preserve_identity() -> None:
    assert history.pool_history_visible_mask(None, (2, 2, 2)) is None
    assert history.resolve_history_keep_mask(None) is None
    mask = torch.ones(1, 3, 5, 7, dtype=torch.bfloat16)
    pooled = history.pool_history_visible_mask(mask, (2, 2, 2))
    assert pooled is not None
    assert pooled.shape == (1, 1, 2, 3, 4)
    assert pooled.dtype == torch.float32
    keep = history.resolve_history_keep_mask(pooled)
    assert keep is not None
    assert keep.dtype == torch.bool
    assert keep.shape == (24,)
    assert keep.all()


@pytest.mark.parametrize(
    "mask,message",
    [
        (torch.ones(1, 2, 3), "reduce to"),
        (torch.ones(1, 2, 1, 1, 1), "channel dimension"),
        (torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]]), "identical masks"),
    ],
)
def test_history_masks_preserve_supported_batch_errors(mask: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        history.resolve_history_keep_mask(mask)


@torch.no_grad()
def test_warp_attention_binds_native_history_key_boost_processor() -> None:
    warp = importlib.import_module(
        "worldfoundry.synthesis.visual_generation.warp_as_history.warp_as_history_runtime."
        "helios.diffusers_version.transformer_helios_diffusers"
    )
    torch.manual_seed(5)
    layer = warp.HeliosAttention(dim=32, heads=2, dim_head=16).eval()
    assert isinstance(layer, HeliosAttention)
    assert layer.processor is None
    assert layer._default_processor_cls is warp.HeliosAttnProcessor
    assert layer._available_processors == [warp.HeliosAttnProcessor]
    layer.set_processor(warp.HeliosAttnProcessor())
    expected_keys = list(layer.state_dict())
    hidden = torch.randn(1, 5, 32)
    original = layer(hidden, original_context_length=3)
    layer.history_key_boost_mask = torch.tensor([True, False, False, False, False])
    layer.history_key_boost_scale = 2.0
    boosted = layer(hidden, original_context_length=3)
    assert not torch.equal(original, boosted)
    layer.fuse_projections()
    assert "to_qkv.weight" in layer.state_dict()
    layer.unfuse_projections()
    assert list(layer.state_dict()) == expected_keys
    torch.testing.assert_close(layer(hidden, original_context_length=3), boosted, rtol=0, atol=0)
