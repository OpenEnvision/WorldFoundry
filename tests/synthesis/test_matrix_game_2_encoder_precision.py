"""MG2 image-conditioning VAE parameters retain their requested precision."""

from __future__ import annotations

import sys

import pytest
import torch

from tests.synthesis.test_matrix_game_2_optimizations import _install_tiny_loader
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants import action_21 as wan_vae
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime import worldfoundry_runtime as runtime


@pytest.mark.parametrize(
    "requested_dtype",
    [None, torch.bfloat16, torch.float16, torch.float32],
    ids=["default-bf16", "bf16", "fp16", "fp32"],
)
def test_image_conditioning_vae_casts_original_parameters_directly(monkeypatch, requested_dtype):
    original_weight = torch.tensor([1.004, -1.004, 1.0038, 0.9999], dtype=torch.float32)

    class Encoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(original_weight.clone())

    monkeypatch.setattr(wan_vae, "VideoVAE_", Encoder)
    original = wan_vae.WanVAE()
    original_mean = original.mean.clone()
    original_std = original.std.clone()
    _install_tiny_loader(monkeypatch)
    module_name = (
        "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.extension_modules.wanx_vae.wanx_vae"
    )
    initial_dtypes = []

    def load_image_conditioner(model_root, dtype):
        assert model_root == "/unused"
        initial_dtypes.append(dtype)
        return wan_vae.WanVAE().to(dtype)

    monkeypatch.setattr(sys.modules[module_name], "get_wanx_vae_wrapper", load_image_conditioner)
    options = {} if requested_dtype is None else {"weight_dtype": requested_dtype}
    loaded = runtime.MatrixGame2Runtime.from_pretrained("unused", device="cpu", **options)
    expected_dtype = requested_dtype or torch.bfloat16

    assert loaded.weight_dtype is expected_dtype
    assert loaded.vae.model.weight.dtype is expected_dtype
    torch.testing.assert_close(loaded.vae.model.weight, original_weight.to(expected_dtype), atol=0, rtol=0)
    torch.testing.assert_close(loaded.vae.mean, original_mean.to(expected_dtype), atol=0, rtol=0)
    torch.testing.assert_close(loaded.vae.std, original_std.to(expected_dtype), atol=0, rtol=0)
    torch.testing.assert_close(loaded.vae.scale[0], original_mean.to(expected_dtype), atol=0, rtol=0)
    torch.testing.assert_close(loaded.vae.scale[1], 1.0 / original_std.to(expected_dtype), atol=0, rtol=0)
    assert initial_dtypes == [expected_dtype]
    assert not loaded.vae.model.weight.requires_grad
    assert not loaded.vae.training
