# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""MAGI-2 refiner packing, numerical sampling, and public loading contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.encoders.magi2_refiner import Magi2RefinerDataProxy
from worldfoundry.base_models.diffusion_model.models.networks.magi2.config import Magi2RefinerConfig
from worldfoundry.base_models.diffusion_model.models.networks.magi2.preview_layers import ModalityDispatcher
from worldfoundry.base_models.diffusion_model.models.networks.magi2.refiner_dit import (
    MLP,
    MLPActivationType,
    MLPConfig,
    Transformer,
    build_range_mask,
)
from worldfoundry.base_models.diffusion_model.runners.magi2_refinement import refine_video_latent, refiner_signal_scale
from worldfoundry.pipelines.magi2 import NativeMagi2Pipeline

pytestmark = pytest.mark.unit


def test_proxy_restores_pixels_and_grid_window_union() -> None:
    latent = torch.arange(48 * 9 * 5 * 13).reshape(1, 48, 9, 5, 13).float()
    proxy = Magi2RefinerDataProxy((9, 5, 13), device=torch.device("cpu"))
    packed = proxy.build_model_inputs(latent, torch.ones(1, 2, 64))
    x = packed["x"]
    coords = packed["coords_mapping"]
    handler = packed["local_attn_handler"]
    from worldfoundry.base_models.diffusion_model.models.networks.magi2.refiner_dit import LocalAttnHandler

    assert isinstance(x, torch.Tensor) and isinstance(coords, torch.Tensor)
    assert isinstance(handler, LocalAttnHandler)
    restored = proxy.unpack_output(x)
    torch.testing.assert_close(restored, latent, rtol=0, atol=0)
    mask = build_range_mask(handler.q_ranges, handler.k_ranges, len(x), torch.device("cpu"))
    coords = coords[: proxy.video_tokens, :3].long()
    blocks = coords // torch.tensor([8, 4, 4])
    expected = (blocks[:, None] - blocks[None, :]).abs().le(2).all(-1)
    assert torch.equal(mask[: proxy.video_tokens, : proxy.video_tokens], expected)
    assert mask[:, proxy.video_tokens :].all()
    assert mask[proxy.video_tokens :].all()


def test_renoise_scale_matches_zero_snr_reference() -> None:
    betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=np.float64) ** 2
    signal = torch.from_numpy(np.cumprod(1 - betas).astype(np.float32)).sqrt()
    first, last = signal[0].clone(), signal[-1].clone()
    signal -= last
    signal *= first / (first - last)
    torch.testing.assert_close(refiner_signal_scale(device=torch.device("cpu")), signal.flip(0)[220], rtol=0, atol=0)


def test_real_tiny_refiner_is_finite_repeatable_and_context_sensitive() -> None:
    model = _tiny_refiner()
    latent = torch.randn(1, 48, 2, 2, 2)
    context = torch.ones(1, 2, 64)
    state = torch.random.get_rng_state().clone()

    def run(text: torch.Tensor) -> torch.Tensor:
        return refine_video_latent(
            model,
            latent,
            text,
            torch.zeros_like(text),
            height=32,
            width=32,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(42),
        )

    first, changed, repeated = run(context), run(context * 2), run(context)
    assert first.shape == (1, 48, 3, 2, 2)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, changed)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), state)


def _tiny_refiner() -> Transformer:
    torch.manual_seed(1)
    model = Transformer(
        Magi2RefinerConfig(
            num_layers=2,
            hidden_size=64,
            head_dim=16,
            num_query_groups=2,
            text_in_channels=64,
            mm_layers=(0, 1),
            local_attn_layers=(0, 1),
        )
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.zero_()
            else:
                parameter.normal_(0, 0.01)
    return model


@pytest.mark.parametrize("modalities", [1, 3])
def test_chunked_dense_mlp_matches_whole_token_math(modalities: int) -> None:
    model = MLP(
        MLPConfig(
            hidden_size=16,
            intermediate_size=32,
            activation_type=MLPActivationType.SWIGLU7,
            params_dtype=torch.bfloat16,
            num_modality=modalities,
            num_layers=1,
            gated_act=True,
        )
    )
    with torch.no_grad():
        model.pre_norm.weight.zero_()
        model.up_gate_proj.weight.normal_(0, 0.02)
        model.down_proj.weight.normal_(0, 0.02)
        inputs = torch.randn(5100, 16)
        tags = torch.zeros(5100, dtype=torch.long)
        if modalities == 3:
            tags[4100:] = 2
        dispatcher = ModalityDispatcher(tags, modalities)
        expected = model.pre_norm(inputs, modality_dispatcher=dispatcher).bfloat16()
        expected = model.up_gate_proj(expected, modality_dispatcher=dispatcher).float()
        expected = model.activation_func(expected).bfloat16()
        expected = model.down_proj(expected, modality_dispatcher=dispatcher).float()
        torch.testing.assert_close(model(inputs, dispatcher), expected, rtol=1e-5, atol=1e-6)


def test_pipeline_refinement_preserves_audio_and_changes_export_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = NativeMagi2Pipeline(device="cpu", preview_transformer=object(), refiner_transformer=_tiny_refiner())
    plan = {
        **pipe._plan(short_edge=512, aspect_ratio="16:9", duration_seconds=10.0),
        "video_latent_t": 2,
        "latent_h": 2,
        "latent_w": 2,
        "frames": 5,
    }
    monkeypatch.setattr(pipe, "_plan", lambda **kwargs: plan)
    preview = torch.randn(1, 48, 2, 2, 2)
    audio = torch.randn(1, 4, 64)
    monkeypatch.setattr(
        "worldfoundry.base_models.diffusion_model.models.networks.magi2.sampler.Magi2PreviewSampler.sample",
        lambda self, **kwargs: (preview, audio),
    )
    monkeypatch.setattr(
        pipe, "_encode_prompt", lambda prompt: torch.ones(1, 2, 64) if prompt == "robot" else torch.zeros(1, 2, 64)
    )
    result = pipe.generate(
        prompt="robot",
        negative_prompt="negative robot",
        use_refiner=True,
        refiner_num_inference_steps=2,
        refiner_height=32,
        refiner_width=32,
        return_latents=True,
    )
    assert result["video_latent"].shape == (1, 48, 3, 2, 2)
    assert result["audio_latent"] is audio
    assert result["fps"] == 25.0


def test_public_loader_loads_refiner_weights_from_declared_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "worldfoundry.base_models.diffusion_model.models.networks.magi2.loading.load_magi2_dit_weights",
        lambda model, root, strict: calls.append((type(model), root, strict)),
    )
    pipe = NativeMagi2Pipeline.from_pretrained(str(tmp_path), device="cpu")
    pipe._ensure_components(load_preview=False, load_decoders=False, load_refiner=True)
    assert calls[0][1:] == (tmp_path / "refiner", True)
    assert isinstance(pipe.refiner_transformer, Transformer)


def test_invalid_refiner_request_rejected_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = NativeMagi2Pipeline(device="cpu")
    monkeypatch.setattr(pipe, "_ensure_components", lambda **kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="refiner_height"):
        pipe.generate(prompt="robot", use_refiner=True, refiner_height=1080)
