"""Wan inference dtype/autocast hot-path contracts."""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.base_models.diffusion_model.models.denoisers.wan import WanDenoiser
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WAN_DENOISE_MASK_IS_ALL_ONES,
)


class _RecordingWan(torch.nn.Module):
    per_token_timestep = False
    inject_sample_info = False
    has_image_input = False
    patch_size = (1, 1, 1)

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.observed: dict[str, torch.dtype] = {}
        self.observed_kwargs: dict[str, object] = {}
        self.observed_timestep: torch.Tensor | None = None

    def forward(self, *, x, timestep, context, **kwargs):
        self.observed = {"x": x.dtype, "context": context.dtype}
        self.observed_timestep = timestep.detach().clone()
        self.observed_kwargs = dict(kwargs)
        return x


def _input(*, conditioning: dict[str, torch.Tensor] | None = None) -> DenoiserInput:
    return DenoiserInput(
        latents=torch.randn(1, 4, 2, 3, 3, dtype=torch.float32),
        timestep=torch.tensor([500.0]),
        next_timestep=torch.tensor([250.0]),
        conditioning=conditioning
        or {"context": torch.randn(1, 4, 8, dtype=torch.float32)},
        step_index=0,
        total_steps=2,
        request_id="dtype-test",
    )


def test_resident_bf16_dit_casts_inputs_once_and_elides_autocast(monkeypatch) -> None:
    model = _RecordingWan(torch.bfloat16)
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)

    def unexpected_autocast(*_args, **_kwargs):
        raise AssertionError("resident BF16 execution must not enter autocast")

    monkeypatch.setattr(torch, "autocast", unexpected_autocast)
    output = denoiser(_input())

    assert model.observed == {
        "x": torch.bfloat16,
        "context": torch.bfloat16,
    }
    assert output.sample.dtype == torch.bfloat16
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["effective"]["denoiser_autocast_context"] == (
        "elided-resident-dtype"
    )


def test_resident_dit_casts_channel_condition_before_concatenation(
    monkeypatch,
) -> None:
    model = _RecordingWan(torch.bfloat16)
    denoiser = WanDenoiser(
        model,
        compute_dtype=torch.bfloat16,
        channel_condition_key="condition_latents",
    )
    conditioning = {
        "context": torch.randn(1, 4, 8, dtype=torch.float32),
        "condition_latents": torch.randn(1, 2, 2, 3, 3, dtype=torch.float32),
    }

    monkeypatch.setattr(
        torch,
        "autocast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resident BF16 execution must not enter autocast")
        ),
    )
    output = denoiser(_input(conditioning=conditioning))

    assert model.observed["x"] == torch.bfloat16
    assert output.sample.shape == (1, 6, 2, 3, 3)


def test_fp32_resident_dit_keeps_managed_bf16_autocast(monkeypatch) -> None:
    model = _RecordingWan(torch.float32)
    denoiser = WanDenoiser(model, compute_dtype=torch.bfloat16)
    calls: list[tuple[str, torch.dtype]] = []

    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    def recording_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return _Context()

    monkeypatch.setattr(torch, "autocast", recording_autocast)
    denoiser(_input())

    assert calls == [("cpu", torch.bfloat16)]
    assert model.observed == {"x": torch.float32, "context": torch.float32}
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["effective"]["denoiser_autocast_context"] == "enabled"


def test_inplace_residual_routes_only_during_no_grad_inference() -> None:
    model = _RecordingWan(torch.float32)
    denoiser = WanDenoiser(
        model,
        compute_dtype=torch.float32,
        inplace_residual=True,
    )

    with torch.no_grad():
        denoiser(_input())

    assert model.observed_kwargs["_worldfoundry_inplace_residual"] is True
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["effective"]["inplace_residual"] == "in-place-executed"
    assert report["runtime"]["inplace_residual"]["inplace_calls"] == 1
    assert report["runtime"]["inplace_residual"]["functional_calls"] == 0


def test_inplace_residual_falls_back_without_mutating_autograd_inputs() -> None:
    model = _RecordingWan(torch.float32)
    denoiser = WanDenoiser(
        model,
        compute_dtype=torch.float32,
        inplace_residual=True,
    )
    model_input = _input()
    model_input.latents.requires_grad_()
    original = model_input.latents.detach().clone()

    output = denoiser(model_input)
    output.sample.sum().backward()

    assert "_worldfoundry_inplace_residual" not in model.observed_kwargs
    torch.testing.assert_close(model_input.latents.detach(), original)
    assert model_input.latents.grad is not None
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["effective"]["inplace_residual"] == (
        "functional-safety-fallback"
    )
    assert report["runtime"]["inplace_residual"][
        "autograd_fallback_calls"
    ] == 1


def test_inplace_residual_falls_back_when_feature_cache_owns_block_inputs() -> None:
    model = _RecordingWan(torch.float32)
    denoiser = WanDenoiser(
        model,
        compute_dtype=torch.float32,
        inplace_residual=True,
        teacache_threshold=0.1,
    )

    with torch.no_grad():
        denoiser(_input())

    assert "feature_cache" in model.observed_kwargs
    assert "_worldfoundry_inplace_residual" not in model.observed_kwargs
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["runtime"]["inplace_residual"][
        "feature_cache_fallback_calls"
    ] == 1


def _per_token_denoiser() -> tuple[_RecordingWan, WanDenoiser]:
    model = _RecordingWan(torch.float32)
    model.per_token_timestep = True
    return model, WanDenoiser(model, compute_dtype=torch.float32)


def test_per_token_wan_uses_scalar_only_with_explicit_all_ones_proof() -> None:
    model, denoiser = _per_token_denoiser()
    conditioning = {
        "context": torch.randn(1, 4, 8),
        "denoise_mask": torch.ones(1, 1, 2, 3, 3),
        WAN_DENOISE_MASK_IS_ALL_ONES: True,
    }

    denoiser(_input(conditioning=conditioning))

    assert model.observed_timestep is not None
    assert model.observed_timestep.shape == (1,)
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["runtime"]["wan_timestep"]["global_calls"] == 1
    assert report["runtime"]["wan_timestep"]["explicit_all_ones_calls"] == 1
    assert report["runtime"]["wan_timestep"]["per_token_calls"] == 0


def test_all_one_mask_without_semantic_proof_stays_per_token() -> None:
    model, denoiser = _per_token_denoiser()
    conditioning = {
        "context": torch.randn(1, 4, 8),
        "denoise_mask": torch.ones(1, 1, 2, 3, 3),
    }

    denoiser(_input(conditioning=conditioning))

    assert model.observed_timestep is not None
    assert model.observed_timestep.shape == (1, 18)
    report = denoiser.runtime_optimization_report("dtype-test")
    assert report["runtime"]["wan_timestep"]["global_calls"] == 0
    assert report["runtime"]["wan_timestep"]["per_token_calls"] == 1


def test_image_condition_first_frame_receives_zero_timestep() -> None:
    model, denoiser = _per_token_denoiser()
    mask = torch.ones(1, 1, 2, 3, 3)
    mask[:, :, :1] = 0
    conditioning = {
        "context": torch.randn(1, 4, 8),
        "denoise_mask": mask,
        WAN_DENOISE_MASK_IS_ALL_ONES: False,
    }

    denoiser(_input(conditioning=conditioning))

    assert model.observed_timestep is not None
    assert model.observed_timestep.shape == (1, 18)
    assert torch.count_nonzero(model.observed_timestep[:, :9]) == 0
    torch.testing.assert_close(
        model.observed_timestep[:, 9:],
        torch.full((1, 9), 500.0),
    )
