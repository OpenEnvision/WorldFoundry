# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

"""Absolute Astra camera progress and request-local diffusion/codec state."""

from __future__ import annotations

from asyncio import CancelledError
from copy import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from worldfoundry.core.nn import FlowMatchScheduler
from worldfoundry.pipelines.kling.pipeline_astra import AstraConfig
from worldfoundry.synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from worldfoundry.synthesis.visual_generation.memory.kling.astra import AstraMemory

pytestmark = pytest.mark.unit


def _latents(start: int, count: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(start, start + count, dtype=dtype).reshape(1, count, 1, 1).repeat(3, 1, 2, 2)


@pytest.mark.parametrize("capacity", [49, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("modality,columns", [("sekai", 13), ("nuscenes", 8), ("openx", 13)])
def test_camera_progress_and_context_follow_unbounded_history(
    capacity: int, dtype: torch.dtype, modality: str, columns: int
) -> None:
    memory = AstraMemory(capacity=capacity)
    camera = torch.arange(151, dtype=dtype).unsqueeze(1).repeat(1, columns)
    original = camera.clone()
    memory.record(_latents(0, 1, dtype))
    for chunk in range(16):
        total = 1 + 8 * chunk
        pack = memory.select(8, camera, 0, modality)
        assert memory.total_latent_frames == pack["current_length"] == total
        assert pack["next_length"] == total + 8
        assert pack["modality_type"] == modality
        assert memory.history_latents is not None
        assert memory.history_latents.shape[1] == min(total, capacity)
        assert memory.history_latents[0, 0, 0, 0].item() == 0
        context = torch.cat([pack["clean_latents_4x"], pack["clean_latents_2x"], pack["clean_latents"][:, -1:]], dim=1)
        expected_context = torch.zeros(3, 19, 2, 2, dtype=dtype)
        available = min(total, 19)
        expected_context[:, -available:] = _latents(total - available, available, dtype)
        torch.testing.assert_close(context, expected_context, rtol=0, atol=0)
        torch.testing.assert_close(
            pack["camera_embedding"][19:27, 0], torch.arange(total, total + 8, dtype=dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(
            pack["camera_embedding"][19 - available : 19, 0],
            torch.arange(total - available, total, dtype=dtype),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            pack["camera_embedding"][19 - available : 19, -1], torch.ones(available, dtype=dtype), rtol=0, atol=0
        )
        assert torch.count_nonzero(pack["camera_embedding"][19:, -1]) == 0
        assert torch.count_nonzero(pack["camera_embedding"][-1]) == 0
        assert pack["latent_indices"].tolist() == list(range(20, 28))
        assert pack["clean_latent_indices"].tolist() == [0, 19]
        assert pack["clean_latent_2x_indices"].tolist() == [17, 18]
        assert pack["clean_latent_4x_indices"].tolist() == list(range(1, 17))
        memory.record(_latents(total, 8, dtype))
    torch.testing.assert_close(camera, original, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [1, 8, 19, 20])
def test_short_windows_keep_pinned_camera_row_aligned_with_retained_latents(capacity: int) -> None:
    memory = AstraMemory(capacity=capacity)
    memory.record(_latents(0, 25))
    camera = torch.arange(40, dtype=torch.float32).unsqueeze(1).repeat(1, 13)
    pack = memory.select(8, camera, 0, "sekai")
    available = min(capacity, 19)
    context = torch.cat([pack["clean_latents_4x"], pack["clean_latents_2x"], pack["clean_latents"][:, -1:]], dim=1)
    torch.testing.assert_close(
        pack["camera_embedding"][19 - available : 19, 0], context[0, -available:, 0, 0], rtol=0, atol=0
    )
    assert pack["camera_embedding"][19, 0].item() == 25
    assert pack["clean_latents"][0, 0, 0, 0].item() == 0


def test_padding_and_reset_preserve_absolute_progress_and_allow_reuse() -> None:
    memory = AstraMemory(capacity=100)
    memory.record(_latents(0, 105))
    camera = torch.arange(108, dtype=torch.float32).unsqueeze(1).repeat(1, 13)
    pack = memory.select(8, camera, 0, "sekai")
    assert pack["camera_embedding"][19:27, 0].tolist() == [105, 106, 107, 0, 0, 0, 0, 0]
    memory.append_record("session", kind="text")
    memory.manage(action="reset")
    assert memory.total_latent_frames == 0 and memory.history_latents is None and memory.storage == []
    with pytest.raises(ValueError, match="requires initial latents"):
        memory.select(8, camera, 0, "sekai")
    memory.record(_latents(0, 1))
    assert memory.select(8, camera, 0, "sekai")["camera_embedding"][19, 0].item() == 1


def test_standalone_framepack_retains_positional_arguments_and_local_indices() -> None:
    memory = AstraMemory()
    camera = torch.arange(40, dtype=torch.float32).unsqueeze(1).repeat(1, 13)
    pack = memory.prepare_framepack_sliding_window_with_camera_moe(_latents(0, 9), 8, camera, 0, "sekai", 49)
    assert pack["current_length"] == 9 and pack["next_length"] == 17
    assert pack["camera_embedding"][19, 0].item() == 9
    assert memory.history_latents is None and memory.total_latent_frames == 0


class _Denoiser(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0625, dtype=dtype), requires_grad=False)

    def forward(self, sample: torch.Tensor, *, context: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None]:
        return sample * 0.125 + self.weight + context, None


def _synthesis(dtype: torch.dtype) -> AstraSynthesis:
    return AstraSynthesis(
        SimpleNamespace(
            device="cpu",
            dit=_Denoiser(dtype),
            scheduler=FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True),
            prepare_extra_input=lambda latents: {},
        )
    )


def _predict(synthesis: AstraSynthesis, *, steps: int, seed: int, value: float, cfg: str = "none") -> torch.Tensor:
    memory = AstraMemory()
    memory.record(torch.zeros(3, 1, 2, 2))
    pack = memory.select(8, torch.zeros(40, 13), 0, "sekai")
    config = AstraConfig(
        device="cpu",
        num_inference_steps=steps,
        use_camera_cfg=cfg == "both",
        text_guidance_scale=2 if cfg != "none" else 1,
    )
    return synthesis.predict(
        pack,
        8,
        {"context": torch.tensor(value)},
        {"context": torch.tensor(-value)} if cfg != "none" else None,
        config,
        torch.zeros(40, 13),
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cfg", ["none", "text", "both"])
def test_scheduler_is_request_local_and_keeps_exact_euler_arithmetic(dtype: torch.dtype, cfg: str) -> None:
    synthesis = _synthesis(dtype)
    scheduler = synthesis.pipe.scheduler
    timesteps, sigmas = scheduler.timesteps, scheduler.sigmas
    weight = synthesis.pipe.dit.weight
    original_weight = weight.clone()
    rng = torch.random.get_rng_state().clone()
    first = _predict(synthesis, steps=3, seed=7, value=0.25, cfg=cfg)
    _predict(synthesis, steps=7, seed=11, value=0.5, cfg=cfg)
    repeated = _predict(synthesis, steps=3, seed=7, value=0.25, cfg=cfg)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    reference_scheduler = copy(scheduler)
    reference_scheduler.set_timesteps(3)
    expected = torch.randn(1, 3, 8, 2, 2, dtype=dtype, generator=torch.Generator().manual_seed(7))
    for timestep in reference_scheduler.timesteps:
        positive = expected * 0.125 + weight + torch.tensor(0.25)
        if cfg == "none":
            noise = positive
        else:
            negative = expected * 0.125 + weight + torch.tensor(-0.25)
            noise = negative + 2 * (positive - negative)
            if cfg == "both":
                noise = negative + 2 * (noise - negative)
        expected = reference_scheduler.step(noise, timestep, expected)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert scheduler.timesteps is timesteps and scheduler.sigmas is sigmas
    assert synthesis.pipe.dit.weight is weight
    torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_interleaved_predictions_share_weights_without_replacing_active_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthesis = _synthesis(torch.float32)
    expected = _predict(synthesis, steps=3, seed=7, value=0.25)
    forward = synthesis.pipe.dit.forward
    interleaved: list[torch.Tensor] = []

    def run(sample: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None]:
        if not interleaved:
            interleaved.append(torch.empty(0))
            interleaved[0] = _predict(synthesis, steps=7, seed=11, value=0.5)
        return forward(sample, **kwargs)

    monkeypatch.setattr(synthesis.pipe.dit, "forward", run)
    actual = _predict(synthesis, steps=3, seed=7, value=0.25)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert interleaved[0].shape == actual.shape


@pytest.mark.parametrize("operation", ["encode", "decode"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_shared_vae_scratch_is_locked_cleared_and_reusable(operation: str, outcome: str) -> None:
    scratch: list[torch.Tensor] = []
    cleanups: list[int] = []

    def clear() -> None:
        scratch.clear()
        cleanups.append(1)

    pipe = SimpleNamespace(device="cpu", vae=SimpleNamespace(model=SimpleNamespace(clear_cache=clear)))
    synthesis = AstraSynthesis(pipe)
    error = CancelledError if outcome == "cancelled" else RuntimeError

    def codec(video: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert synthesis._vae_lock.locked() and not torch.is_grad_enabled()
        assert not scratch
        scratch.append(video)
        if outcome != "success":
            raise error("codec stopped")
        return video

    pipe.encode_video = pipe.decode_video = codec
    video = torch.full((3, 1, 2, 2), 0.5)
    original = video.clone()
    invoke = (
        (lambda: synthesis.encode_frames(video))
        if operation == "encode"
        else (lambda: synthesis.decode_video(video.unsqueeze(0)))
    )
    if outcome == "success":
        actual = invoke()
        if operation == "encode":
            torch.testing.assert_close(actual, video.to(torch.bfloat16), rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(actual, np.full((1, 2, 2, 3), 191, np.uint8))
    else:
        with pytest.raises(error, match="codec stopped"):
            invoke()
    assert scratch == [] and len(cleanups) == 1 and not synthesis._vae_lock.locked()
    outcome = "success"
    invoke()
    assert scratch == [] and len(cleanups) == 2 and not synthesis._vae_lock.locked()
    torch.testing.assert_close(video, original, rtol=0, atol=0)


def test_real_small_wan_vae_reuses_weights_without_retaining_temporal_scratch() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        vae = WanVideoVAE(base_dim=4).to(torch.bfloat16).eval()
    pipe = SimpleNamespace(
        device="cpu",
        vae=vae,
        encode_video=lambda frames, **kwargs: vae.encode(frames, device="cpu", **kwargs),
        decode_video=lambda latents, **kwargs: vae.decode(latents, device="cpu", **kwargs),
    )
    synthesis = AstraSynthesis(pipe)
    frames = torch.linspace(-0.5, 0.5, 3 * 5 * 16 * 16).reshape(3, 5, 16, 16)
    original = frames.clone()
    weights = tuple(vae.parameters())
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        expected_latents = vae.encode(frames.unsqueeze(0).to(torch.bfloat16), device="cpu", **synthesis.tiler_kwargs)
        decoded = vae.decode(expected_latents, device="cpu", **synthesis.tiler_kwargs)
    expected_pixels = ((decoded[0].float().permute(1, 2, 3, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    first = synthesis.encode_frames(frames)
    torch.testing.assert_close(first, expected_latents[0], rtol=0, atol=0)
    assert all(value is None for value in vae.model._enc_feat_map)
    pixels = synthesis.decode_video(first.unsqueeze(0))
    np.testing.assert_array_equal(pixels, expected_pixels)
    assert all(value is None for value in vae.model._feat_map)
    changed = synthesis.encode_frames(-frames)
    assert not torch.equal(changed, first)
    synthesis.decode_video(changed.unsqueeze(0))
    repeated = synthesis.encode_frames(frames)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    np.testing.assert_array_equal(synthesis.decode_video(repeated.unsqueeze(0)), pixels)
    assert all(value is None for value in vae.model._enc_feat_map + vae.model._feat_map)
    assert all(before is after for before, after in zip(weights, vae.parameters()))
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
