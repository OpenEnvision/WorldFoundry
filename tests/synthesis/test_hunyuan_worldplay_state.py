# SPDX-FileCopyrightText: WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical AR operations with analytic forwards and real small Hunyuan VAE decode."""

from __future__ import annotations

from asyncio import CancelledError
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.h15 import AutoencoderKLConv3D
from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.operators.hunyuan_worldplay_operator import HunyuanWorldPlayOperator
from worldfoundry.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.realtime import (
    HunyuanWorldPlayRealtimeSession,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.runtime import (
    HunyuanWorldPlayRuntime,
    _HunyuanWorldPlayInternalPipeline,
)
from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.utils.retrieval_context import (
    generate_points_in_sphere,
)

pytestmark = pytest.mark.unit


class AnalyticVAE(torch.nn.Module):
    """Substitute codec; real small codec execution is checked separately."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.config = SimpleNamespace(
            block_out_channels=(4, 4, 4),
            ffactor_spatial=16,
            ffactor_temporal=4,
            scaling_factor=0.5,
            shift_factor=0.25,
        )

    def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
        assert return_dict is False
        return (latents.repeat_interleave(4, dim=2)[:, :, 3:].repeat(1, 3, 1, 1, 1),)


class AnalyticTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.125), requires_grad=False)
        self.config = SimpleNamespace(in_channels=1, ideal_resolution="480p", ideal_task="i2v")
        self.double_blocks = [None]

    def forward(self, *, kv_cache: list[dict[str, Any]], ar_txt_inference: bool, **kwargs: Any) -> Any:
        if ar_txt_inference:
            text = kwargs["text_states"].mean()
            kv_cache[0]["k_txt"] = text.clone()
            kv_cache[0]["v_txt"] = text.clone()
            return kv_cache
        if kwargs["cache_vision"]:
            kv_cache[0]["k_vision"] = kwargs["hidden_states"].mean().clone()
            kv_cache[0]["v_vision"] = kwargs["hidden_states"].mean().clone()
            return kv_cache
        hidden = kwargs["hidden_states"]
        prediction = hidden[:, :1] * self.weight + hidden[:, 1:2] * 0.0625 + kv_cache[0]["k_txt"] * 0.03125
        return (prediction,)


class AnalyticTextEncoder(torch.nn.Module):
    max_length = 4


class AnalyticCore(_HunyuanWorldPlayInternalPipeline):
    """Real core cache/AR/scheduler methods; heavyweight conditions are substitutes."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__(
            vae=cast(Any, AnalyticVAE()),
            text_encoder=cast(Any, AnalyticTextEncoder()),
            transformer=cast(Any, AnalyticTransformer()),
            scheduler=self._create_scheduler(5.0),
            execution_device="cpu",
            glyph_byT5_v2=True,
            enable_offloading=False,
            byt5_model=AnalyticTextEncoder(),
            vision_encoder=AnalyticTextEncoder(),
        )
        self.target_dtype = dtype
        self.autocast_enabled = False
        self.vae_autocast_enabled = False

    def get_closest_resolution_given_reference_image(
        self, reference_image: Image.Image, target_resolution: str
    ) -> tuple[int, int]:
        return 16, 16

    def encode_prompt(self, prompt: str, device: Any, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        code = float(len(prompt)) / 16
        return (torch.full((1, 1, 1), code), torch.full((1, 1, 1), -code), torch.ones(1, 1), torch.ones(1, 1))

    def _prepare_byt5_embeddings(self, prompts: str, device: Any) -> dict[str, torch.Tensor]:
        return {"byt5_text_states": torch.zeros(2, 1, 1), "byt5_text_mask": torch.ones(2, 1)}

    def get_image_condition_latents(
        self, task_type: str, reference_image: Image.Image, height: int, width: int
    ) -> torch.Tensor:
        pixel = reference_image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return torch.tensor(float(pixel[0]) / 255).reshape(1, 1, 1, 1, 1)

    def _prepare_vision_states(
        self, reference_image: Any, target_resolution: str, latents: torch.Tensor, device: Any
    ) -> torch.Tensor:
        return torch.zeros(2 if self.do_classifier_free_guidance else 1, 1, 1)


def pipeline(runtime: HunyuanWorldPlayRuntime) -> HunyuanWorldPlayPipeline:
    return HunyuanWorldPlayPipeline(
        synthesis_model=cast(Any, SimpleNamespace(runtime=runtime)),
        operators=HunyuanWorldPlayOperator(),
        device="cpu",
    )


def get_session(pipe: HunyuanWorldPlayPipeline) -> HunyuanWorldPlayRealtimeSession:
    session = pipe._ensure_realtime_session()
    assert isinstance(session, HunyuanWorldPlayRealtimeSession)
    return session


def configure(pipe: HunyuanWorldPlayPipeline, *, b: bool = False, few_step: bool = True) -> Any:
    return pipe.configure_realtime(
        Image.new("RGB", (16, 16), "blue" if b else "red"),
        prompt="rainy night" if b else "garden",
        seed=19 if b else 7,
        num_inference_steps=5 if b else 3,
        few_step=few_step,
        guidance_scale=2.0,
        flow_shift=7.0 if b else 5.0,
    )


def chunk(pipe: HunyuanWorldPlayPipeline, direction: str = "forward") -> dict[str, Any]:
    result = pipe.stream_realtime(interactions=[direction])
    assert result is not None
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("few_step", [False, True])
def test_shared_runtime_interleaves_rollouts_without_calendar_or_history_leaks(
    dtype: torch.dtype, few_step: bool
) -> None:
    runtime = HunyuanWorldPlayRuntime(model=AnalyticCore(dtype))
    owner = runtime.model
    scheduler = owner.scheduler
    original_grid = scheduler.timesteps
    original_sigmas = scheduler.sigmas
    weights = tuple(owner.transformer.parameters())
    rng = torch.random.get_rng_state().clone()
    a, b = pipeline(runtime), pipeline(runtime)
    configure(a, few_step=few_step)
    a_first = chunk(a)
    a_second = chunk(a, "camera_r")
    a_session = get_session(a)
    assert a_session._latents is not None
    reference = a_session._latents.clone()
    reference_camera = list(a_session._motions)
    a_text = a_session.core._kv_cache[0]["k_txt"].clone()
    configure(a, few_step=few_step)
    np.testing.assert_array_equal(chunk(a)["video"], a_first["video"])
    configure(b, b=True, few_step=few_step)
    b_first = chunk(b, "backward")
    np.testing.assert_array_equal(chunk(a, "camera_r")["video"], a_second["video"])
    torch.testing.assert_close(a_session._latents, reference, rtol=0, atol=0)
    torch.testing.assert_close(a_session.core._kv_cache[0]["k_txt"], a_text, rtol=0, atol=0)
    assert a_session._motions == reference_camera
    b_session = get_session(b)
    assert a_session.core is not b_session.core and a_session.core is not owner
    assert a_session.core.scheduler is not b_session.core.scheduler
    assert a_session.core._kv_cache is not b_session.core._kv_cache
    assert a_session._generator is not b_session._generator
    assert a_session.core.points_local is not b_session.core.points_local
    assert a_session.core.transformer is b_session.core.transformer is owner.transformer
    assert a_session.core.vae is b_session.core.vae is owner.vae
    assert a_session._execution_lock is b_session._execution_lock is runtime._realtime_lock
    assert scheduler is owner.scheduler and scheduler.timesteps is original_grid and scheduler.sigmas is original_sigmas
    assert scheduler.step_index is None
    assert not hasattr(owner, "_kv_cache") and not hasattr(owner, "points_local")
    assert all(left is right for left, right in zip(weights, owner.transformer.parameters()))
    assert a_first["video"].shape[0] == 13 and a_second["video"].shape[0] == 16
    for result, frames in ((a_first, 13), (a_second, 16), (b_first, 13)):
        declaration = result["video_chunk"]
        assert declaration.frames is result["video"]
        assert (declaration.layout, declaration.value_range, declaration.frame_count) == ("thwc", "0,255", frames)
        np.testing.assert_array_equal(video_chunk_to_uint8_frames(declaration), result["video"])
    b_history = b_session._latents
    a.reset_realtime()
    assert not a_session.configured and a_session._latents is a_session.core._kv_cache is None
    assert b_session._latents is b_history and b_session.configured
    configure(a, few_step=few_step)
    np.testing.assert_array_equal(chunk(a)["video"], a_first["video"])
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("few_step", [False, True])
def test_native_ar_euler_matches_independent_analytic_updates(dtype: torch.dtype, few_step: bool) -> None:
    pipe = pipeline(HunyuanWorldPlayRuntime(model=AnalyticCore(dtype)))
    configure(pipe, few_step=few_step)
    session = get_session(pipe)
    condition = torch.zeros(1, 1, 4, 1, 1)
    condition[:, :, :1] = 1
    expected = torch.randn((1, 1, 4, 1, 1), generator=torch.Generator().manual_seed(7), dtype=dtype)
    path = torch.linspace(1, 0, 4)
    sigmas = 5 * path / (1 + 4 * path)
    for index in range(3):
        positive = expected.float() * 0.125 + condition * 0.0625 + (6 / 16) * 0.03125
        if few_step:
            prediction = positive
        else:
            negative = expected.float() * 0.125 + condition * 0.0625 - (6 / 16) * 0.03125
            prediction = negative + 2 * (positive - negative)
        expected = (expected.float() + (sigmas[index + 1] - sigmas[index]) * prediction).to(dtype)
    chunk(pipe)
    torch.testing.assert_close(session._latents, expected, rtol=0, atol=0)


def test_shared_module_forwards_hold_the_runtime_owner_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HunyuanWorldPlayRuntime(model=AnalyticCore())
    original = AnalyticTransformer.forward
    calls: list[bool] = []

    def forward(self: AnalyticTransformer, **kwargs: Any) -> Any:
        acquired = runtime._realtime_lock.acquire(blocking=False)
        if acquired:
            runtime._realtime_lock.release()
        assert not acquired
        calls.append(kwargs["ar_txt_inference"])
        return original(self, **kwargs)

    monkeypatch.setattr(AnalyticTransformer, "forward", forward)
    pipe = pipeline(runtime)
    configure(pipe)
    chunk(pipe)
    assert any(calls) and not all(calls)
    assert runtime._realtime_lock.acquire(blocking=False)
    runtime._realtime_lock.release()


@pytest.mark.parametrize("stage", ["prompt", "image", "vision", "text-cache", "denoise", "decode"])
@pytest.mark.parametrize("error_type", [RuntimeError, CancelledError])
def test_failure_or_cancellation_releases_only_its_session_and_allows_reconfigure(
    monkeypatch: pytest.MonkeyPatch, stage: str, error_type: type[BaseException]
) -> None:
    runtime = HunyuanWorldPlayRuntime(model=AnalyticCore())
    a, b = pipeline(runtime), pipeline(runtime)
    configure(a)
    expected = chunk(a)["video"].copy()
    configure(b, b=True)
    chunk(b)
    b_session = get_session(b)
    assert b_session._latents is not None
    b_history = b_session._latents.clone()
    b_cache = b_session.core._kv_cache
    mapping = {
        "prompt": "encode_prompt",
        "image": "get_image_condition_latents",
        "vision": "_prepare_vision_states",
        "text-cache": "prepare_ar_text_cache",
        "denoise": "denoise_ar_chunk",
    }

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error_type("interrupted")

    with monkeypatch.context() as patch:
        if stage == "decode":
            patch.setattr(AnalyticVAE, "decode", fail)
        else:
            patch.setattr(AnalyticCore, mapping[stage], fail)
        with pytest.raises(error_type, match="interrupted"):
            if stage in {"denoise", "decode"}:
                chunk(a)
            else:
                configure(a)
    session = get_session(a)
    assert not session.configured
    assert session._latents is session._cond_latents is session._generator is None
    assert session.core._kv_cache is session.core._kv_cache_neg is None
    assert session.core.points_local is None and session._motions == []
    assert b_session.core._kv_cache is b_cache and b_session.configured
    torch.testing.assert_close(b_session._latents, b_history, rtol=0, atol=0)
    assert runtime._realtime_lock.acquire(blocking=False)
    runtime._realtime_lock.release()
    configure(a)
    np.testing.assert_array_equal(chunk(a)["video"], expected)
    assert chunk(b)["video"].shape[0] == 16


def test_non_output_rank_advances_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = pipeline(HunyuanWorldPlayRuntime(model=AnalyticCore()))
    session = get_session(pipe)
    session.rank = 1
    monkeypatch.setattr(session, "_decode", lambda: pytest.fail("non-output rank decoded pixels"))
    assert configure(pipe) is None
    for count in (4, 8):
        assert pipe.stream_realtime(interactions=["forward"]) is None
        assert session.generated_latent_frames == count
        assert len(session._motions) == count - 1


@pytest.mark.parametrize("steps,height,width", [(0, None, None), (1, 0, 16), (1, 17, 16), (1, 16, 15)])
def test_invalid_configuration_rejects_before_prompt_encoding(
    monkeypatch: pytest.MonkeyPatch, steps: int, height: int | None, width: int | None
) -> None:
    runtime = HunyuanWorldPlayRuntime(model=AnalyticCore())
    pipe = pipeline(runtime)
    monkeypatch.setattr(AnalyticCore, "encode_prompt", lambda *a, **k: pytest.fail("invalid config encoded prompt"))
    with pytest.raises(ValueError, match="positive"):
        pipe.configure_realtime(
            Image.new("RGB", (16, 16)), num_inference_steps=steps, user_height=height, user_width=width
        )
    assert not get_session(pipe).configured


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_real_small_hunyuan_vae_prefix_suffix_pixels_match_canonical_decode(dtype: torch.dtype) -> None:
    rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        vae = (
            AutoencoderKLConv3D(
                in_channels=3,
                out_channels=3,
                latent_channels=1,
                block_out_channels=(8, 8, 8),
                layers_per_block=0,
                ffactor_spatial=4,
                ffactor_temporal=4,
                sample_size=8,
                sample_tsize=13,
                scaling_factor=0.5,
                shift_factor=0.25,
            )
            .eval()
            .requires_grad_(False)
            .to(dtype)
        )
    runtime = HunyuanWorldPlayRuntime(model=SimpleNamespace(execution_device="cpu", target_dtype=dtype))
    session = HunyuanWorldPlayRealtimeSession(runtime, HunyuanWorldPlayOperator())
    session.core.vae = vae
    session.core.vae_dtype = dtype
    session.core.vae_autocast_enabled = False
    session.core.enable_offloading = False
    weights = tuple(vae.parameters())
    latents = torch.linspace(-0.5, 0.5, 8 * 4, dtype=dtype).reshape(1, 1, 8, 2, 2)
    original = latents.clone()
    emitted = 0
    for count, new_frames in ((4, 13), (8, 16)):
        session._latents = latents[:, :, :count]
        expected = vae.decode(session._latents / 0.5 + 0.25, return_dict=False)[0][:, :, emitted:]
        expected_pixels = (expected / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 4, 1).numpy()[0]
        frames = session._decode()
        assert frames.shape == (new_frames, 8, 8, 3)
        np.testing.assert_array_equal(frames, expected_pixels)
        emitted += new_frames
    session.reset()
    assert session._decoded_frames == 0 and session._latents is None
    session._latents = latents[:, :, :4]
    assert session._decode().shape[0] == 13
    assert all(left is right for left, right in zip(weights, vae.parameters()))
    torch.testing.assert_close(latents, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_geometry_sampling_is_request_seeded_without_changing_legacy_default() -> None:
    state = torch.random.get_rng_state().clone()
    a = generate_points_in_sphere(32, 8.0, generator=torch.Generator().manual_seed(7))
    b = generate_points_in_sphere(32, 8.0, generator=torch.Generator().manual_seed(19))
    again = generate_points_in_sphere(32, 8.0, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, again, rtol=0, atol=0)
    assert not torch.equal(a, b) and (a.norm(dim=-1) <= 8).all()
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        legacy = generate_points_in_sphere(32, 8.0)
    torch.testing.assert_close(a, legacy, rtol=0, atol=0)
