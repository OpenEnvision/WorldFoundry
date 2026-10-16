# SPDX-FileCopyrightText: WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""HY-WorldPlay sessions over the loaded Hunyuan components and AR operations."""

from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from threading import RLock
from typing import Any, Generator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist

from worldfoundry.core import autocast_context
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.media.processing.postprocess import VideoChunk

from .commons import auto_offload_model
from .commons.pose_utils import pose_to_input
from .generate_custom_trajectory import generate_camera_trajectory_local
from .pipelines.pipeline_utils import retrieve_timesteps
from .utils.retrieval_context import generate_points_in_sphere

_KEY_TO_CONTROL = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "i": "camera_up",
    "k": "camera_down",
    "j": "camera_l",
    "l": "camera_r",
}
_CONTROL_ALIASES = {"back": "backward", "camera_left": "camera_l", "camera_right": "camera_r"}


class HunyuanWorldPlayRealtimeSession:
    """Session-local conditioning, calendars and history with shared weights.

    Canonical Hunyuan 1.5 decoding evaluates the causal latent prefix and emits
    only new frames. Decode work grows with the prefix; no bounded-cost cached
    VAE or latency guarantee is implied.
    """

    fps = 24
    latent_frames_per_chunk = 4
    first_output_frames = 13
    steady_output_frames = 16

    def __init__(self, runtime: Any, operator: Any) -> None:
        self.runtime = runtime
        self.core = copy(runtime.model)
        self.operator = operator
        self.device = torch.device(self.core.execution_device)
        self.dtype = self.core.target_dtype
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._state_lock = RLock()
        self._execution_lock = runtime._realtime_lock
        self._reset_unlocked()

    @contextmanager
    def _execution(self) -> Generator[None, None, None]:
        # Placement/offload acts on shared modules, not just Python cache state.
        with self._state_lock, self._execution_lock:
            try:
                yield
            finally:
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()

    def realtime_spec(self) -> RealtimeSpec:
        """Return the native four-latent playback cadence without loading weights."""
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
            controls=tuple(_KEY_TO_CONTROL.values()),
        )

    def next_output_frames(self) -> int:
        return self.first_output_frames if self.generated_latent_frames == 0 else self.steady_output_frames

    @property
    def generated_latent_frames(self) -> int:
        return 0 if self._latents is None else int(self._latents.shape[2])

    @staticmethod
    def _normalize_controls(controls: Sequence[str]) -> list[str]:
        return [_CONTROL_ALIASES.get(value, value) for item in controls if (value := str(item).strip().lower())]

    def _controls_for_latents(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
        *,
        count: int,
    ) -> list[list[str]]:
        fallback = self._normalize_controls(interactions)
        segments: list[tuple[float, list[str]]] = []
        for segment in control_segments or ():
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            controls = [
                _KEY_TO_CONTROL[key] for raw in segment.get("keys", ()) if (key := str(raw).lower()) in _KEY_TO_CONTROL
            ]
            if duration > 0.0:
                segments.append((duration, controls))
        if not segments:
            return [fallback] * count
        total = sum(duration for duration, _ in segments)
        result: list[list[str]] = []
        segment_index = 0
        segment_end = segments[0][0]
        for index in range(count):
            sample_time = (index + 0.5) * total / count
            while segment_index + 1 < len(segments) and sample_time > segment_end:
                segment_index += 1
                segment_end += segments[segment_index][0]
            result.append(segments[segment_index][1])
        return result

    def _motion(self, controls: Sequence[str]) -> dict[str, float]:
        controls = self._normalize_controls(controls)
        values = {
            "forward": (float("forward" in controls) - float("backward" in controls)) * self.operator.forward_speed,
            "right": (float("right" in controls) - float("left" in controls)) * self.operator.forward_speed,
            "yaw": (float("camera_r" in controls) - float("camera_l" in controls))
            * np.deg2rad(self.operator.yaw_speed_deg),
            "pitch": (float("camera_up" in controls) - float("camera_down" in controls))
            * np.deg2rad(self.operator.pitch_speed_deg),
        }
        return {name: float(value) for name, value in values.items() if value}

    def _camera_conditions(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = self.latent_frames_per_chunk - (self.generated_latent_frames == 0)
        controls = self._controls_for_latents(interactions, control_segments, count=count)
        self._motions.extend(self._motion(items) for items in controls)
        poses = generate_camera_trajectory_local(self._motions)
        intrinsic = [[969.6969696969696, 0.0, 960.0], [0.0, 969.6969696969696, 540.0], [0.0, 0.0, 1.0]]
        pose_data = {str(index): {"extrinsic": pose.tolist(), "K": intrinsic} for index, pose in enumerate(poses)}
        viewmats, intrinsics, actions = pose_to_input(pose_data, len(poses))
        return viewmats.unsqueeze(0), intrinsics.unsqueeze(0), actions.unsqueeze(0)

    def _condition_block(self) -> torch.Tensor:
        assert self._image_condition is not None
        condition = torch.zeros(
            (
                1,
                int(self.core.transformer.config.in_channels),
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        mask = torch.zeros(
            (1, 1, self.latent_frames_per_chunk, self._latent_height, self._latent_width), device=self.device
        )
        if self.generated_latent_frames == 0:
            condition[:, :, :1] = self._image_condition.to(device=self.device, dtype=self.dtype)
            mask[:, :, :1] = 1
        return torch.cat((condition, mask), dim=1)

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str,
        seed: int = 1,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        flow_shift: float | None = None,
        guidance_scale: float | None = None,
        few_step: bool = True,
        user_height: int | None = None,
        user_width: int | None = None,
    ) -> dict[str, Any] | None:
        """Encode invariant inputs and create session-owned text/vision KV."""
        if not isinstance(image, Image.Image):
            raise TypeError("HY-WorldPlay realtime configuration requires a PIL image.")
        if num_inference_steps < 1:
            raise ValueError("HY-WorldPlay num_inference_steps must be positive.")
        with self._execution():
            self._reset_unlocked()
            self.core = copy(self.runtime.model)
            try:
                return self._configure(
                    image.convert("RGB"),
                    prompt=prompt,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    flow_shift=flow_shift,
                    guidance_scale=guidance_scale,
                    few_step=few_step,
                    user_height=user_height,
                    user_width=user_width,
                )
            except BaseException:
                self._reset_unlocked()
                raise

    def _configure(
        self,
        image: Image.Image,
        *,
        prompt: str,
        seed: int,
        negative_prompt: str,
        num_inference_steps: int,
        flow_shift: float | None,
        guidance_scale: float | None,
        few_step: bool,
        user_height: int | None,
        user_width: int | None,
    ) -> dict[str, Any] | None:
        started = time.perf_counter()
        core = self.core
        height, width = core.get_closest_resolution_given_reference_image(image, core.ideal_resolution)
        self._height = int(height if user_height is None else user_height)
        self._width = int(width if user_width is None else user_width)
        if min(self._height, self._width) < 16 or self._height % 16 or self._width % 16:
            raise ValueError("HY-WorldPlay realtime height and width must be positive multiples of 16.")
        _, self._latent_height, self._latent_width = core.get_latent_size(
            self.first_output_frames, self._height, self._width
        )
        self._num_inference_steps = int(num_inference_steps)
        core.scheduler = core._create_scheduler(core.config.flow_shift if flow_shift is None else flow_shift)
        core._guidance_scale = (
            1.0 if few_step else (core.config.guidance_scale if guidance_scale is None else guidance_scale)
        )
        core._guidance_rescale = 0.0
        core._clip_skip = None
        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))

        prompt_started = time.perf_counter()
        with auto_offload_model(core.text_encoder, core.execution_device, enabled=core.enable_offloading):
            prompt_embeds, negative_embeds, prompt_mask, negative_mask = core.encode_prompt(
                prompt,
                self.device,
                1,
                core.do_classifier_free_guidance,
                negative_prompt,
                clip_skip=None,
                data_type="video",
            )
        extra_kwargs = {}
        if core.config.glyph_byT5_v2:
            with auto_offload_model(core.byt5_model, core.execution_device, enabled=core.enable_offloading):
                extra_kwargs = core._prepare_byt5_embeddings(prompt, self.device)
        if core.do_classifier_free_guidance:
            prompt_embeds = torch.cat((negative_embeds, prompt_embeds))
            if prompt_mask is not None:
                prompt_mask = torch.cat((negative_mask, prompt_mask))
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

        template = torch.zeros(
            (
                1,
                int(core.transformer.config.in_channels),
                self.latent_frames_per_chunk,
                self._latent_height,
                self._latent_width,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        n_tokens = self.latent_frames_per_chunk * self._latent_height * self._latent_width
        timestep_kwargs = core.prepare_extra_func_kwargs(core.scheduler.set_timesteps, {"n_tokens": n_tokens})
        self._timesteps, self._num_inference_steps = retrieve_timesteps(
            core.scheduler, self._num_inference_steps, self.device, **timestep_kwargs
        )
        core.num_warmup_steps = len(self._timesteps) - self._num_inference_steps * core.scheduler.order
        core._num_timesteps = len(self._timesteps)
        core.num_inference_steps = self._num_inference_steps
        core.chunk_latent_frames = self.latent_frames_per_chunk
        geometry_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        core.points_local = generate_points_in_sphere(50000, 8.0, generator=geometry_generator).to(self.device)
        if self.world_size > 1:
            dist.broadcast(core.points_local, src=0)

        visual_started = time.perf_counter()
        with auto_offload_model(core.vae, core.execution_device, enabled=core.enable_offloading):
            self._image_condition = core.get_image_condition_latents("i2v", image, self._height, self._width)
        with auto_offload_model(core.vision_encoder, core.execution_device, enabled=core.enable_offloading):
            vision_states = core._prepare_vision_states(np.asarray(image), core.ideal_resolution, template, self.device)
        visual_ms = (time.perf_counter() - visual_started) * 1000.0
        cache_started = time.perf_counter()
        core.prepare_ar_text_cache(
            latents=template,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            vision_states=vision_states,
            task_type="i2v",
            extra_kwargs=extra_kwargs,
            device=self.device,
        )
        cache_ms = (time.perf_counter() - cache_started) * 1000.0
        self._latents = template[:, :, :0].clone()
        self._cond_latents = torch.zeros(
            (1, template.shape[1] + 1, 0, self._latent_height, self._latent_width),
            device=self.device,
        )
        self.configured = True
        self.configure_metrics = {
            "prompt_ms": prompt_ms,
            "visual_condition_ms": visual_ms,
            "text_cache_ms": cache_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "world_size": float(self.world_size),
        }
        if self.rank != 0:
            return None
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.configure_metrics),
        }

    def _decode(self) -> np.ndarray:
        """Decode the causal prefix through the canonical VAE and emit its suffix."""
        assert self._latents is not None
        core = self.core
        latents = self._latents / core.vae.config.scaling_factor
        shift = getattr(core.vae.config, "shift_factor", None)
        if shift:
            latents = latents + shift
        with (
            autocast_context(self.device, dtype=core.vae_dtype, enabled=core.vae_autocast_enabled),
            auto_offload_model(core.vae, core.execution_device, enabled=core.enable_offloading),
        ):
            video = core.vae.decode(latents.to(dtype=core.vae_dtype), return_dict=False)[0]
        total_frames = self.generated_latent_frames * 4 - 3
        if video.shape[2] != total_frames:
            raise RuntimeError(f"HY-WorldPlay VAE produced {video.shape[2]} frames; expected {total_frames}.")
        video = video[:, :, self._decoded_frames :]
        frames = (
            (video / 2 + 0.5).clamp(0, 1).mul(255).to(device="cpu", dtype=torch.uint8).permute(0, 2, 3, 4, 1).numpy()[0]
        )
        self._decoded_frames = total_frames
        return np.ascontiguousarray(frames)

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Advance one native AR block; failed or cancelled calls release its state."""
        with self._execution():
            if not self.configured:
                raise RuntimeError("HY-WorldPlay realtime session is not configured.")
            try:
                return self._generate(interactions, control_segments)
            except BaseException:
                self._reset_unlocked()
                raise

    def _generate(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any] | None:
        assert self._latents is not None and self._cond_latents is not None
        assert self._generator is not None and self._timesteps is not None
        started = time.perf_counter()
        core = self.core
        start_idx = self.generated_latent_frames
        viewmats, intrinsics, actions = self._camera_conditions(interactions, control_segments)
        condition = self._condition_block()
        noise = core.prepare_latents(
            1,
            int(core.transformer.config.in_channels),
            self._latent_height,
            self._latent_width,
            self.latent_frames_per_chunk,
            self.dtype,
            self.device,
            self._generator,
        )
        self._latents = torch.cat((self._latents, noise), dim=2)
        self._cond_latents = torch.cat((self._cond_latents, condition), dim=2)
        selected: list[int] = []
        if start_idx:
            selected = core.select_ar_context_indices(
                viewmats=viewmats,
                current_frame_idx=start_idx,
                chunk_latent_frames=self.latent_frames_per_chunk,
                device=self.device,
            )
            core.rebuild_ar_vision_cache(
                latents=self._latents,
                cond_latents=self._cond_latents,
                viewmats=viewmats,
                Ks=intrinsics,
                action=actions,
                selected_frame_indices=selected,
                timesteps=self._timesteps,
                task_type="i2v",
                device=self.device,
            )
            core.scheduler.set_timesteps(self._num_inference_steps, device=self.device)
        condition_ms = (time.perf_counter() - started) * 1000.0
        model_started = time.perf_counter()
        core.denoise_ar_chunk(
            latents=self._latents,
            cond_latents=self._cond_latents,
            timesteps=self._timesteps,
            task_type="i2v",
            viewmats=viewmats,
            Ks=intrinsics,
            action=actions,
            start_idx=start_idx,
            chunk_latent_frames=self.latent_frames_per_chunk,
            selected_frame_indices=selected,
            device=self.device,
            show_progress=False,
        )
        model_done = time.perf_counter()
        if self.rank != 0:
            return None
        frames = self._decode()
        return {
            "video": frames,
            "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255", metadata={"fps": self.fps}),
            "fps": self.fps,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": {
                "condition_ms": condition_ms,
                "model_ms": (model_done - model_started) * 1000.0,
                "decode_ms": (time.perf_counter() - model_done) * 1000.0,
                "total_ms": (time.perf_counter() - started) * 1000.0,
                "context_latent_frames": len(selected),
                "generated_latent_frames": self.generated_latent_frames,
                "world_size": self.world_size,
            },
        }

    def reset(self) -> None:
        """Release this session's histories and KV without moving shared weights."""
        with self._state_lock:
            self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self.configured = False
        self.configure_metrics: dict[str, float] = {}
        self._generator: torch.Generator | None = None
        self._timesteps: torch.Tensor | None = None
        self._latents: torch.Tensor | None = None
        self._cond_latents: torch.Tensor | None = None
        self._image_condition: torch.Tensor | None = None
        self._motions: list[dict[str, float]] = []
        self._height = self._width = 0
        self._latent_height = self._latent_width = 0
        self._num_inference_steps = 0
        self._decoded_frames = 0
        self.core._kv_cache = None
        self.core._kv_cache_neg = None
        self.core.points_local = None


__all__ = ["HunyuanWorldPlayRealtimeSession"]
