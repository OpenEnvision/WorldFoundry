"""Resident causal DreamX-World AR inference.

Unlike the bidirectional 5B-Cam adapter, this runtime never feeds decoded RGB
back into the model.  It keeps the distilled model's latent blocks, attention
KV cache, cross-attention cache, causal VAE cache, and camera pose alive for the
whole interactive session.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from worldfoundry.core.geometry.transforms import euler_angles_to_rotation_matrix_zyx
from worldfoundry.core.execution.realtime.contracts import DEFAULT_REALTIME_CONTROLS, RealtimeSpec
from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime

from .checkpoints import enforce_offline_model_loading, resolve_checkpoint
from .realtime import _frame_keys, _resize_cover
NATIVE_FPS = 16
HEIGHT = 704
WIDTH = 1280
LATENT_HEIGHT = HEIGHT // 16
LATENT_WIDTH = WIDTH // 16
LATENT_CHANNELS = 48
LATENT_FRAMES_PER_BLOCK = 3
SPATIAL_TOKENS_PER_FRAME = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
FIRST_PIXEL_FRAMES = 9
STEADY_PIXEL_FRAMES = 12
DISTILLED_TIMESTEPS = (1000, 750, 500, 250)

_AR_REQUIRED = ("config.json", "model.safetensors")
_WAN_REQUIRED = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _rotation_matrix(pitch_degrees: float, yaw_degrees: float) -> np.ndarray:
    """Return the native DreamX camera rotation for pitch/yaw controls."""

    return euler_angles_to_rotation_matrix_zyx(
        np.radians([pitch_degrees, yaw_degrees, 0.0])
    ).astype(np.float32)


def _c2w(position: np.ndarray, pitch_degrees: float, yaw_degrees: float) -> np.ndarray:
    rotation = _rotation_matrix(pitch_degrees, yaw_degrees)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ position
    return np.linalg.inv(w2c).astype(np.float32)


def _pixel_key_frames(
    interactions: Sequence[str],
    control_segments: Sequence[Mapping[str, Any]] | None,
    *,
    first_block: bool,
) -> list[frozenset[str]]:
    output_frames = FIRST_PIXEL_FRAMES - 1 if first_block else STEADY_PIXEL_FRAMES
    return _frame_keys(
        control_segments,
        interactions,
        frame_count=output_frames,
        fps=NATIVE_FPS,
    )


class DreamXWorldARRealtimeSession:
    """One quality-preserving causal DreamX-World rollout."""

    def __init__(
        self,
        checkpoint_source: str | Path | None = None,
        *,
        wan_model_path: str | Path | None = None,
    ) -> None:
        enforce_offline_model_loading()
        if not torch.cuda.is_available():
            raise RuntimeError("DreamX-World AR realtime inference requires CUDA.")
        self.device = torch.device(os.getenv("WORLDFOUNDRY_DREAMX_AR_DEVICE", "cuda:0"))
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16

        checkpoint = resolve_checkpoint(
            checkpoint_source,
            default_name="DreamX-World-5B",
            required=_AR_REQUIRED,
            label="DreamX-World 5B AR",
        )
        wan_checkpoint = resolve_checkpoint(
            wan_model_path,
            default_name="Wan2.2-TI2V-5B",
            required=_WAN_REQUIRED,
            label="Wan2.2 TI2V 5B",
        )
        # Staging is an explicit deployment choice. The open-source runtime
        # never infers storage topology from a host-specific path.
        if _env_flag("WORLDFOUNDRY_DREAMX_STAGE_CHECKPOINT", False):
            checkpoint = stage_checkpoint_for_realtime(
                checkpoint,
                required_paths=_AR_REQUIRED,
                include_paths=_AR_REQUIRED,
            )
            wan_checkpoint = stage_checkpoint_for_realtime(
                wan_checkpoint,
                required_paths=_WAN_REQUIRED,
                include_paths=(
                    "Wan2.2_VAE.pth",
                    "models_t5_umt5-xxl-enc-bf16.pth",
                    "google/umt5-xxl",
                ),
            )
        self.checkpoint = checkpoint
        self.wan_model_path = wan_checkpoint

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.generator, self.text_encoder, self.vae, self.timesteps = self._load_components()
        self._kv_cache: list[dict[str, torch.Tensor]] | None = None
        self._crossattn_cache: list[dict[str, Any]] | None = None
        self._prompt_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._conditional_dict: dict[str, torch.Tensor] | None = None
        self._initial_latent: torch.Tensor | None = None
        self._configured = False
        self._first_block = True
        self._current_start_frame = 0
        self._seed = 42
        self._chunk_index = 0
        self._position = np.zeros(3, dtype=np.float32)
        self._pitch = 0.0
        self._yaw = 0.0
        self._previous_latent_c2w = np.eye(4, dtype=np.float32)
        self.last_metrics: dict[str, float] = {}

    def _load_components(self) -> tuple[Any, Any, Any, torch.Tensor]:
        from worldfoundry.base_models.diffusion_model.models.denoisers.wan_dreamx import (
            WanDiffusionCameraWrapper,
            WanTextEncoder,
            WanVAEWrapper,
        )

        started = time.perf_counter()
        generator = WanDiffusionCameraWrapper(
            model_config_path=str(self.checkpoint / "config.json"),
            num_output_frames=LATENT_FRAMES_PER_BLOCK,
            # The released AR checkpoint compresses its parallel PRoPE branch
            # from 3072 to 768 channels (attn_compress=4).
            attn_compress=4,
            local_attn_size=12,
            sink_size=3,
        )
        state = load_file(str(self.checkpoint / "model.safetensors"), device="cpu")
        missing, unexpected = generator.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "DreamX-World AR checkpoint does not match the causal model "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )
        del state
        text_encoder = WanTextEncoder(
            text_encoder_path=str(self.wan_model_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(self.wan_model_path / "google/umt5-xxl"),
            dtype=self.dtype,
        )
        vae = WanVAEWrapper(vae_path=str(self.wan_model_path / "Wan2.2_VAE.pth"))
        generator = generator.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        text_encoder = text_encoder.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        vae = vae.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        scheduler_steps = generator.get_scheduler().timesteps.cpu()
        scheduler_steps = torch.cat((scheduler_steps, torch.tensor([0], dtype=torch.float32)))
        indices = torch.tensor([1000 - value for value in DISTILLED_TIMESTEPS], dtype=torch.long)
        timesteps = scheduler_steps[indices].to(self.device)
        self.last_metrics = {"load_ms": (time.perf_counter() - started) * 1000.0}
        return generator, text_encoder, vae, timesteps

    def _initialize_caches(self) -> None:
        frame_seq_length = SPATIAL_TOKENS_PER_FRAME
        cache_frames = int(getattr(self.generator.model, "local_attn_size", 12))
        cache_tokens = cache_frames * frame_seq_length
        num_layers = len(self.generator.model.blocks)
        num_heads = int(self.generator.model.num_heads)
        head_dim = int(self.generator.model.dim) // num_heads
        if self._kv_cache is None:
            self._kv_cache = [
                {
                    "k": torch.zeros(
                        1, cache_tokens, num_heads, head_dim, device=self.device, dtype=self.dtype
                    ),
                    "v": torch.zeros(
                        1, cache_tokens, num_heads, head_dim, device=self.device, dtype=self.dtype
                    ),
                    # Cache positions drive Python slicing/control flow. Keep
                    # them on the host so every transformer block does not
                    # force a CUDA synchronization through ``Tensor.item``.
                    "global_end_index": 0,
                    "local_end_index": 0,
                }
                for _ in range(num_layers)
            ]
        if self._crossattn_cache is None:
            self._crossattn_cache = [
                {
                    "k": torch.zeros(
                        1, 512, num_heads, head_dim, device=self.device, dtype=self.dtype
                    ),
                    "v": torch.zeros(
                        1, 512, num_heads, head_dim, device=self.device, dtype=self.dtype
                    ),
                    "is_init": False,
                }
                for _ in range(num_layers)
            ]
        self._reset_cache_indices()

    def _reset_cache_indices(self) -> None:
        for cache in self._kv_cache or ():
            cache["global_end_index"] = 0
            cache["local_end_index"] = 0
        for cache in self._crossattn_cache or ():
            cache["is_init"] = False

    def _encode_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            return cached
        encoded = self.text_encoder(text_prompts=[prompt])
        if len(self._prompt_cache) >= 4:
            self._prompt_cache.pop(next(iter(self._prompt_cache)))
        self._prompt_cache[prompt] = encoded
        return encoded

    def _advance_pose(self, keys: frozenset[str]) -> np.ndarray:
        # Controls arrive at the playback cadence. Integrating one pixel frame
        # at a time preserves short taps and lets us sample the exact causal
        # VAE latent timestamps (0, 1, 5, 9, ...).
        dt = 1.0 / NATIVE_FPS
        translation = float(os.getenv("WORLDFOUNDRY_DREAMX_AR_TRANSLATION_PER_SECOND", "0.8")) * dt
        rotation = float(os.getenv("WORLDFOUNDRY_DREAMX_AR_ROTATION_DEGREES_PER_SECOND", "20")) * dt
        if "j" in keys:
            self._yaw += rotation
        if "l" in keys:
            self._yaw -= rotation
        if "i" in keys:
            self._pitch -= rotation
        if "k" in keys:
            self._pitch += rotation
        self._pitch = float(np.clip(self._pitch, -80.0, 80.0))
        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)
        forward = np.asarray(
            [-math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)],
            dtype=np.float32,
        )
        right = np.asarray([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32)
        if "w" in keys:
            self._position += forward * translation
        if "s" in keys:
            self._position -= forward * translation
        if "d" in keys:
            self._position += right * translation
        if "a" in keys:
            self._position -= right * translation
        return _c2w(self._position, self._pitch, self._yaw)

    def _camera_condition(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, torch.Tensor]:
        frame_keys = _pixel_key_frames(
            interactions,
            control_segments,
            first_block=self._first_block,
        )
        reference = self._previous_latent_c2w.copy()
        poses: list[np.ndarray] = []
        if self._first_block:
            poses.append(reference.copy())
            sample_offsets = {0, 4}
        else:
            sample_offsets = {0, 4, 8}
        for frame_index, keys in enumerate(frame_keys):
            pose = self._advance_pose(keys)
            if frame_index in sample_offsets:
                poses.append(pose)
        if len(poses) != LATENT_FRAMES_PER_BLOCK:
            raise RuntimeError(f"DreamX AR camera block has {len(poses)} latent poses; expected 3.")
        self._previous_latent_c2w = poses[-1].copy()
        # Keep transforms close to identity during long sessions. The
        # reference is the preceding sampled latent pose, not the camera state
        # at the chunk boundary (which may be several pixel frames later).
        reference_w2c = np.linalg.inv(reference)
        relative_c2w = np.stack([reference_w2c @ pose for pose in poses]).astype(np.float32)
        viewmats = torch.as_tensor(
            np.linalg.inv(relative_c2w), device=self.device, dtype=self.dtype
        )
        viewmats = viewmats.unsqueeze(1).expand(-1, SPATIAL_TOKENS_PER_FRAME, -1, -1)
        viewmats = viewmats.reshape(1, -1, 4, 4)
        intrinsics = torch.zeros(1, 3, 3, device=self.device, dtype=self.dtype)
        intrinsics[:, 0, 0] = 969.6969696969696 / (960.0 * 2)
        intrinsics[:, 1, 1] = 969.6969696969696 / (540.0 * 2)
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5
        intrinsics[:, 2, 2] = 1.0
        intrinsics = intrinsics.unsqueeze(1).expand(-1, viewmats.shape[1], -1, -1)
        return {"viewmats": viewmats, "K": intrinsics}

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=NATIVE_FPS,
            first_chunk_frames=FIRST_PIXEL_FRAMES,
            steady_chunk_frames=STEADY_PIXEL_FRAMES,
            controls=DEFAULT_REALTIME_CONTROLS,
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "resident": True,
            "autoregressive": True,
            "causal_kv_cache": True,
            "rgb_feedback": False,
            "distilled_steps": list(DISTILLED_TIMESTEPS),
            "latent_frames_per_block": LATENT_FRAMES_PER_BLOCK,
            "resolution": [HEIGHT, WIDTH],
            "device": str(self.device),
        }

    @torch.inference_mode()
    def prepare(self) -> dict[str, Any]:
        return {
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image,
        *,
        prompt: str,
        seed: int = 42,
        fps: int = NATIVE_FPS,
        **_: Any,
    ) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("DreamX-World AR realtime requires a PIL image.")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("DreamX-World AR realtime requires a user-provided prompt.")
        if int(fps) != NATIVE_FPS:
            raise ValueError(f"DreamX-World AR runs at {NATIVE_FPS} FPS.")
        self.reset()
        started = time.perf_counter()
        resized = _resize_cover(image, height=HEIGHT, width=WIDTH)
        pixels = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=self.dtype)
        self._initial_latent = self.vae.encode_to_latent(tensor)
        self._conditional_dict = self._encode_prompt(prompt)
        self._initialize_caches()
        self._seed = int(seed)
        self._configured = True
        self.last_metrics = {"condition_ms": (time.perf_counter() - started) * 1000.0}
        return {
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if not self._configured or self._conditional_dict is None or self._initial_latent is None:
            raise RuntimeError("DreamX-World AR realtime session is not configured.")
        if prompt is not None and str(prompt).strip():
            self._conditional_dict = self._encode_prompt(str(prompt).strip())
            for cache in self._crossattn_cache or ():
                cache["is_init"] = False
        started = time.perf_counter()
        camera = self._camera_condition(list(interactions or ()), control_segments)
        generator = torch.Generator(device=self.device).manual_seed(
            int(seed) if seed is not None else self._seed + self._chunk_index
        )
        noise = torch.randn(
            1,
            LATENT_FRAMES_PER_BLOCK,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        mask = torch.ones_like(noise)
        if self._first_block:
            noise[:, 0] = self._initial_latent[:, 0]
            mask[:, 0] = 0
        latents = noise
        timestep: torch.Tensor | None = None
        for index, current_timestep in enumerate(self.timesteps):
            temp = ((mask[0, :, 0, ::2, ::2]) * current_timestep).flatten()
            timestep = temp.unsqueeze(0)
            _, denoised = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=self._conditional_dict,
                y=None,
                y_camera=camera,
                timestep=timestep,
                kv_cache=self._kv_cache,
                crossattn_cache=self._crossattn_cache,
                current_start=self._current_start_frame * SPATIAL_TOKENS_PER_FRAME,
                # Every denoising evaluation needs the current block in its
                # attention context, but intermediate noisy states must not
                # become persistent history. The attention implementation
                # builds a temporary current-block view when updates are
                # disabled, then the clean-latent pass below commits once.
                cache_update_policy="none",
            )
            if index + 1 < len(self.timesteps):
                next_timestep = self.timesteps[index + 1].expand(
                    1, LATENT_FRAMES_PER_BLOCK
                ).clone()
                if self._first_block:
                    next_timestep[:, 0] = 0
                fresh_noise = torch.randn(
                    denoised.shape,
                    device=self.device,
                    dtype=self.dtype,
                    generator=generator,
                )
                latents = self.generator.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    fresh_noise.flatten(0, 1),
                    next_timestep.flatten(),
                ).unflatten(0, denoised.shape[:2])
                latents = latents * mask + noise * (1 - mask)
            else:
                latents = denoised * mask + noise * (1 - mask)
        assert timestep is not None
        context_timestep = torch.ones_like(timestep) * 0.1
        self.generator(
            noisy_image_or_video=latents,
            conditional_dict=self._conditional_dict,
            y=None,
            y_camera=camera,
            timestep=context_timestep,
            kv_cache=self._kv_cache,
            crossattn_cache=self._crossattn_cache,
            current_start=self._current_start_frame * SPATIAL_TOKENS_PER_FRAME,
            cache_update_policy="commit_detached",
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        decode_started = time.perf_counter()
        video = self.vae.decode_to_pixel(latents, use_cache=True)
        video = (video * 0.5 + 0.5).clamp_(0, 1)
        frames = (
            video[0].permute(0, 2, 3, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        expected = FIRST_PIXEL_FRAMES if self._first_block else STEADY_PIXEL_FRAMES
        if len(frames) != expected:
            raise RuntimeError(f"DreamX AR decoded {len(frames)} frames; expected {expected}.")
        self._current_start_frame += LATENT_FRAMES_PER_BLOCK
        self._first_block = False
        self._chunk_index += 1
        self.last_metrics = {
            "inference_ms": inference_ms,
            "decode_ms": decode_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
        return {
            "frames": np.ascontiguousarray(frames),
            "video": np.ascontiguousarray(frames),
            "fps": NATIVE_FPS,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def next_output_frames(self) -> int:
        return FIRST_PIXEL_FRAMES if self._first_block else STEADY_PIXEL_FRAMES

    def reset(self) -> None:
        self._conditional_dict = None
        self._initial_latent = None
        self._configured = False
        self._first_block = True
        self._current_start_frame = 0
        self._chunk_index = 0
        self._position = np.zeros(3, dtype=np.float32)
        self._pitch = 0.0
        self._yaw = 0.0
        self._previous_latent_c2w = np.eye(4, dtype=np.float32)
        self._reset_cache_indices()
        if hasattr(self, "vae"):
            self.vae.model.clear_cache()
        self.last_metrics = {}


__all__ = [
    "DreamXWorldARRealtimeSession",
]
