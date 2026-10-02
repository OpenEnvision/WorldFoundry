"""Persistent low-latency runtime for LingBot-World Fast.

The offline generator accepts a complete video and therefore creates prompt,
VAE, and attention state inside every call.  This module exposes the model as
an autoregressive session: one rollout owns all causal state and each action
only advances it by one three-latent-frame chunk.
"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

from worldfoundry.core import autocast_context
from worldfoundry.core.vram.encoder_lifecycle import offload_module_to_cpu
from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.io.paths import checkpoint_root_path
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_module_cached
from worldfoundry.synthesis.visual_generation.inspatio_world.inspatio_world_runtime.utils.taehv import (
    TAEHV,
    StreamingTAEHV,
)

_TOKEN_TO_KEY = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "j",
    "camera_r": "l",
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), minimum)
    except ValueError:
        return max(int(default), minimum)


def _is_fsdp_module(module: Any) -> bool:
    """Return whether ``module`` is an FSDP wrapper without requiring FSDP."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        return isinstance(module, FullyShardedDataParallel)
    except (ImportError, RuntimeError):
        return "FullyShardedDataParallel" in type(module).__name__


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    if axis == "x":
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
            dtype=np.float32,
        )
    if axis == "y":
        return np.asarray(
            ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported rotation axis: {axis}")


@dataclass(slots=True)
class RealtimeCameraState:
    """Integrate piecewise-constant controls without resetting at chunk edges."""

    move_speed_per_second: float = 0.8
    rotate_speed_radians_per_second: float = float(np.deg2rad(32.0))
    pitch_limit_radians: float = float(np.deg2rad(85.0))
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    pitch: float = 0.0

    def reset(self) -> None:
        self.pose = np.eye(4, dtype=np.float32)
        self.pitch = 0.0

    def _advance(self, keys: frozenset[str], duration: float) -> None:
        if duration <= 0.0:
            return

        yaw_rate = 0.0
        if "j" in keys:
            yaw_rate -= self.rotate_speed_radians_per_second
        if "l" in keys:
            yaw_rate += self.rotate_speed_radians_per_second
        pitch_rate = 0.0
        if "i" in keys:
            pitch_rate += self.rotate_speed_radians_per_second
        if "k" in keys:
            pitch_rate -= self.rotate_speed_radians_per_second

        pitch_delta = pitch_rate * duration
        next_pitch = self.pitch + pitch_delta
        if -self.pitch_limit_radians <= next_pitch <= self.pitch_limit_radians:
            self.pitch = next_pitch
        else:
            pitch_delta = 0.0

        rotation = self.pose[:3, :3]
        translation = self.pose[:3, 3]
        rotation = _rotation_matrix("y", yaw_rate * duration) @ rotation @ _rotation_matrix("x", pitch_delta)

        forward_rate = float("w" in keys) - float("s" in keys)
        forward = np.asarray((rotation[0, 2], 0.0, rotation[2, 2]), dtype=np.float32)
        right = np.asarray((rotation[0, 0], 0.0, rotation[2, 0]), dtype=np.float32)
        forward_norm = float(np.linalg.norm(forward))
        right_norm = float(np.linalg.norm(right))
        if forward_norm > 0.0:
            forward /= forward_norm
        if right_norm > 0.0:
            right /= right_norm

        strafe_rate = float("d" in keys) - float("a" in keys)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation + (forward * forward_rate + right * strafe_rate) * (
            self.move_speed_per_second * duration
        )
        self.pose = pose

    def integrate(
        self,
        segments: Sequence[Mapping[str, Any]],
        *,
        num_frames: int,
        fps: int,
    ) -> np.ndarray:
        if num_frames < 1 or fps < 1:
            raise ValueError("num_frames and fps must be positive")

        normalized: list[tuple[float, frozenset[str]]] = []
        for segment in segments:
            duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
            keys = frozenset(str(key).lower() for key in segment.get("keys", ()))
            if duration:
                normalized.append((duration, keys))
        expected_duration = num_frames / float(fps)
        if not normalized:
            normalized = [(expected_duration, frozenset())]
        actual_duration = sum(duration for duration, _ in normalized)
        scale = expected_duration / actual_duration if actual_duration > 0.0 else 1.0
        normalized = [(duration * scale, keys) for duration, keys in normalized]

        frame_times = [(index + 1) / float(fps) for index in range(num_frames)]
        frames: list[np.ndarray] = []
        elapsed = 0.0
        frame_index = 0
        for duration, keys in normalized:
            segment_end = elapsed + duration
            while frame_index < num_frames and frame_times[frame_index] <= segment_end + 1e-8:
                target = frame_times[frame_index]
                self._advance(keys, target - elapsed)
                elapsed = target
                frames.append(self.pose.copy())
                frame_index += 1
            if segment_end > elapsed:
                self._advance(keys, segment_end - elapsed)
                elapsed = segment_end
        while frame_index < num_frames:
            target = frame_times[frame_index]
            self._advance(normalized[-1][1], target - elapsed)
            elapsed = target
            frames.append(self.pose.copy())
            frame_index += 1
        return np.stack(frames, axis=0).astype(np.float32)


class LingBotRealtimeSession:
    """One persistent three-latent-frame LingBot autoregressive rollout."""

    chunk_latent_frames = 3
    first_output_frames = 9
    steady_output_frames = 12
    temporal_window_frames = 15
    sink_frames = 3
    cache_frames = 18
    max_text_tokens = 512
    timestep_indices = (0, 179, 358, 679)
    camera_move_speed_per_second = 0.8
    camera_rotate_speed_degrees_per_second = 32.0
    normalize_camera_translation = False
    compile_namespace = "lingbot-world"
    compile_env = "WORLDFOUNDRY_LINGBOT_REALTIME_COMPILE"
    condition_prefetch_chunks = 0
    condition_replenish_chunks = 1

    def __init__(self, core_model: Any) -> None:
        required = ("model", "vae", "text_encoder", "scheduler", "config", "device")
        missing = [name for name in required if not hasattr(core_model, name)]
        if missing:
            raise TypeError(f"LingBot realtime requires the fast runtime; missing {missing}.")
        self.core = core_model
        self.device = core_model.device
        self.rank = int(getattr(core_model, "rank", 0))
        self.sp_size = int(getattr(core_model, "sp_size", 1))
        self.param_dtype = core_model.param_dtype
        self.pipe_dtype = core_model.pipe_dtype
        self.height = int(os.getenv("WORLDFOUNDRY_LINGBOT_REALTIME_HEIGHT", "464"))
        self.width = int(os.getenv("WORLDFOUNDRY_LINGBOT_REALTIME_WIDTH", "832"))
        if self.height % 16 or self.width % 16:
            raise ValueError("LingBot realtime height and width must be divisible by 16.")
        self.latent_height = self.height // int(core_model.vae_stride[1])
        self.latent_width = self.width // int(core_model.vae_stride[2])
        self.frame_sequence_length = (
            self.latent_height * self.latent_width // (int(core_model.patch_size[1]) * int(core_model.patch_size[2]))
        )
        self.chunk_sequence_length = self.chunk_latent_frames * self.frame_sequence_length
        self.max_sequence_length = int(math.ceil(self.chunk_sequence_length / self.sp_size) * self.sp_size)
        self.cache_sequence_length = self.cache_frames * self.frame_sequence_length
        self.fps = 16
        self.world_scale = float(os.getenv("WORLDFOUNDRY_LINGBOT_WORLD_SCALE", "1.271182656288147"))

        self.camera = RealtimeCameraState(
            move_speed_per_second=self.camera_move_speed_per_second,
            rotate_speed_radians_per_second=float(np.deg2rad(self.camera_rotate_speed_degrees_per_second)),
        )
        self._last_camera_pose: torch.Tensor | None = None
        self._camera_directions: torch.Tensor | None = None
        self._decoder_model: TAEHV | None = None
        self._decoder: StreamingTAEHV | None = None
        self._context: list[torch.Tensor] | None = None
        self._vae_state: dict[str, Any] | None = None
        self._pending_first_condition: torch.Tensor | None = None
        self._prefetched_conditions: list[torch.Tensor] = []
        self._self_cache: list[dict[str, Any]] | None = None
        self._cross_cache: list[dict[str, Any]] | None = None
        self._generator: torch.Generator | None = None
        self._timesteps: torch.Tensor | None = None
        self._sigmas: torch.Tensor | None = None
        self._cross_attention_initialized = False
        self._prompt: str | None = None
        self._state_lock = threading.RLock()
        self._state_ready_event: torch.cuda.Event | None = None
        self.autoregressive_index = 0
        self.configured = False
        self.configure_metrics: dict[str, float] = {}

        self._configure_attention_window()
        self._compile_model_if_enabled()
        if self.rank == 0:
            self._load_decoder()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _synchronize_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Keep the replicated I2V encoder state bit-identical across SP ranks."""

        if self.sp_size > 1 and dist.is_available() and dist.is_initialized():
            dist.broadcast(condition, src=0)
        return condition

    def _wait_for_state_stream(self, *, synchronize: bool = False) -> None:
        """Order stateful work when callers use different CUDA streams."""

        if self.device.type == "cuda" and self._state_ready_event is not None:
            if synchronize:
                # Destructive mutations must not release storage still consumed
                # by a previous stream. Prompt changes are rare, so this one-off
                # host wait is preferable to recording every cache tensor.
                self._state_ready_event.synchronize()
            else:
                torch.cuda.current_stream(self.device).wait_event(self._state_ready_event)

    def _record_state_stream(self) -> None:
        """Publish completion of the latest state mutation without a CPU sync."""

        if self.device.type != "cuda":
            return
        event = torch.cuda.Event(blocking=False)
        event.record(torch.cuda.current_stream(self.device))
        self._state_ready_event = event

    @property
    def model(self) -> Any:
        return self.core.model

    @property
    def raw_model(self) -> Any:
        model = self.model
        return getattr(model, "_orig_mod", getattr(model, "module", model))

    def _configure_attention_window(self) -> None:
        model = self.raw_model
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("LingBot fast model does not expose transformer blocks.")
        for block in blocks:
            block.self_attn.local_attn_size = self.temporal_window_frames
            block.self_attn.sink_size = self.sink_frames

    def _compile_model_if_enabled(self) -> None:
        """Compile the resident DiT once; warmup absorbs graph construction."""

        enabled = _env_flag(
            self.compile_env,
            _env_flag("WORLDFOUNDRY_REALTIME_COMPILE", True),
        )
        if not enabled:
            return
        if hasattr(self.model, "_orig_mod"):
            return
        if "FullyShardedDataParallel" in type(self.model).__name__:
            return
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            self.core.model = compile_module_cached(
                self.model,
                policy=CompilePolicy(
                    mode=os.getenv(
                        "WORLDFOUNDRY_LINGBOT_COMPILE_MODE",
                        "max-autotune-no-cudagraphs",
                    ),
                    fullgraph=False,
                    dynamic=_env_flag("WORLDFOUNDRY_LINGBOT_COMPILE_DYNAMIC", True),
                ),
                namespace=self.compile_namespace,
            )
        except Exception as exc:
            if self.rank == 0:
                print(f"[lingbot-realtime] torch.compile unavailable: {exc}", flush=True)

    @staticmethod
    def _decoder_checkpoint() -> Path:
        configured = os.getenv("WORLDFOUNDRY_LINGBOT_LIGHTTAE_PATH", "").strip()
        return Path(configured).expanduser() if configured else checkpoint_root_path("taehv", "lighttaew2_1.pth")

    def _load_decoder(self) -> None:
        if self._decoder_model is not None:
            return
        checkpoint = self._decoder_checkpoint()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"LingBot realtime LightTAE checkpoint is missing: {checkpoint}. "
                "Set WORLDFOUNDRY_LINGBOT_LIGHTTAE_PATH to lighttaew2_1.pth."
            )
        decoder = TAEHV(checkpoint_path=str(checkpoint)).eval().requires_grad_(False)
        decoder = decoder.to(device=self.device, dtype=torch.bfloat16)
        self._decoder_model = decoder
        self._decoder = StreamingTAEHV(decoder)

    def next_output_frames(self) -> int:
        return self.first_output_frames if self.autoregressive_index == 0 else self.steady_output_frames

    def realtime_spec(self) -> RealtimeSpec:
        """Describe this model's native causal playback cadence to Studio."""

        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.first_output_frames,
            steady_chunk_frames=self.steady_output_frames,
        )

    def _encode_prompt(self, prompt: str) -> list[torch.Tensor]:
        cached_encoder = getattr(self.core, "_encode_prompt", None)
        if callable(cached_encoder):
            return cached_encoder(prompt, offload_model=True)
        if self.core.t5_cpu:
            return [item.to(self.device) for item in self.core.text_encoder([prompt], torch.device("cpu"))]
        text_model = self.core.text_encoder.model
        text_is_fsdp = _is_fsdp_module(text_model)
        if not text_is_fsdp:
            text_model.to(self.device)
        context = self.core.text_encoder([prompt], self.device)
        if not text_is_fsdp and _env_flag("WORLDFOUNDRY_LINGBOT_OFFLOAD_ONESHOT_ENCODERS", True):
            offload_module_to_cpu(
                text_model,
                device=self.device,
                empty_cuda_cache=self.device.type == "cuda",
                torch_module=torch,
            )
        return context

    def _prepare_camera_grid(self) -> None:
        base = torch.tensor(
            [502.9115905761719, 503.1081237792969, 415.7778625488281, 239.7777862548828],
            device=self.device,
            dtype=torch.float32,
        )
        scale_x = self.width / 832.0
        scale_y = self.height / 480.0
        fx, fy, cx, cy = base
        fx, cx = fx * scale_x, cx * scale_x
        fy, cy = fy * scale_y, cy * scale_y
        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(self.width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        directions = torch.stack(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)), dim=-1)
        self._camera_directions = F.normalize(directions, dim=-1)

    def _initialize_model_caches(self) -> None:
        model_config = self.raw_model.config
        head_dim = int(model_config.dim) // int(model_config.num_heads)
        local_heads = int(model_config.num_heads) // self.sp_size
        self._self_cache = self.core._initialize_self_kv_cache(
            num_layers=int(model_config.num_layers),
            shape=[1, self.cache_sequence_length, local_heads, head_dim],
            dtype=self.pipe_dtype,
            device=self.device,
        )
        self._cross_cache = self._new_cross_attention_cache()

    def _new_cross_attention_cache(self) -> list[dict[str, Any]]:
        """Allocate an empty text KV cache without touching video self-KV."""

        model_config = self.raw_model.config
        head_dim = int(model_config.dim) // int(model_config.num_heads)
        return self.core._initialize_crossattn_cache(
            num_layers=int(model_config.num_layers),
            shape=[1, self.max_text_tokens, int(model_config.num_heads), head_dim],
            dtype=self.pipe_dtype,
            device=self.device,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> bool:
        """Apply a prompt to future chunks while retaining causal video state.

        Text encoding and replacement-cache allocation happen before committing
        either object.  A failed update therefore leaves the previous prompt
        usable.  The lock also prevents a chunk from observing a mixed context
        and cross-attention cache.
        """

        normalized_prompt = str(prompt or "")
        with self._state_lock:
            self._wait_for_state_stream(synchronize=True)
            try:
                if not self.configured:
                    raise RuntimeError("Configure the LingBot realtime session before updating its prompt.")
                if normalized_prompt == self._prompt:
                    return False

                context = self._encode_prompt(normalized_prompt)
                cross_cache = self._new_cross_attention_cache()
                self._context = context
                self._cross_cache = cross_cache
                self._cross_attention_initialized = False
                self._prompt = normalized_prompt
                return True
            finally:
                self._record_state_stream()

    def _image_chunk(
        self,
        image: Image.Image | None,
        *,
        first: bool,
        chunks: int = 1,
    ) -> torch.Tensor:
        if chunks < 1:
            raise ValueError("chunks must be positive")
        if first and chunks != 1:
            raise ValueError("The causal seed image must be encoded as one chunk.")
        frames = self.first_output_frames if first else self.steady_output_frames * chunks
        chunk = torch.zeros(3, frames, self.height, self.width, device=self.device)
        if first:
            if image is None:
                raise ValueError("The first LingBot realtime chunk requires an image.")
            tensor = TF.to_tensor(image.convert("RGB")).sub_(0.5).div_(0.5).to(self.device)
            resized = F.interpolate(
                tensor[None],
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )[0]
            chunk[:, 0] = resized
        return chunk

    def _encode_i2v_condition(
        self,
        image: Image.Image | None,
        *,
        first: bool,
        chunks: int = 1,
    ) -> torch.Tensor:
        if self._vae_state is None:
            raise RuntimeError("LingBot streaming VAE state is not initialized.")
        pixels = self._image_chunk(image, first=first, chunks=chunks)
        latent = self.core.vae.encode_streaming(pixels, self._vae_state)
        expected_latents = self.chunk_latent_frames * chunks
        if latent.shape[1] != expected_latents:
            raise RuntimeError(f"Streaming VAE produced {latent.shape[1]} latent frames; expected {expected_latents}.")
        mask = torch.zeros(
            4,
            expected_latents,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=latent.dtype,
        )
        if first:
            mask[:, 0] = 1.0
        return torch.cat((mask, latent), dim=0)

    def _prefetch_steady_conditions(self, chunks: int) -> None:
        """Amortize deterministic zero-frame VAE conditioning outside key input."""

        if chunks < 1:
            return
        condition = self._synchronize_condition(self._encode_i2v_condition(None, first=False, chunks=chunks))
        self._prefetched_conditions.extend(
            chunk.contiguous() for chunk in condition.split(self.chunk_latent_frames, dim=1)
        )

    @torch.no_grad()
    def configure(self, *, image: Image.Image, prompt: str, seed: int = 42, fps: int = 16) -> dict[str, Any]:
        with self._state_lock:
            self._wait_for_state_stream(synchronize=True)
            try:
                return self._configure_unlocked(image=image, prompt=prompt, seed=seed, fps=fps)
            finally:
                self._record_state_stream()

    def _configure_unlocked(
        self,
        *,
        image: Image.Image,
        prompt: str,
        seed: int,
        fps: int,
    ) -> dict[str, Any]:
        self._reset_unlocked(release_decoder=False)
        self.fps = max(int(fps), 1)
        started = time.perf_counter()
        normalized_prompt = str(prompt or "")
        self._context = self._encode_prompt(normalized_prompt)
        self._prompt = normalized_prompt
        prompt_done = time.perf_counter()
        self._vae_state = self.core.vae.prepare_streaming_encode()
        self._pending_first_condition = self._synchronize_condition(self._encode_i2v_condition(image, first=True))
        prefetch_chunks = _env_int(
            "WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS",
            self.condition_prefetch_chunks,
            minimum=0,
        )
        self._prefetch_steady_conditions(prefetch_chunks)
        condition_done = time.perf_counter()
        self._prepare_camera_grid()
        self._initialize_model_caches()
        self.core.scheduler.set_timesteps(self.core.num_train_timesteps, shift=10.0)
        indices = list(self.timestep_indices)
        self._timesteps = self.core.scheduler.timesteps[indices].to(self.device)
        self._sigmas = self.core.scheduler.sigmas[indices].to(self.device, dtype=torch.float32)
        self._generator = torch.Generator(device=self.device).manual_seed(max(int(seed), 0))
        self._cross_attention_initialized = False
        self.autoregressive_index = 0
        self.camera.reset()
        self._last_camera_pose = None
        if self._decoder is not None:
            self._decoder.reset()
        self.configured = True
        finished = time.perf_counter()
        self.configure_metrics = {
            "prompt_ms": (prompt_done - started) * 1000.0,
            "seed_vae_ms": (condition_done - prompt_done) * 1000.0,
            "cache_ms": (finished - condition_done) * 1000.0,
            "total_ms": (finished - started) * 1000.0,
        }
        return {
            "configured": True,
            "realtime_metrics": dict(self.configure_metrics),
            "realtime_spec": self.realtime_spec().to_payload(),
        }

    def _fallback_segments(self, interactions: Sequence[str], num_frames: int) -> list[dict[str, Any]]:
        keys = sorted(
            {_TOKEN_TO_KEY[token] for raw in interactions if (token := str(raw).strip().lower()) in _TOKEN_TO_KEY}
        )
        return [{"duration": num_frames / float(self.fps), "keys": keys}]

    def _camera_condition(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> torch.Tensor:
        num_frames = self.next_output_frames()
        segments = list(control_segments or self._fallback_segments(interactions, num_frames))
        poses = self.camera.integrate(segments, num_frames=num_frames, fps=self.fps)
        offset = 0 if self.autoregressive_index == 0 else 3
        indices = tuple(offset + 4 * index for index in range(self.chunk_latent_frames))
        selected = torch.from_numpy(poses[list(indices)]).to(self.device, dtype=torch.float32)
        anchor = selected[:1] if self._last_camera_pose is None else self._last_camera_pose
        previous = torch.cat((anchor, selected), dim=0)
        rotation = previous[:, :3, :3]
        translation = previous[:, :3, 3:]
        inverse = torch.eye(4, device=self.device, dtype=torch.float32)[None].repeat(previous.shape[0], 1, 1)
        inverse[:, :3, :3] = rotation.transpose(-1, -2)
        inverse[:, :3, 3:] = -torch.bmm(rotation.transpose(-1, -2), translation)
        relative = torch.bmm(inverse[:-1], previous[1:])
        translations = relative[:, :3, 3]
        if self.normalize_camera_translation:
            norms = torch.linalg.vector_norm(translations, dim=-1, keepdim=True)
            relative[:, :3, 3] = torch.where(
                norms > 0,
                translations / norms.clamp_min(1e-8),
                translations,
            )
        else:
            relative[:, :3, 3] /= self.world_scale
        self._last_camera_pose = selected[-1:].clone()

        if self._camera_directions is None:
            raise RuntimeError("Camera grid is not initialized.")
        rays_d = torch.einsum(
            "hwc,fdc->fhwd",
            self._camera_directions,
            relative[:, :3, :3],
        )
        rays_o = relative[:, None, None, :3, 3].expand(-1, self.height, self.width, -1)
        plucker = torch.cat((rays_o, rays_d), dim=-1)
        return rearrange(
            plucker,
            "f (h a) (w b) c -> 1 (c a b) f h w",
            a=int(self.core.vae_stride[1]),
            b=int(self.core.vae_stride[2]),
        ).to(self.param_dtype)

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
        if self.rank != 0:
            raise RuntimeError("Only rank zero decodes realtime frames.")
        if self._decoder is None:
            raise RuntimeError("LightTAE decoder is not initialized.")
        mean = self.core.vae.mean.to(dtype=torch.bfloat16).view(1, 1, 16, 1, 1)
        std = self.core.vae.std.to(dtype=torch.bfloat16).view(1, 1, 16, 1, 1)
        z = latent.permute(1, 0, 2, 3)[None].to(torch.bfloat16)
        z = z * std + mean
        frames: list[torch.Tensor] = []
        frame = self._decoder.decode(z)
        while frame is not None:
            frames.append(frame)
            frame = self._decoder.decode()
        if not frames:
            raise RuntimeError("LightTAE produced no frames.")
        video = torch.cat(frames, dim=1)[0]
        expected = self.next_output_frames()
        if video.shape[0] != expected:
            raise RuntimeError(f"LightTAE produced {video.shape[0]} frames; expected {expected}.")
        return video.permute(0, 2, 3, 1).mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()

    def generate(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        with self._state_lock:
            self._wait_for_state_stream()
            try:
                return self._generate_unlocked(
                    interactions=interactions,
                    control_segments=control_segments,
                    seed=seed,
                )
            finally:
                self._record_state_stream()

    @torch.no_grad()
    def _generate_unlocked(
        self,
        *,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        del seed  # A rollout owns one continuous RNG stream, seeded at configure().
        if not self.configured:
            raise RuntimeError("Configure the LingBot realtime session before generating.")
        if any(
            value is None
            for value in (
                self._context,
                self._vae_state,
                self._self_cache,
                self._cross_cache,
                self._generator,
                self._timesteps,
                self._sigmas,
            )
        ):
            raise RuntimeError("LingBot realtime session state is incomplete.")

        wall_started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        if self.autoregressive_index == 0:
            condition = self._pending_first_condition
            self._pending_first_condition = None
        else:
            if not self._prefetched_conditions:
                self._prefetch_steady_conditions(self.condition_replenish_chunks)
            condition = self._prefetched_conditions.pop(0)
        if condition is None:
            raise RuntimeError("The first LingBot I2V condition was already consumed.")
        camera = self._camera_condition(interactions, control_segments)
        events[1].record()

        latent = torch.randn(
            16,
            self.chunk_latent_frames,
            self.latent_height,
            self.latent_width,
            dtype=torch.float32,
            generator=self._generator,
            device=self.device,
        )
        current_start = self.autoregressive_index * self.chunk_sequence_length
        model_kwargs = {
            "context": [self._context[0]],
            "seq_len": self.max_sequence_length,
            "y": [condition],
            "dit_cond_dict": {"c2ws_plucker_emb": camera.chunk(1, dim=0)},
            "kv_cache": self._self_cache,
            "crossattn_cache": self._cross_cache,
            "current_start": current_start,
            "max_attention_size": self.cache_sequence_length,
            "frame_seqlen": self.frame_sequence_length,
        }

        @contextmanager
        def no_sync():
            yield

        sync_context = getattr(self.model, "no_sync", no_sync)
        with (
            autocast_context(self.device, dtype=self.param_dtype),
            sync_context(),
        ):
            x0 = latent
            for index, timestep in enumerate(self._timesteps):
                flow = self.model(
                    x=[latent],
                    t=timestep[None],
                    cross_attn_first_call=not self._cross_attention_initialized,
                    **model_kwargs,
                )[0]
                self._cross_attention_initialized = True
                sigma = self._sigmas[index].to(device=latent.device, dtype=latent.dtype)
                x0 = latent - sigma * flow
                if index + 1 < len(self._timesteps):
                    next_sigma = self._sigmas[index + 1].to(
                        device=latent.device,
                        dtype=latent.dtype,
                    )
                    noise = torch.randn(
                        x0.shape,
                        dtype=x0.dtype,
                        generator=self._generator,
                        device=x0.device,
                    )
                    latent = (1.0 - next_sigma) * x0 + next_sigma * noise
            self.model(
                x=[x0],
                t=torch.zeros(1, device=self.device),
                cross_attn_first_call=False,
                **model_kwargs,
            )
        events[2].record()

        video: np.ndarray | None = None
        if self.rank == 0:
            video = self._decode(x0)
        events[3].record()
        if self.rank == 0:
            torch.cuda.synchronize(self.device)

        metrics = {
            "ar_index": self.autoregressive_index,
            "condition_ms": events[0].elapsed_time(events[1]) if self.rank == 0 else 0.0,
            "model_ms": events[1].elapsed_time(events[2]) if self.rank == 0 else 0.0,
            "decode_ms": events[2].elapsed_time(events[3]) if self.rank == 0 else 0.0,
            "total_ms": (time.perf_counter() - wall_started) * 1000.0,
            "cache_frames": self.cache_frames,
            "world_size": self.sp_size,
        }
        self.autoregressive_index += 1
        if self.rank != 0:
            return None
        return {
            "video": video,
            "realtime_metrics": metrics,
            "realtime_spec": self.realtime_spec().to_payload(),
        }

    def reset(self, *, release_decoder: bool = False) -> None:
        with self._state_lock:
            self._wait_for_state_stream(synchronize=True)
            try:
                self._reset_unlocked(release_decoder=release_decoder)
            finally:
                self._record_state_stream()

    def _reset_unlocked(self, *, release_decoder: bool) -> None:
        self.configured = False
        self._context = None
        self._prompt = None
        self._vae_state = None
        self._pending_first_condition = None
        self._prefetched_conditions.clear()
        self._self_cache = None
        self._cross_cache = None
        self._generator = None
        self._timesteps = None
        self._sigmas = None
        self._cross_attention_initialized = False
        self.autoregressive_index = 0
        self.camera.reset()
        self._last_camera_pose = None
        if self._decoder is not None:
            self._decoder.reset()
        if release_decoder:
            self._decoder = None
            self._decoder_model = None


__all__ = ["LingBotRealtimeSession", "RealtimeCameraState"]
