"""Resident, stateful Helios-Distilled autoregressive inference.

Helios is a prompt-conditioned long-video model, not a keyboard world model.
The interaction boundary is one native 9-latent/33-RGB-frame segment: callers
may replace the text prompt between segments while the model keeps its three
released latent-history bands and random stream resident on CUDA.

The implementation deliberately calls the in-tree Helios pipeline directly.
It never starts a subprocess, writes an MP4, clears CUDA caches, or reloads
weights between segments.  Multi-GPU execution is enabled only when the
process was launched with ``torchrun`` and uses the transformer's released
Diffusers context-parallel implementation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime

NATIVE_LATENT_FRAMES = 9
NATIVE_RGB_FRAMES = 33
NATIVE_HISTORY_SIZES = (16, 2, 1)
DISTILLED_PYRAMID_STEPS = (2, 2, 2)
NATIVE_HEIGHT = 384
NATIVE_WIDTH = 640

_CONTEXT_PARALLEL_BACKENDS = {
    "auto",
    "ring",
    "ulysses",
    "unified",
    "ulysses_anything",
}

_REQUIRED_CHECKPOINT_PATHS = (
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_checkpoint(source: str | Path) -> Path:
    checkpoint = Path(source).expanduser().resolve()
    missing = [item for item in _REQUIRED_CHECKPOINT_PATHS if not (checkpoint / item).exists()]
    if missing:
        raise FileNotFoundError(
            f"Helios checkpoint is incomplete at {checkpoint}; missing: {', '.join(missing)}"
        )
    try:
        model_index = json.loads((checkpoint / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Helios model_index.json from {checkpoint}.") from exc
    if not bool(model_index.get("is_distilled", False)):
        raise ValueError(
            "Resident Helios segment interaction currently requires the official "
            "Helios-Distilled checkpoint. Base and Mid remain available through "
            "the offline generation path."
        )
    return checkpoint


def _requested_context_parallel_backend() -> str:
    backend = os.getenv("WORLDFOUNDRY_HELIOS_CP_BACKEND", "auto").strip().lower()
    if backend not in _CONTEXT_PARALLEL_BACKENDS:
        allowed = ", ".join(sorted(_CONTEXT_PARALLEL_BACKENDS))
        raise ValueError(f"WORLDFOUNDRY_HELIOS_CP_BACKEND must be one of: {allowed}.")
    return backend


def _distributed_backend(context_parallel_backend: str) -> str:
    # Ulysses Anything exchanges small, per-rank shape metadata on every
    # uneven collective. A dual process group keeps that traffic on Gloo while
    # tensor collectives continue to use NCCL. ``auto`` may resolve to this
    # backend after the model configuration is available, so prepare it too.
    if context_parallel_backend in {"auto", "ulysses_anything"}:
        return "cpu:gloo,cuda:nccl"
    return "nccl"


def _distributed_device(context_parallel_backend: str = "ulysses") -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Helios realtime inference requires CUDA.")
    if "RANK" in os.environ:
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not provide distributed execution.")
        if not dist.is_initialized():
            dist.init_process_group(backend=_distributed_backend(context_parallel_backend))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.getenv("LOCAL_RANK", str(rank % torch.cuda.device_count())))
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return rank, world_size, device


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _context_parallel_sequence_lengths(
    *,
    height: int,
    width: int,
    vae_scale_factor: int,
    patch_size: Sequence[int],
) -> tuple[int, ...]:
    """Return every Helios sequence length sharded by the Diffusers CP plan.

    Each transformer block shards both the current tokens used by cross
    attention and the current-plus-history tokens used by self attention and
    the feed-forward network. The three pyramid stages therefore contribute
    six independent divisibility constraints.
    """

    p_t, p_h, p_w = (int(value) for value in patch_size)
    latent_height = int(height) // int(vae_scale_factor)
    latent_width = int(width) // int(vae_scale_factor)

    short_frames = 1 + NATIVE_HISTORY_SIZES[-1]
    short_tokens = (
        (short_frames // p_t)
        * (latent_height // p_h)
        * (latent_width // p_w)
    )
    mid_tokens = (
        _ceil_div(NATIVE_HISTORY_SIZES[1], 2 * p_t)
        * _ceil_div(latent_height, 2 * p_h)
        * _ceil_div(latent_width, 2 * p_w)
    )
    long_tokens = (
        _ceil_div(NATIVE_HISTORY_SIZES[0], 4 * p_t)
        * _ceil_div(latent_height, 4 * p_h)
        * _ceil_div(latent_width, 4 * p_w)
    )
    history_tokens = short_tokens + mid_tokens + long_tokens

    stage_height = latent_height
    stage_width = latent_width
    for _ in range(2):
        stage_height //= 2
        stage_width //= 2

    lengths: list[int] = []
    for stage_index in range(3):
        if stage_index:
            stage_height *= 2
            stage_width *= 2
        current_tokens = (
            (NATIVE_LATENT_FRAMES // p_t)
            * (stage_height // p_h)
            * (stage_width // p_w)
        )
        lengths.extend((current_tokens, history_tokens + current_tokens))
    return tuple(lengths)


def _equipartition_issues(
    backend: str,
    world_size: int,
    *,
    attention_heads: int,
    sequence_lengths: Sequence[int],
) -> tuple[str, ...]:
    if world_size <= 1 or backend == "ulysses_anything":
        return ()

    issues: list[str] = []
    incompatible = tuple(length for length in sequence_lengths if length % world_size)
    if incompatible:
        issues.append(
            f"sequence lengths {incompatible} are not divisible by the {world_size}-GPU mesh"
        )

    ulysses_degree = 1
    if backend == "ulysses":
        ulysses_degree = world_size
    elif backend == "unified":
        ulysses_degree = world_size // 2 if world_size % 2 == 0 else 0
    if ulysses_degree == 0:
        issues.append("unified context parallelism requires an even GPU count")
    elif attention_heads % ulysses_degree:
        issues.append(
            f"{attention_heads} attention heads are not divisible by Ulysses degree {ulysses_degree}"
        )
    return tuple(issues)


def _resolve_context_parallel_backend(
    requested_backend: str,
    world_size: int,
    *,
    attention_heads: int,
    sequence_lengths: Sequence[int],
) -> str:
    if requested_backend != "auto":
        return requested_backend
    if world_size <= 1:
        return "ulysses"
    issues = _equipartition_issues(
        "ulysses",
        world_size,
        attention_heads=attention_heads,
        sequence_lengths=sequence_lengths,
    )
    return "ulysses_anything" if issues else "ulysses"


def _validate_context_parallel_shape(
    backend: str,
    world_size: int,
    *,
    attention_heads: int,
    sequence_lengths: Sequence[int],
    height: int,
    width: int,
) -> None:
    issues = _equipartition_issues(
        backend,
        world_size,
        attention_heads=attention_heads,
        sequence_lengths=sequence_lengths,
    )
    if not issues:
        return
    detail = "; ".join(issues)
    raise ValueError(
        f"Helios {backend} context parallelism cannot run {height}x{width} on "
        f"{world_size} GPUs: {detail}. Use WORLDFOUNDRY_HELIOS_CP_BACKEND="
        "ulysses_anything for uneven sequence sharding, or choose a compatible GPU count."
    )


def _enable_context_parallelism(pipe: Any, backend: str, world_size: int) -> None:
    """Attach the transformer's real context-parallel implementation."""

    if world_size <= 1:
        return
    from diffusers import ContextParallelConfig

    if backend == "ring":
        config = ContextParallelConfig(ring_degree=world_size)
    elif backend == "unified":
        if world_size % 2:
            raise ValueError("Helios unified context parallelism requires an even GPU count.")
        config = ContextParallelConfig(ring_degree=2, ulysses_degree=world_size // 2)
    elif backend == "ulysses":
        config = ContextParallelConfig(ulysses_degree=world_size)
    elif backend == "ulysses_anything":
        config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
    elif backend == "auto":
        raise ValueError("Helios context-parallel backend must be resolved before enabling it.")
    else:
        raise ValueError(
            "WORLDFOUNDRY_HELIOS_CP_BACKEND must be auto, ring, ulysses, unified, or ulysses_anything."
        )
    pipe.transformer.enable_parallelism(config=config)


def _as_rgb_frames(value: Any) -> np.ndarray:
    """Normalize Diffusers batch output to contiguous uint8 ``F,H,W,3``."""

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    frames = np.asarray(value)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Helios returned an invalid RGB frame array: {frames.shape}")
    if np.issubdtype(frames.dtype, np.floating):
        scale = 255.0 if frames.size == 0 or float(np.nanmax(frames)) <= 1.5 else 1.0
        frames = np.clip(frames * scale, 0.0, 255.0).round().astype(np.uint8)
    else:
        frames = np.clip(frames, 0, 255).astype(np.uint8, copy=False)
    if len(frames) != NATIVE_RGB_FRAMES:
        raise RuntimeError(
            f"Helios decoded {len(frames)} frames; its native {NATIVE_LATENT_FRAMES}-latent "
            f"segment must decode to {NATIVE_RGB_FRAMES} RGB frames."
        )
    return np.ascontiguousarray(frames)


def _prompt_from_segments(
    prompt: str | None,
    segments: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    """Resolve an optional prompt update at the current chunk boundary."""

    resolved = str(prompt).strip() if prompt is not None else ""
    if resolved:
        return resolved
    for segment in reversed(tuple(segments or ())):
        candidate = str(segment.get("prompt") or "").strip()
        if candidate:
            return candidate
    return None


@dataclass(slots=True)
class _AutoregressiveState:
    generator: torch.Generator
    prompt: str
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    history_latents: torch.Tensor
    image_latents: torch.Tensor | None
    latents_mean: torch.Tensor
    latents_std: torch.Tensor
    indices_hidden_states: torch.Tensor
    indices_history_short: torch.Tensor
    indices_history_mid: torch.Tensor
    indices_history_long: torch.Tensor
    height: int
    width: int
    chunk_index: int = 0


class HeliosRealtimeSession:
    """One resident, quality-preserving Helios-Distilled rollout."""

    fps = 12

    def __init__(self, checkpoint_source: str | Path) -> None:
        source = _resolve_checkpoint(checkpoint_source)
        self.requested_cp_backend = _requested_context_parallel_backend()
        self.rank, self.world_size, self.device = _distributed_device(self.requested_cp_backend)
        self.checkpoint = _resolve_checkpoint(
            stage_checkpoint_for_realtime(
                source,
                required_paths=_REQUIRED_CHECKPOINT_PATHS,
                distributed=dist,
            )
        )
        self.cp_backend = self.requested_cp_backend
        self.weight_dtype = torch.bfloat16
        self._state: _AutoregressiveState | None = None
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._cp_vae_scale_factor = 8
        self._cp_patch_size = (1, 2, 2)
        self._cp_attention_heads = 40

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.pipe = self._load_pipeline()

    def _load_pipeline(self) -> Any:
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
        # Avoid 8 ranks each spawning 8 shard readers. The checkpoint is
        # already node-local, so a small per-rank pool gives better aggregate
        # load throughput without saturating CPU memory bandwidth.
        os.environ.setdefault(
            "HF_PARALLEL_LOADING_WORKERS",
            str(max(2, 8 // max(self.world_size, 1))),
        )
        from diffusers.models import AutoencoderKLWan

        from .kernels import (
            replace_all_norms_with_flash_norms,
            replace_rmsnorm_with_fp32,
            replace_rope_with_flash_rope,
        )
        from .pipeline_helios_diffusers import HeliosPipeline
        from .scheduling_helios_diffusers import HeliosScheduler
        from .transformer_helios_diffusers import HeliosTransformer3DModel

        transformer = HeliosTransformer3DModel.from_pretrained(
            self.checkpoint,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
        if not _env_flag("WORLDFOUNDRY_HELIOS_DISABLE_FLASH_ATTENTION", True):
            backend = "_flash_3_hub" if torch.cuda.get_device_capability(self.device)[0] >= 9 else "flash_hub"
            try:
                transformer.set_attention_backend(backend)
            except Exception:
                if backend != "_flash_3_hub":
                    raise
                transformer.set_attention_backend("flash_hub")

        vae = AutoencoderKLWan.from_pretrained(
            self.checkpoint,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        scheduler = HeliosScheduler.from_pretrained(self.checkpoint, subfolder="scheduler")
        pipe = HeliosPipeline.from_pretrained(
            self.checkpoint,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=self.weight_dtype,
        )
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for component_name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipe, component_name, None)
            if component is not None:
                component.eval()
                component.requires_grad_(False)
        self._cp_vae_scale_factor = int(pipe.vae_scale_factor_spatial)
        self._cp_patch_size = tuple(int(value) for value in pipe.transformer.config.patch_size)
        self._cp_attention_heads = int(pipe.transformer.config.num_attention_heads)
        native_lengths = _context_parallel_sequence_lengths(
            height=NATIVE_HEIGHT,
            width=NATIVE_WIDTH,
            vae_scale_factor=self._cp_vae_scale_factor,
            patch_size=self._cp_patch_size,
        )
        self.cp_backend = _resolve_context_parallel_backend(
            self.requested_cp_backend,
            self.world_size,
            attention_heads=self._cp_attention_heads,
            sequence_lengths=native_lengths,
        )
        _enable_context_parallelism(pipe, self.cp_backend, self.world_size)
        return pipe

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=NATIVE_RGB_FRAMES,
            steady_chunk_frames=NATIVE_RGB_FRAMES,
            controls=("prompt_update",),
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "variant": "distilled",
            "resident": True,
            "autoregressive": True,
            "latent_frames_per_chunk": NATIVE_LATENT_FRAMES,
            "rgb_frames_per_chunk": NATIVE_RGB_FRAMES,
            "history_sizes": list(NATIVE_HISTORY_SIZES),
            "pyramid_num_inference_steps_list": list(DISTILLED_PYRAMID_STEPS),
            "first_chunk_amplification": True,
            "world_size": self.world_size,
            "context_parallel": self.world_size > 1,
            "context_parallel_backend": self.cp_backend if self.world_size > 1 else None,
            "interaction": "prompt updates at native segment boundaries",
        }

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            return cached
        self.pipe._guidance_scale = 1.0
        positive, negative = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=self.device,
        )
        result = (
            positive.to(device=self.device, dtype=self.pipe.transformer.dtype),
            None if negative is None else negative.to(device=self.device, dtype=self.pipe.transformer.dtype),
        )
        # Prompt scheduling is normally short. Bound the resident cache so a
        # very long session cannot grow GPU memory without limit.
        if len(self._prompt_cache) >= 8:
            self._prompt_cache.pop(next(iter(self._prompt_cache)))
        self._prompt_cache[prompt] = result
        return result

    @torch.inference_mode()
    def configure(
        self,
        image: Image.Image | None,
        *,
        prompt: str,
        seed: int = 42,
        height: int = 384,
        width: int = 640,
        fps: int = 12,
    ) -> dict[str, Any]:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Helios realtime requires a non-empty user prompt.")
        height, width = int(height), int(width)
        # The released three-stage pyramid downsamples the VAE spatial grid by
        # another factor of four, so RGB dimensions must be divisible by 32.
        if height < 64 or width < 64 or height % 32 or width % 32:
            raise ValueError("Helios height and width must be at least 64 and divisible by 32.")
        if image is not None and not isinstance(image, Image.Image):
            raise TypeError("Helios image conditioning requires a PIL image.")

        sequence_lengths = _context_parallel_sequence_lengths(
            height=height,
            width=width,
            vae_scale_factor=self._cp_vae_scale_factor,
            patch_size=self._cp_patch_size,
        )
        _validate_context_parallel_shape(
            self.cp_backend,
            self.world_size,
            attention_heads=self._cp_attention_heads,
            sequence_lengths=sequence_lengths,
            height=height,
            width=width,
        )

        self.reset()
        started = time.perf_counter()
        self.fps = max(int(fps), 1)
        pipe = self.pipe
        pipe._guidance_scale = 1.0
        pipe._attention_kwargs = None
        pipe._interrupt = False
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        prompt_started = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(prompt)
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0

        vae_dtype = pipe.vae.dtype
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(self.device, vae_dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(self.device, vae_dtype)

        image_latents = None
        fake_image_latents = None
        image_ms = 0.0
        if image is not None:
            from diffusers.utils.torch_utils import randn_tensor

            image_started = time.perf_counter()
            prepared = pipe.video_processor.preprocess(
                image.convert("RGB"), height=height, width=width
            )
            image_latents, fake_image_latents = pipe.prepare_image_latents(
                prepared,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=NATIVE_LATENT_FRAMES,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            image_sigma = torch.rand(1, device=self.device, generator=generator) * (0.135 - 0.111) + 0.111
            image_latents = image_sigma * randn_tensor(
                image_latents.shape, generator=generator, device=self.device
            ) + (1 - image_sigma) * image_latents
            fake_sigma = torch.rand(1, device=self.device, generator=generator) * (0.135 - 0.111) + 0.111
            fake_image_latents = fake_sigma * randn_tensor(
                fake_image_latents.shape, generator=generator, device=self.device
            ) + (1 - fake_sigma) * fake_image_latents
            image_ms = (time.perf_counter() - image_started) * 1000.0

        channels = int(pipe.transformer.config.in_channels)
        latent_height = height // int(pipe.vae_scale_factor_spatial)
        latent_width = width // int(pipe.vae_scale_factor_spatial)
        history_latents = torch.zeros(
            1,
            channels,
            sum(NATIVE_HISTORY_SIZES),
            latent_height,
            latent_width,
            device=self.device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat(
                [history_latents[:, :, :-1], fake_image_latents.to(dtype=torch.float32)], dim=2
            )

        indices = torch.arange(0, 1 + sum(NATIVE_HISTORY_SIZES) + NATIVE_LATENT_FRAMES)
        prefix, history_long, history_mid, history_1x, hidden = indices.split(
            [1, *NATIVE_HISTORY_SIZES, NATIVE_LATENT_FRAMES]
        )
        history_short = torch.cat([prefix, history_1x], dim=0)
        self._state = _AutoregressiveState(
            generator=generator,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            history_latents=history_latents,
            image_latents=image_latents,
            latents_mean=latents_mean,
            latents_std=latents_std,
            indices_hidden_states=hidden.unsqueeze(0),
            indices_history_short=history_short.unsqueeze(0),
            indices_history_mid=history_mid.unsqueeze(0),
            indices_history_long=history_long.unsqueeze(0),
            height=height,
            width=width,
        )
        configure_ms = (time.perf_counter() - started) * 1000.0
        return {
            "status": "configured",
            "realtime_spec": self.realtime_spec().to_payload(),
            "runtime_info": self.runtime_info(),
            "realtime_metrics": {
                "configure_ms": configure_ms,
                "prompt_encode_ms": prompt_ms,
                "image_encode_ms": image_ms,
            },
        }

    @torch.inference_mode()
    def generate_next(
        self,
        *,
        prompt: str | None = None,
        prompt_segments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self._state
        if state is None:
            raise RuntimeError("Configure Helios with a user prompt before requesting a segment.")
        next_prompt = _prompt_from_segments(prompt, prompt_segments)
        prompt_ms = 0.0
        if next_prompt is not None and next_prompt != state.prompt:
            started = time.perf_counter()
            state.prompt_embeds, state.negative_prompt_embeds = self._encode_prompt(next_prompt)
            state.prompt = next_prompt
            prompt_ms = (time.perf_counter() - started) * 1000.0

        pipe = self.pipe
        history_long, history_mid, history_1x = state.history_latents[:, :, -sum(NATIVE_HISTORY_SIZES) :].split(
            NATIVE_HISTORY_SIZES, dim=2
        )
        if state.image_latents is None and state.chunk_index == 0:
            prefix = torch.zeros(
                (1, history_1x.shape[1], 1, history_1x.shape[-2], history_1x.shape[-1]),
                device=self.device,
                dtype=history_1x.dtype,
            )
        else:
            prefix = state.image_latents
        history_short = torch.cat([prefix, history_1x], dim=2)

        total_started = time.perf_counter()
        latents = pipe.prepare_latents(
            1,
            int(pipe.transformer.config.in_channels),
            state.height,
            state.width,
            NATIVE_RGB_FRAMES,
            dtype=torch.float32,
            device=self.device,
            generator=state.generator,
            latents=None,
        )
        model_started = time.perf_counter()
        steps = sum(DISTILLED_PYRAMID_STEPS) * (2 if state.chunk_index == 0 else 1)
        pipe._num_timesteps = steps
        with pipe.progress_bar(total=steps) as progress:
            latents = pipe.stage2_sample(
                latents=latents,
                pyramid_num_stages=3,
                pyramid_num_inference_steps_list=list(DISTILLED_PYRAMID_STEPS),
                prompt_embeds=state.prompt_embeds,
                negative_prompt_embeds=state.negative_prompt_embeds,
                guidance_scale=1.0,
                indices_hidden_states=state.indices_hidden_states,
                indices_latents_history_short=state.indices_history_short,
                indices_latents_history_mid=state.indices_history_mid,
                indices_latents_history_long=state.indices_history_long,
                latents_history_short=history_short,
                latents_history_mid=history_mid,
                latents_history_long=history_long,
                attention_kwargs=None,
                device=self.device,
                transformer_dtype=pipe.transformer.dtype,
                generator=state.generator,
                use_zero_init=False,
                zero_steps=1,
                is_amplify_first_chunk=state.chunk_index == 0,
                progress_bar=progress,
            )
        model_ms = (time.perf_counter() - model_started) * 1000.0

        if state.chunk_index == 0 and state.image_latents is None:
            # This is the released keep_first_frame anchor. It remains fixed
            # while the rolling history bands advance.
            state.image_latents = latents[:, :, :1]
        state.history_latents = torch.cat([state.history_latents, latents], dim=2)[
            :, :, -sum(NATIVE_HISTORY_SIZES) :
        ]
        decode_started = time.perf_counter()
        decode_latents = latents.to(pipe.vae.dtype) / state.latents_std + state.latents_mean
        decoded = pipe.vae.decode(decode_latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(decoded, output_type="np")
        frames = _as_rgb_frames(video)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        state.chunk_index += 1
        total_ms = (time.perf_counter() - total_started) * 1000.0

        return {
            "status": "ok",
            "frames": frames,
            "realtime_spec": self.realtime_spec().to_payload(),
            "runtime_info": self.runtime_info(),
            "realtime_metrics": {
                "chunk_index": state.chunk_index,
                "prompt_encode_ms": prompt_ms,
                "model_ms": model_ms,
                "decode_ms": decode_ms,
                "generation_ms": total_ms,
                "generated_frames": len(frames),
            },
        }

    def reset(self) -> None:
        """Drop per-session tensors while retaining all model weights."""

        self._state = None
        self._prompt_cache.clear()
        self.pipe._interrupt = False
        self.pipe._current_timestep = None


__all__ = [
    "DISTILLED_PYRAMID_STEPS",
    "HeliosRealtimeSession",
    "NATIVE_HISTORY_SIZES",
    "NATIVE_LATENT_FRAMES",
    "NATIVE_RGB_FRAMES",
]
