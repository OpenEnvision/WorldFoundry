"""Resident in-tree DreamX-World camera-controlled inference.

The released model generates image-conditioned ``1 + 4k`` frame windows.
WorldFoundry keeps its transformer, text encoder, VAE, prompt embeddings and
distributed topology resident, then feeds the final decoded frame into the
next window.  Only rank zero decodes the video; the small final RGB frame is
broadcast to the other sequence-parallel ranks for the next iteration.
"""

from __future__ import annotations

import os
import inspect
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.core.execution.realtime.contracts import DEFAULT_REALTIME_CONTROLS, RealtimeSpec
from worldfoundry.runtime.local_checkpoint_cache import stage_checkpoint_for_realtime

from .checkpoints import enforce_offline_model_loading, resolve_checkpoint as _resolve_checkpoint

NATIVE_FPS = 16
MIN_MODEL_FRAMES = 5
DEFAULT_MODEL_FRAMES = 33
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，形态畸形的肢体"
)

_CONFIG_PATH = resolve_data_path(
    "models", "runtime", "configs", "dreamx_world", "wan_ti2v_5b.yaml"
)
_DREAMX_REQUIRED = (
    "config.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
)
_WAN_REQUIRED = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/spiece.model",
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
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
def _distributed_device() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("DreamX-World realtime inference requires CUDA.")
    if "RANK" not in os.environ:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        return 0, 1, device

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if not dist.is_available():
        raise RuntimeError("This PyTorch build does not provide distributed execution.")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), device


def _snap_model_frames(value: int) -> int:
    """Snap to the temporal shape supported by the released 3:8 VAE."""

    value = max(int(value), MIN_MODEL_FRAMES)
    return 1 + 4 * max(1, (value - 1) // 4)


def _resize_cover(image: Image.Image, *, height: int, width: int) -> Image.Image:
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(round(image.width * scale), width), max(round(image.height * scale), height)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _image_tensor(image: Image.Image, *, height: int, width: int) -> torch.Tensor:
    array = np.asarray(_resize_cover(image, height=height, width=width), dtype=np.float32)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 255.0


def _frame_keys(
    segments: Sequence[Mapping[str, Any]] | None,
    interactions: Sequence[str],
    *,
    frame_count: int,
    fps: int,
) -> list[frozenset[str]]:
    fallback = frozenset(
        _TOKEN_TO_KEY[token]
        for item in interactions
        if (token := str(item).strip().lower()) in _TOKEN_TO_KEY
    )
    if not segments:
        return [fallback] * frame_count

    rows: list[tuple[float, frozenset[str]]] = []
    for segment in segments:
        duration = max(float(segment.get("duration", 0.0) or 0.0), 0.0)
        keys = frozenset(
            _TOKEN_TO_KEY[token]
            for item in (segment.get("keys") or ())
            if (token := str(item).strip().lower()) in _TOKEN_TO_KEY
        )
        if duration > 0.0:
            rows.append((duration, keys))
    if not rows:
        return [fallback] * frame_count

    total = sum(duration for duration, _ in rows)
    scale = (frame_count / float(fps)) / total
    boundaries: list[tuple[float, frozenset[str]]] = []
    elapsed = 0.0
    for duration, keys in rows:
        elapsed += duration * scale
        boundaries.append((elapsed, keys))
    result: list[frozenset[str]] = []
    row_index = 0
    for frame_index in range(frame_count):
        timestamp = (frame_index + 1) / float(fps)
        while row_index + 1 < len(boundaries) and timestamp > boundaries[row_index][0] + 1e-8:
            row_index += 1
        result.append(boundaries[row_index][1])
    return result


def _normalize_decoded_video(value: Any, *, expected_frames: int) -> np.ndarray:
    video = np.asarray(value)
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise RuntimeError(f"DreamX-World returned an invalid decoded array: {video.shape}")
    if video.shape[0] == 3:
        video = video.transpose(1, 2, 3, 0)
    elif video.shape[-1] != 3:
        raise RuntimeError(f"DreamX-World returned an invalid channel layout: {video.shape}")
    if len(video) != expected_frames:
        raise RuntimeError(
            f"DreamX-World returned {len(video)} frames; expected {expected_frames}."
        )
    if video.dtype != np.uint8:
        maximum = float(video.max(initial=0.0))
        if maximum <= 1.0001:
            video = video * 255.0
        video = np.clip(video, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(video)


class DreamXWorldRealtimeSession:
    """One persistent DreamX-World image-and-camera rollout."""

    def __init__(
        self,
        checkpoint_source: str | Path | None = None,
        *,
        wan_model_path: str | Path | None = None,
    ) -> None:
        enforce_offline_model_loading()
        self.checkpoint = _resolve_checkpoint(
            checkpoint_source,
            default_name="DreamX-World-5B-Cam",
            required=_DREAMX_REQUIRED,
            label="DreamX-World 5B Cam",
        )
        self.wan_model_path = _resolve_checkpoint(
            wan_model_path,
            default_name="Wan2.2-TI2V-5B",
            required=_WAN_REQUIRED,
            label="Wan2.2 TI2V 5B",
        )
        self.rank, self.world_size, self.device = _distributed_device()
        if self.world_size not in {1, 2, 3, 4, 6, 8}:
            raise ValueError(
                "DreamX-World supports 1, 2, 3, 4, 6, or 8 sequence-parallel GPUs "
                f"(24 attention heads); got {self.world_size}."
            )
        if _env_flag("WORLDFOUNDRY_DREAMX_STAGE_CHECKPOINT", True):
            self.checkpoint = stage_checkpoint_for_realtime(
                self.checkpoint,
                required_paths=_DREAMX_REQUIRED,
                include_paths=_DREAMX_REQUIRED,
                distributed=dist,
            )
            self.wan_model_path = stage_checkpoint_for_realtime(
                self.wan_model_path,
                required_paths=_WAN_REQUIRED,
                include_paths=(
                    "Wan2.2_VAE.pth",
                    "models_t5_umt5-xxl-enc-bf16.pth",
                    "google/umt5-xxl",
                ),
                distributed=dist,
            )
        self.weight_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.pipe, self.boundary = self._load_pipeline()
        self.fps = NATIVE_FPS
        self.model_frames = _snap_model_frames(
            int(os.getenv("WORLDFOUNDRY_DREAMX_NUM_FRAMES", str(DEFAULT_MODEL_FRAMES)))
        )
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH
        self.num_inference_steps = max(
            int(os.getenv("WORLDFOUNDRY_DREAMX_STEPS", str(DEFAULT_STEPS))), 1
        )
        self.guidance_scale = float(
            os.getenv("WORLDFOUNDRY_DREAMX_GUIDANCE_SCALE", str(DEFAULT_GUIDANCE_SCALE))
        )
        self.negative_prompt = os.getenv(
            "WORLDFOUNDRY_DREAMX_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT
        )
        self._prompt_cache: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._image: Image.Image | None = None
        self._prompt = ""
        self._seed = 42
        self._configured = False
        self._chunk_index = 0
        self._prewarmed = False
        self.last_metrics: dict[str, float] = {}

    def _load_pipeline(self) -> tuple[Any, float]:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from omegaconf import OmegaConf

        from transformers import AutoTokenizer

        from worldfoundry.base_models.diffusion_model.loaders.wan_variant import (
            load_wan_transformer,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.dreamx_world.vae import (
            AutoencoderKLWan3_8,
        )
        from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.dreamx_world.text_encoder import (
            WanT5EncoderModel,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamx_world.transformer import (
            Wan2_2Transformer3DModel,
        )
        from worldfoundry.core.distributed.sequence_parallel.runtime import (
            set_multi_gpus_devices,
        )
        from .runtime.pipeline import Wan2_2_CameraPipeline

        # xFuser attaches its sequence-parallel collectives to the already
        # initialized Studio torchrun process group.
        device = set_multi_gpus_devices(self.world_size, 1)
        if torch.device(device) != self.device:
            self.device = torch.device(device)
            torch.cuda.set_device(self.device)

        config = OmegaConf.load(_CONFIG_PATH)
        transformer_kwargs = OmegaConf.to_container(
            config["transformer_additional_kwargs"], resolve=True
        )
        transformer_kwargs["cam_method"] = "prope"
        transformer_kwargs["add_control_adapter"] = True
        transformer = load_wan_transformer(
            Wan2_2Transformer3DModel,
            str(self.checkpoint),
            additional_kwargs=transformer_kwargs,
            low_cpu_mem_usage=_env_flag("WORLDFOUNDRY_DREAMX_LOW_CPU_MEM_USAGE", True),
            torch_dtype=self.weight_dtype,
        )

        vae_kwargs = OmegaConf.to_container(config["vae_kwargs"], resolve=True)
        vae = AutoencoderKLWan3_8.from_pretrained(
            str(self.wan_model_path / str(config["vae_kwargs"]["vae_subpath"])),
            additional_kwargs=vae_kwargs,
        ).to(self.weight_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.wan_model_path / str(config["text_encoder_kwargs"]["tokenizer_subpath"])),
            local_files_only=True,
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            str(self.wan_model_path / str(config["text_encoder_kwargs"]["text_encoder_subpath"])),
            additional_kwargs=OmegaConf.to_container(
                config["text_encoder_kwargs"], resolve=True
            ),
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        )
        scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"], resolve=True)
        scheduler_parameters = set(
            inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            **{
                key: value
                for key, value in scheduler_kwargs.items()
                if key in scheduler_parameters
            }
        )
        pipe = Wan2_2_CameraPipeline(
            transformer=transformer,
            transformer_2=None,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder.eval(),
            scheduler=scheduler,
        )
        if self.world_size > 1:
            transformer.enable_multi_gpus_inference()
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for component_name in ("transformer", "vae", "text_encoder"):
            component = getattr(pipe, component_name, None)
            if component is not None:
                component.eval()
                component.requires_grad_(False)
        boundary = float(config["transformer_additional_kwargs"].get("boundary", 0.875))
        return pipe, boundary

    @property
    def output_frames(self) -> int:
        # The first model frame is the conditioning image. Dropping it avoids
        # a duplicate frame at every interactive segment boundary.
        return self.model_frames - 1

    def realtime_spec(self) -> RealtimeSpec:
        return RealtimeSpec(
            fps=self.fps,
            first_chunk_frames=self.output_frames,
            steady_chunk_frames=self.output_frames,
            controls=DEFAULT_REALTIME_CONTROLS,
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "wan_model_path": str(self.wan_model_path),
            "resident": True,
            "autoregressive": True,
            "model_frames": self.model_frames,
            "output_frames": self.output_frames,
            "resolution": [self.height, self.width],
            "sampling_steps": self.num_inference_steps,
            "world_size": self.world_size,
            "sequence_parallel": self.world_size > 1,
            "sequence_parallel_backend": "xFuser Ulysses" if self.world_size > 1 else None,
            "rank": self.rank,
            "prewarmed": self._prewarmed,
        }

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            return cached
        positive, negative = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=self.device,
            dtype=self.weight_dtype,
        )
        if len(self._prompt_cache) >= 8:
            self._prompt_cache.pop(next(iter(self._prompt_cache)))
        self._prompt_cache[prompt] = (positive, negative)
        return positive, negative

    @torch.inference_mode()
    def prepare(self) -> dict[str, Any]:
        """Pay one-time CUDA/shape initialization before accepting a session.

        User-input-only Studio entries deliberately have no packaged demo image.
        A neutral in-memory frame primes the exact interactive tensor shape
        without reading a fixture or leaking any synthetic content into a user
        rollout. One denoising step is sufficient because the same transformer
        graph is reused for every scheduler step.
        """

        if self._prewarmed or not _env_flag("WORLDFOUNDRY_DREAMX_PREWARM", True):
            return {
                "realtime_spec": self.realtime_spec().to_payload(),
                "realtime_metrics": dict(self.last_metrics),
                "runtime_info": self.runtime_info(),
            }

        started = time.perf_counter()
        frame_budget = max(
            int(os.getenv("WORLDFOUNDRY_REALTIME_CHUNK_FRAMES", "5")),
            MIN_MODEL_FRAMES,
        )
        self.model_frames = _snap_model_frames(frame_budget)
        self.height = DEFAULT_HEIGHT
        self.width = DEFAULT_WIDTH
        self.num_inference_steps = 1
        self.guidance_scale = DEFAULT_GUIDANCE_SCALE
        self._seed = 40_999
        self._image = Image.new("RGB", (self.width, self.height), (127, 127, 127))
        self._prompt = ""
        self._encode_prompt("")
        self._configured = True
        try:
            self.generate(interactions=("forward",), seed=self._seed)
        finally:
            self.reset()
        self._prewarmed = True
        self.last_metrics = {
            "compile_warmup_ms": (time.perf_counter() - started) * 1000.0
        }
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
        num_frames: int | None = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    ) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("DreamX-World realtime requires a PIL image.")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("DreamX-World realtime requires a user-provided prompt.")
        if int(fps) != NATIVE_FPS:
            raise ValueError(f"DreamX-World realtime runs at the checkpoint-native {NATIVE_FPS} FPS.")
        height, width = int(height), int(width)
        if height < 64 or width < 64 or height % 16 or width % 16:
            raise ValueError("DreamX-World height and width must be at least 64 and divisible by 16.")

        self.reset()
        self.fps = NATIVE_FPS
        self.model_frames = _snap_model_frames(num_frames or self.model_frames)
        self.height, self.width = height, width
        self.num_inference_steps = max(int(num_inference_steps), 1)
        self.guidance_scale = float(guidance_scale)
        self._seed = int(seed)
        self._image = _resize_cover(image, height=height, width=width)
        self._prompt = prompt
        started = time.perf_counter()
        self._encode_prompt(prompt)
        self._configured = True
        self.last_metrics = {"condition_ms": (time.perf_counter() - started) * 1000.0}
        return {
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def _camera_condition(
        self,
        interactions: Sequence[str],
        control_segments: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, torch.Tensor]:
        from .runtime.camera import camera_condition_from_actions

        frames = _frame_keys(
            control_segments,
            interactions,
            frame_count=self.model_frames - 1,
            fps=self.fps,
        )
        action_ids = ["".join(sorted(keys)) for keys in frames]
        base_speed = float(os.getenv("WORLDFOUNDRY_DREAMX_ACTION_SPEED", "6.0"))
        per_frame_speed = base_speed / max(self.model_frames - 1, 1)
        return camera_condition_from_actions(
            action_ids,
            [per_frame_speed] * len(action_ids),
            duration=1,
            target_length=self.model_frames,
            dtype=self.weight_dtype,
            device=self.device,
        )

    def _broadcast_last_frame(self, frame: np.ndarray | None) -> Image.Image:
        if self.world_size <= 1:
            if frame is None:
                raise RuntimeError("DreamX-World rank zero did not decode a final frame.")
            return Image.fromarray(frame, mode="RGB")
        if self.rank == 0:
            assert frame is not None
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        else:
            tensor = torch.empty(
                self.height, self.width, 3, dtype=torch.uint8, device=self.device
            )
        dist.broadcast(tensor, src=0)
        return Image.fromarray(tensor.cpu().numpy(), mode="RGB")

    @torch.inference_mode()
    def generate(
        self,
        *,
        interactions: Sequence[str] | None = None,
        control_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if not self._configured or self._image is None:
            raise RuntimeError("DreamX-World realtime session is not configured.")
        if prompt is not None and str(prompt).strip() and str(prompt).strip() != self._prompt:
            self._prompt = str(prompt).strip()

        started = time.perf_counter()
        prompt_started = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(self._prompt)
        prompt_ms = (time.perf_counter() - prompt_started) * 1000.0
        camera = self._camera_condition(list(interactions or ()), control_segments)
        generator = torch.Generator(device="cpu").manual_seed(
            self._seed + self._chunk_index if seed is None else int(seed)
        )
        latent_output = self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_frames=self.model_frames,
            height=self.height,
            width=self.width,
            generator=generator,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            start_image=_image_tensor(self._image, height=self.height, width=self.width),
            control_camera_video=camera,
            boundary=self.boundary,
            output_type="latent",
            return_dict=True,
        ).videos
        inference_ms = (time.perf_counter() - started) * 1000.0

        frames: np.ndarray | None = None
        decode_ms = 0.0
        final_frame: np.ndarray | None = None
        if self.rank == 0:
            decode_started = time.perf_counter()
            decoded = self.pipe.decode_latents(latent_output)
            full_video = _normalize_decoded_video(decoded, expected_frames=self.model_frames)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            final_frame = full_video[-1]
            frames = np.ascontiguousarray(full_video[1:])
        self._image = self._broadcast_last_frame(final_frame)
        self._chunk_index += 1
        self.last_metrics = {
            "prompt_ms": prompt_ms,
            "inference_ms": inference_ms,
            "decode_ms": decode_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
        return {
            "frames": frames,
            "video": frames,
            "fps": self.fps,
            "realtime_spec": self.realtime_spec().to_payload(),
            "realtime_metrics": dict(self.last_metrics),
            "runtime_info": self.runtime_info(),
        }

    def next_output_frames(self) -> int:
        return self.output_frames

    def reset(self) -> None:
        self._image = None
        self._prompt = ""
        self._configured = False
        self._chunk_index = 0
        self.last_metrics = {}


__all__ = [
    "DreamXWorldRealtimeSession",
]
