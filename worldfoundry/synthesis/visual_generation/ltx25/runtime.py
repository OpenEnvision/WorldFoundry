"""In-process input, SOL-Engine acceleration, and artifact adapter for native LTX-2.5.

Supports:
- Distilled 8+3 two-stage audio-video generation (``ltx-2.5-22b-distilled``),
- Full multi-step CFG dev generation (``ltx-2.5-22b-dev``, 20-40 steps),
- High-resolution 3-step Video-to-Video refinement (for MiniMax-H3 Super Acceleration),
- Full SOL-Engine stack: FirstBlockCache (FBCache), kernel fusion (QKV/KV + LTX
  Triton ops), Hopper FlashAttention-3, calibrated rowwise FP8 W8A8 + fused
  GELU-FP8 FFN, 3D-tiled Sol-Attn, split piecewise CUDA Graphs, accelerated
  Conv/Diff VideoVAE, and CFG=2 / Ulysses SP=2/4 multi-GPU parallelism.

Prediction uses local checkpoints and never downloads.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from worldfoundry.core.execution.deadline import check_deadline
from worldfoundry.runtime.assets import expand_worldfoundry_path, local_file_is_ready
from worldfoundry.runtime.env import resolve_hfd_root

if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F

    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality

    from .acceleration import (
        LTX25AcceleratedExecutor,
        LTX25AccelerationConfig,
    )
else:
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]
        F = None  # type: ignore[assignment]

UPSTREAM_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"
CHECKPOINT_REVISION = "bf86adedf518142442575d1ce2e767b7d01c8c76"
DEFAULT_CHECKPOINT_ROOT = resolve_hfd_root() / "Lightricks--LTX-2.5"

DEFAULT_COMPONENTS = {
    "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae_path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler_path": ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    "duration_head_path": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}

DEV_TRANSFORMER_RELPATH = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
CONV_VIDEO_VAE_RELPATH = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
DIFF_VIDEO_VAE_RELPATH = "vae/ltx-2.5-video-vae-bf16.safetensors"

LTX25_STAGE_1_SIGMAS: tuple[float, ...] = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX25_STAGE_2_SIGMAS: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _torch_inference_mode() -> Any:
    return torch.inference_mode() if torch is not None else (lambda fn: fn)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _normalize_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev


def _same_device(left: torch.device | str, right: torch.device | str) -> bool:
    return _normalize_device(left) == _normalize_device(right)


def _normalize_video_bcthw(video: torch.Tensor) -> torch.Tensor:
    """Normalize 4D/5D RGB video tensors (``BCTHW``, ``BFHWC``, ``FHWC``, ``CFHW``) to ``[B, 3, F, H, W]``."""
    tensor = video.detach()
    if tensor.ndim == 4:
        # Disambiguate [F, H, W, 3] (even when F == 3) vs [3, F, H, W]:
        # in FHWC spatial height/width (dims 1, 2) are > 3 while dim -1 is 3.
        if int(tensor.shape[-1]) == 3 and (int(tensor.shape[0]) != 3 or int(tensor.shape[1]) > 3):
            return tensor.permute(3, 0, 1, 2).unsqueeze(0)
        if int(tensor.shape[0]) == 3:
            return tensor.unsqueeze(0)
        raise ValueError(f"unexpected 4D video shape: {tuple(tensor.shape)}")
    if tensor.ndim == 5:
        if int(tensor.shape[-1]) == 3 and (int(tensor.shape[1]) != 3 or int(tensor.shape[2]) > 3):
            return tensor.permute(0, 4, 1, 2, 3)
        if int(tensor.shape[1]) == 3:
            return tensor
        raise ValueError(f"expected 3 RGB channels in 5D video, got {tuple(tensor.shape)}")
    raise ValueError(f"expected 4D or 5D video tensor, got {tensor.ndim}D")


def _conditioning_images(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike, Mapping)):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError("LTX-2.5 images must be a local path, mapping, or sequence of those values.")

    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            path_value = item.get("path") or item.get("image") or item.get("image_path")
            frame_index = int(item.get("frame_index", item.get("frame_idx", 0)))
            strength = float(item.get("strength", item.get("image_strength", 1.0)))
            crf = item.get("crf")
        else:
            path_value = item
            frame_index = 0
            strength = 1.0
            crf = None
        path = _expand_path(path_value)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"LTX-2.5 conditioning image {index} not found: {path}")
        if frame_index < 0:
            raise ValueError("LTX-2.5 conditioning frame_index must be non-negative.")
        if strength < 0:
            raise ValueError("LTX-2.5 conditioning strength must be non-negative.")
        resolved.append({"path": str(path), "frame_index": frame_index, "strength": strength, "crf": crf})
    return resolved


def _align_ltx25_num_frames(num_frames: int) -> int:
    """Round ``num_frames`` to the nearest valid LTX-2.5 count ``8 * k + 1`` (>= 9)."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames!r}")
    if num_frames == 1:
        return 1
    k = max(1, round((num_frames - 1) / 8))
    return 8 * k + 1


def resample_video_for_ltx25(
    video: torch.Tensor,
    *,
    target_frames: int,
    target_height: int,
    target_width: int,
    mode: Literal["bicubic", "bilinear"] = "bicubic",
    value_range: Literal["auto", "0_1", "-1_1", "0_255"] = "auto",
) -> torch.Tensor:
    """Resample 4D/5D RGB video (``[0, 1]``, ``[-1, 1]``, or ``[0, 255]``) for LTX-2.5.

    Returns a ``[B, 3, target_frames, target_height, target_width]`` tensor in ``[-1, 1]``.
    """
    if target_height <= 0 or target_width <= 0 or target_height % 32 or target_width % 32:
        raise ValueError(
            f"LTX-2.5 target height/width must be positive multiples of 32, got {target_width}x{target_height}"
        )
    if target_frames < 1 or (target_frames - 1) % 8 != 0:
        raise ValueError(f"LTX-2.5 target_frames must satisfy 8*k + 1, got {target_frames}")
    if mode not in ("bicubic", "bilinear"):
        raise ValueError(f"unsupported spatial upscale mode: {mode!r}")
    if value_range not in ("auto", "0_1", "-1_1", "0_255"):
        raise ValueError(f"unsupported value_range: {value_range!r}")

    tensor = _normalize_video_bcthw(video)
    x = tensor.float()
    if x.numel() > 0:
        if value_range == "0_255" or (value_range == "auto" and float(x.max().item()) > 1.5):
            x = x / 255.0
        elif value_range == "-1_1" or (
            value_range == "auto" and float(x.min().item()) < -0.25 and float(x.max().item()) <= 1.5
        ):
            x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)

    b, c, src_f, src_h, src_w = x.shape
    # 1. Temporal alignment (linear interpolation across frames when counts differ).
    if src_f != target_frames:
        if src_f == 1:
            x = x.expand(b, c, target_frames, src_h, src_w).contiguous()
        else:
            # Interpolate along temporal axis only: [B, C*H*W, F] -> [B, C*H*W, target_F]
            x_flat = x.permute(0, 1, 3, 4, 2).reshape(b, c * src_h * src_w, src_f)
            x_interp = F.interpolate(x_flat, size=target_frames, mode="linear", align_corners=True)
            x = x_interp.reshape(b, c, src_h, src_w, target_frames).permute(0, 1, 4, 2, 3).contiguous()

    # 2. Spatial upscaling per frame with antialiasing.
    if src_h != target_height or src_w != target_width:
        frames_2d = x.permute(0, 2, 1, 3, 4).reshape(b * target_frames, c, src_h, src_w)
        upscaled = F.interpolate(
            frames_2d,
            size=(target_height, target_width),
            mode=mode,
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        x = upscaled.reshape(b, target_frames, c, target_height, target_width).permute(0, 2, 1, 3, 4).contiguous()

    return (x * 2.0 - 1.0).contiguous()


@dataclass(frozen=True)
class LTX25RuntimePlan:
    """Normalized native request and placement for one audio-video artifact."""

    request: Mapping[str, Any]
    output_path: str
    task: str
    offload_mode: str
    quantization: str | None
    diffvae_optimization: str

    def to_dict(self) -> dict[str, Any]:
        """Describe the native recipe without importing or loading model weights."""
        return {
            "recipe": "ltx-2.5",
            "request": dict(self.request),
            "output_path": self.output_path,
            "task": self.task,
            "offload_mode": self.offload_mode,
            "quantization": self.quantization,
            "diffvae_optimization": self.diffvae_optimization,
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }


class LTX25DistilledRuntime:
    """Expose native LTX-2.5 (distilled, dev, and 3-step V2V refinement) with SOL-Engine acceleration."""

    transformer_path: Path
    text_encoder_path: Path
    video_vae_path: Path
    audio_vae_path: Path
    spatial_upsampler_path: Path
    duration_head_path: Path

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        transformer_path: str | Path | None = None,
        text_encoder_path: str | Path | None = None,
        video_vae_path: str | Path | None = None,
        audio_vae_path: str | Path | None = None,
        spatial_upsampler_path: str | Path | None = None,
        duration_head_path: str | Path | None = None,
        prompt_enhancer_gemma_root: str | Path | None = None,
        offload_mode: str = "cpu",
        quantization: str | None = None,
        diffvae_optimization: str = "chunked_eager",
        device: str = "cuda",
        model_variant: Literal["distilled", "dev"] = "distilled",
        video_vae_variant: Literal["auto", "diffvae", "conv"] = "auto",
        acceleration: LTX25AccelerationConfig | Mapping[str, Any] | str | None = None,
    ) -> None:
        self.checkpoint_dir = _expand_path(checkpoint_dir) or DEFAULT_CHECKPOINT_ROOT
        self.device = str(device)
        self.offload_mode = str(offload_mode).strip().lower()
        self.quantization = str(quantization).strip().lower() if quantization else None
        self.diffvae_optimization = str(diffvae_optimization).strip().lower()
        mv = str(model_variant).strip().lower()
        if mv not in {"distilled", "dev"}:
            raise ValueError(f"Unsupported LTX-2.5 model_variant: {model_variant!r}; expected 'distilled' or 'dev'")
        self.model_variant: Literal["distilled", "dev"] = mv  # type: ignore[assignment]
        vae_var = str(video_vae_variant).strip().lower()
        if vae_var not in {"auto", "diffvae", "conv"}:
            raise ValueError(f"Unsupported LTX-2.5 video_vae_variant: {video_vae_variant!r}")
        self.video_vae_variant: Literal["auto", "diffvae", "conv"] = vae_var  # type: ignore[assignment]
        self._validate_execution_options(self.offload_mode, self.quantization, self.diffvae_optimization)

        default_transformer_rel = (
            DEV_TRANSFORMER_RELPATH if self.model_variant == "dev" else DEFAULT_COMPONENTS["transformer_path"]
        )
        default_video_vae_rel = (
            CONV_VIDEO_VAE_RELPATH if self.video_vae_variant == "conv" else DEFAULT_COMPONENTS["video_vae_path"]
        )
        resolved_defaults = {
            **DEFAULT_COMPONENTS,
            "transformer_path": default_transformer_rel,
            "video_vae_path": default_video_vae_rel,
        }
        overrides = {
            "transformer_path": transformer_path,
            "text_encoder_path": text_encoder_path,
            "video_vae_path": video_vae_path,
            "audio_vae_path": audio_vae_path,
            "spatial_upsampler_path": spatial_upsampler_path,
            "duration_head_path": duration_head_path,
        }
        for name, relative in resolved_defaults.items():
            setattr(self, name, _expand_path(overrides[name]) or self.checkpoint_dir / relative)
        self.prompt_enhancer_gemma_root = _expand_path(prompt_enhancer_gemma_root)

        sm_major = 9
        if self.device.lower().startswith("cuda") and torch is not None and torch.cuda.is_available():
            try:
                sm_major = torch.cuda.get_device_capability(torch.device(self.device))[0]
            except Exception:
                sm_major = 9
        self.acceleration: LTX25AccelerationConfig | None
        if acceleration is None:
            self.acceleration = None
        else:
            from .acceleration import coerce_ltx25_acceleration_config

            self.acceleration = coerce_ltx25_acceleration_config(
                acceleration,
                variant=self.model_variant,
                sm_major=sm_major,
            )

        # Native recipe cache (unaccelerated default path)
        self._pipeline: NativeDiffusionPipeline | None = None
        self._pipeline_options: tuple[object, ...] | None = None

        # Direct accelerated component cache
        self._transformer_wrapper: Any = None
        self._executor: LTX25AcceleratedExecutor | None = None
        self._conditioner: Any = None
        self._video_encoder: Any = None
        self._video_decoder: Any = None
        self._video_decoder_kind: Literal["conv", "diffvae"] | None = None
        self._video_vae_config: dict[str, Any] | None = None
        self._video_vae_model_bytes: int = 0
        self._video_vae_accel_report: dict[str, Any] | None = None
        self._video_encoder_accel_report: dict[str, Any] | None = None
        self._audio_decoder: Any = None
        self._vocoder: Any = None
        self._compiled_vocoder: Any = None
        self._audio_graph_runner: Any = None
        self._audio_streams: dict[int, torch.cuda.Stream] = {}
        self._spatial_processor: Any = None
        self._last_acceleration_report: dict[str, Any] | None = None

    @staticmethod
    def _validate_execution_options(offload: str, quantization: str | None, diffvae: str) -> None:
        if offload not in {"none", "cpu", "disk"}:
            raise ValueError("LTX-2.5 offload_mode must be one of: none, cpu, disk.")
        if quantization not in {None, "fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"}:
            raise ValueError("Unsupported LTX-2.5 quantization policy.")
        if diffvae not in {"chunked_eager", "chunked_compile", "combined_compile", "blackwell_dsl"}:
            raise ValueError(f"Unsupported LTX-2.5 diffvae_optimization: {diffvae!r}")

    def configure_acceleration(
        self,
        acceleration: LTX25AccelerationConfig | Mapping[str, Any] | str | None,
    ) -> LTX25AccelerationConfig | None:
        """Set or replace the active SOL-Engine acceleration configuration."""
        sm_major = 9
        if self.device.lower().startswith("cuda") and torch is not None and torch.cuda.is_available():
            try:
                sm_major = torch.cuda.get_device_capability(torch.device(self.device))[0]
            except Exception:
                sm_major = 9
        if acceleration is None:
            parsed = None
        else:
            from .acceleration import coerce_ltx25_acceleration_config

            parsed = coerce_ltx25_acceleration_config(acceleration, variant=self.model_variant, sm_major=sm_major)
        self.acceleration = parsed
        # Drop cached executor so next load applies the updated acceleration stack cleanly.
        self._executor = None
        self._transformer_wrapper = None
        self._video_decoder = None
        self._video_encoder = None
        self._video_vae_accel_report = None
        self._video_encoder_accel_report = None
        self._audio_graph_runner = None
        self._compiled_vocoder = None
        return parsed

    def acceleration_report(self) -> dict[str, Any]:
        """Return the latest auditable SOL-Engine acceleration receipt."""
        if self._last_acceleration_report is not None:
            return dict(self._last_acceleration_report)
        report: dict[str, Any] = {
            "model_variant": self.model_variant,
            "video_vae_variant": self._video_decoder_kind or self.video_vae_variant,
            "config": self.acceleration.to_dict() if self.acceleration is not None else None,
        }
        if self._executor is not None:
            report["transformer"] = self._executor.acceleration_report()
            report["zero_fallbacks_verified"] = bool(report["transformer"].get("zero_fallbacks_verified", False))
        if self._video_vae_accel_report is not None:
            report["video_vae_decoder"] = dict(self._video_vae_accel_report)
        if self._video_encoder_accel_report is not None:
            report["video_vae_encoder"] = dict(self._video_encoder_accel_report)
        return report

    def preflight(self) -> dict[str, Any]:
        """Inspect local assets and tools without importing the model environment."""
        required = [getattr(self, name) for name in DEFAULT_COMPONENTS if name != "duration_head_path"]
        missing = [str(path) for path in required if not path.is_file()]
        incomplete = [str(path) for path in required if path.is_file() and not local_file_is_ready(path)]
        ffmpeg = shutil.which("ffmpeg")
        device_ready = self.device.lower().startswith(("cuda", "mps"))
        blocked = [] if device_ready else ["LTX-2.5 inference requires a CUDA or MPS accelerator."]
        blocked.extend(f"Checkpoint download is incomplete or empty: {path}" for path in incomplete)
        return {
            "status": "ready" if not missing and not incomplete and ffmpeg and device_ready else "blocked",
            "model_id": "ltx-2.5",
            "pipeline": "ltx-2.5",
            "model_variant": self.model_variant,
            "video_vae_variant": self.video_vae_variant,
            "checkpoint_dir": str(self.checkpoint_dir),
            "component_paths": {name: str(getattr(self, name)) for name in DEFAULT_COMPONENTS},
            "prompt_enhancer_gemma_root": str(self.prompt_enhancer_gemma_root)
            if self.prompt_enhancer_gemma_root
            else None,
            "missing_checkpoint_files": missing,
            "incomplete_checkpoint_files": incomplete,
            "missing_system_tools": [] if ffmpeg else ["ffmpeg"],
            "blocked_reasons": blocked,
            "device": self.device,
            "device_ready": device_ready,
            "ffmpeg": ffmpeg,
            "license": "LTX-2.x Community License Agreement (August 11, 2026)",
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }

    def build_plan(self, *, request: Mapping[str, Any], output_path: str | Path) -> LTX25RuntimePlan:
        """Validate public request options before constructing native components."""
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("LTX-2.5 inference requires a non-empty prompt.")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
        images = _conditioning_images(request.get("images") or request.get("image"))
        height, width = int(request.get("height", 512)), int(request.get("width", 768))
        frame_rate = float(request.get("fps", request.get("frame_rate", 24.0)))
        offload = str(request.get("offload_mode", self.offload_mode)).strip().lower()
        quantization = request.get("quantization", self.quantization)
        quantization = str(quantization).strip().lower() if quantization else None
        diffvae = str(request.get("diffvae_optimization", self.diffvae_optimization)).strip().lower()
        self._validate_execution_options(offload, quantization, diffvae)
        if self.model_variant == "dev":
            if height <= 0 or width <= 0 or height % 32 or width % 32:
                raise ValueError("LTX-2.5 dev height and width must be positive multiples of 32.")
        elif height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError("Two-stage LTX-2.5 height and width must be positive multiples of 64.")
        if frame_rate <= 0:
            raise ValueError("LTX-2.5 frame_rate must be positive.")
        auto_duration = request.get("auto_duration")
        if request.get("num_frames") is not None or auto_duration is None:
            num_frames = int(request.get("num_frames", 121))
            if num_frames < 1 or (num_frames - 1) % 8:
                raise ValueError("LTX-2.5 num_frames must satisfy num_frames = 8 * k + 1.")
            auto_duration = None
        else:
            if not self.duration_head_path.is_file():
                raise FileNotFoundError(f"LTX-2.5 auto_duration requires duration_head_path: {self.duration_head_path}")
            if not isinstance(auto_duration, Sequence) or len(auto_duration) != 2:
                raise ValueError("LTX-2.5 auto_duration must contain [min_seconds, max_seconds].")
            auto_duration = tuple(float(value) for value in auto_duration)
            num_frames = None
        enhance_prompt = bool(request.get("enhance_prompt", False))
        if enhance_prompt and (self.prompt_enhancer_gemma_root is None or not self.prompt_enhancer_gemma_root.is_dir()):
            raise FileNotFoundError("LTX-2.5 enhance_prompt requires a local prompt_enhancer_gemma_root.")
        generated_keyframes = int(request.get("num_generated_keyframes", 0))
        if generated_keyframes < 0:
            raise ValueError("LTX-2.5 num_generated_keyframes must be non-negative.")
        return LTX25RuntimePlan(
            request={
                "prompt": prompt,
                "negative_prompt": str(request.get("negative_prompt") or ""),
                "height": height,
                "width": width,
                "frame_rate": frame_rate,
                "seed": int(request.get("seed", 42)),
                "num_frames": num_frames,
                "auto_duration": auto_duration,
                "images": images,
                "enhance_prompt": enhance_prompt,
                "num_generated_keyframes": generated_keyframes,
                "num_inference_steps": request.get("num_inference_steps"),
                "guidance_scale": request.get("guidance_scale"),
            },
            output_path=str(output),
            task="image-to-video" if images else "text-to-video",
            offload_mode=offload,
            quantization=quantization,
            diffvae_optimization=diffvae,
        )

    def _native_pipeline(self, plan: LTX25RuntimePlan) -> NativeDiffusionPipeline:
        from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.optimizations import (
            OffloadMode,
            OffloadPolicy,
            RuntimePolicy,
            parse_offload_policy,
            parse_quantization_policy,
        )

        load_duration = self.duration_head_path.is_file()
        enhance = bool(plan.request["enhance_prompt"])
        options = (plan.offload_mode, plan.quantization, plan.diffvae_optimization, load_duration, enhance)
        if self._pipeline is None or self._pipeline_options != options:
            self._pipeline = None
            self._pipeline_options = None
            offload = parse_offload_policy(plan.offload_mode, owner="LTX")
            if offload.mode is OffloadMode.BLOCK and torch.device(self.device).type == "mps":
                offload = OffloadPolicy(mode=OffloadMode.COMPONENT, target="cpu")
            checkpoints = {
                "transformer": self.transformer_path,
                "text_encoder": self.text_encoder_path,
                "video_vae": self.video_vae_path,
                "audio_vae": self.audio_vae_path,
                "upsampler": self.spatial_upsampler_path,
                "duration_head": self.duration_head_path,
            }
            conditioner_options: dict[str, object] = {"load_duration_head": load_duration}
            if enhance:
                conditioner_options["prompt_enhancer_checkpoint"] = CheckpointSpec(
                    source=str(self.prompt_enhancer_gemma_root)
                )
            self._pipeline = NativeDiffusionPipeline.from_pretrained(
                "ltx-2.5",
                policy=RuntimePolicy(
                    device=self.device,
                    dtype=torch.bfloat16,
                    offload=offload,
                    quantization=parse_quantization_policy(plan.quantization, owner="LTX"),
                ),
                checkpoint_overrides={name: CheckpointSpec(source=str(path)) for name, path in checkpoints.items()},
                component_options={
                    "conditioner:main": conditioner_options,
                    "decoder:main": {"diffvae_optimization": plan.diffvae_optimization},
                },
            )
            self._pipeline_options = options
        return self._pipeline

    # ------------------------------------------------------------------ #
    # Direct SOL-Engine Component Loaders (Resident GPU / Zero Fallback)
    # ------------------------------------------------------------------ #

    def _effective_accel_config(self) -> LTX25AccelerationConfig:
        if self.acceleration is not None:
            return self.acceleration
        from .acceleration import LTX25AccelerationConfig

        return LTX25AccelerationConfig(
            fuse_qkv=False,
            fuse_cross_kv=False,
            fused_ops=False,
            attention_backend="torch",
            sol_attn=False,
            quantization="none",
            fbcache=None,
            cuda_graph="off",
            vae_channels_last_3d=False,
            vae_compile=False,
            vae_tiled=True,
            diffvae_optimization=self.diffvae_optimization,
            audio_cuda_graph=False,
            parallel_audio=False,
            compile_vocoder=False,
        )

    def _resident_policy(self) -> Any:
        from worldfoundry.base_models.diffusion_model.optimizations import (
            OffloadMode,
            OffloadPolicy,
            QuantizationPolicy,
            RuntimePolicy,
        )

        return RuntimePolicy(
            device=self.device,
            dtype=torch.bfloat16,
            offload=OffloadPolicy(mode=OffloadMode.NONE),
            quantization=QuantizationPolicy(),
        )

    def load_transformer_executor(
        self,
        *,
        sp_group: Any = None,
        sp_rank: int = 0,
    ) -> LTX25AcceleratedExecutor:
        """Load LTX-2.5 transformer weights directly onto ``self.device`` and install SOL passes."""
        if self._executor is not None:
            return self._executor

        from worldfoundry.base_models.diffusion_model.loaders import (
            CheckpointSpec,
            ModuleLoadSpec,
            NativeModuleLoader,
            safetensors_json_metadata,
        )
        from worldfoundry.base_models.diffusion_model.models.denoisers.ltx import (
            LTXAVTransformerModule,
            convert_ltx_transformer_state_dict,
        )

        policy = self._resident_policy()
        wrapper = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXAVTransformerModule,
                config={"split_checkpoint": True},
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=partial(convert_ltx_transformer_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            CheckpointSpec(source=str(self.transformer_path)),
            policy,
        )
        if not isinstance(wrapper, LTXAVTransformerModule):
            raise TypeError(f"expected LTXAVTransformerModule, got {type(wrapper).__name__}")
        wrapper.eval()
        wrapper.velocity_model.refresh_preprocessor_bindings()
        self._transformer_wrapper = wrapper
        from .acceleration import LTX25AcceleratedExecutor

        accel_cfg = self._effective_accel_config()
        self._executor = LTX25AcceleratedExecutor(
            wrapper.velocity_model,
            accel_cfg,
            device=self.device,
            sp_group=sp_group,
            sp_rank=sp_rank,
            apply_passes=True,
        )
        return self._executor

    def load_conditioner(self, *, device: str | None = None) -> Any:
        """Load the Gemma-4 12B split conditioner + LTX-2.5 embedding connectors."""
        if self._conditioner is not None:
            return self._conditioner

        from worldfoundry.base_models.diffusion_model.components import (
            ComponentBuildContext,
            ComponentKey,
            ComponentKind,
        )
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import (
            build_ltx_gemma_prompt_conditioner,
        )
        from worldfoundry.base_models.diffusion_model.optimizations import (
            OffloadMode,
            OffloadPolicy,
            QuantizationPolicy,
            RuntimePolicy,
        )

        cond_device = device or self.device
        policy = RuntimePolicy(
            device=cond_device,
            dtype=torch.bfloat16,
            offload=OffloadPolicy(mode=OffloadMode.NONE),
            quantization=QuantizationPolicy(),
        )
        checkpoints: dict[str, CheckpointSpec] = {
            "weights": CheckpointSpec(source=str(self.transformer_path)),
            "text_encoder": CheckpointSpec(source=str(self.text_encoder_path)),
        }
        load_duration = self.duration_head_path.is_file()
        if load_duration:
            checkpoints["duration_head"] = CheckpointSpec(source=str(self.duration_head_path))
        ctx = ComponentBuildContext(
            model_id="ltx-2.5",
            key=ComponentKey(ComponentKind.CONDITIONER),
            policy=policy,
            checkpoints=checkpoints,
            component_options={"load_duration_head": load_duration},
        )
        try:
            self._conditioner = build_ltx_gemma_prompt_conditioner(ctx)
        except (ModuleNotFoundError, KeyError, ValueError) as exc:
            if "gemma4_unified" not in str(exc):
                raise
            self._conditioner = self._load_native_gemma4_conditioner(policy=policy, load_duration=load_duration)
        return self._conditioner

    def _load_native_gemma4_conditioner(self, *, policy: Any, load_duration: bool) -> Any:
        """Fallback native PyTorch Gemma-4 Unified conditioner when ``transformers < 5`` lacks it."""
        from worldfoundry.base_models.diffusion_model.loaders import (
            CheckpointSpec,
            ModuleLoadSpec,
            NativeCheckpointResolver,
            NativeModuleLoader,
            safetensors_json_metadata,
        )
        from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import (
            LTXDurationHeadModule,
            LTXEmbeddingProcessorModule,
            LTXGemmaPromptConditioner,
            convert_ltx_duration_head_state_dict,
            convert_ltx_embedding_processor_state_dict,
        )
        from worldfoundry.base_models.diffusion_model.optimizations import OffloadPolicy

        from .gemma4_text import load_native_gemma4_unified_encoder

        gemma, assets = load_native_gemma4_unified_encoder(
            self.text_encoder_path,
            device=policy.device,
            dtype=policy.dtype,
        )
        transformer_ckpt = NativeCheckpointResolver().materialize(CheckpointSpec(source=str(self.transformer_path)))
        tx_config = safetensors_json_metadata(transformer_ckpt)
        processor = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXEmbeddingProcessorModule,
                config={"checkpoint_config": tx_config, "gemma_config": assets.config_dict},
                state_dict_converter=partial(convert_ltx_embedding_processor_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            CheckpointSpec(source=tuple(str(p) for p in (*transformer_ckpt.paths, *assets.weight_paths))),
            policy,
        )
        processor.eval()
        head = (
            None
            if not load_duration or not self.duration_head_path.is_file()
            else NativeModuleLoader().load(
                ModuleLoadSpec(
                    module_class=LTXDurationHeadModule,
                    config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                    state_dict_converter=convert_ltx_duration_head_state_dict,
                ),
                CheckpointSpec(source=str(self.duration_head_path)),
                replace(policy, offload=OffloadPolicy()),
            )
        )
        return LTXGemmaPromptConditioner(
            gemma,  # type: ignore[arg-type]
            processor,  # type: ignore[arg-type]
            duration_head=head,
            supports_generated_keyframes=bool(
                tx_config.get("transformer", {}).get("use_keyframes_abs_pos_embedding", False)
            ),
        )

    def offload_conditioner(self) -> None:
        """Move Gemma-4 12B conditioner weights off GPU to free ~25 GiB VRAM."""
        if self._conditioner is None:
            return
        for attr in ("gemma", "processor", "duration_head", "enhancer"):
            mod = getattr(self._conditioner, attr, None)
            if isinstance(mod, torch.nn.Module):
                mod.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_prompt(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        offload_after: bool = False,
        device: str | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode positive (and optional negative) prompt via Gemma-4 + LTX-2.5 connectors."""
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest

        target_dev = torch.device(device or self.device)
        conditioner = self.load_conditioner(device=str(target_dev))
        # Ensure conditioner lives on target device if previously offloaded.
        for attr in ("gemma", "processor"):
            mod = getattr(conditioner, attr, None)
            if isinstance(mod, torch.nn.Module):
                first_p = next(mod.parameters(), None)
                if first_p is not None and not _same_device(first_p.device, target_dev):
                    mod.to(target_dev)
                if attr == "gemma" and hasattr(mod, "encoder"):
                    mod.encoder.execution_device = target_dev

        with torch.inference_mode():
            pos_cond = conditioner.encode(
                DiffusionRequest(prompt=prompt),
                device=target_dev,
                dtype=torch.bfloat16,
            )
            result: dict[str, dict[str, torch.Tensor]] = {
                "positive": {k: v for k, v in pos_cond.positive.items() if isinstance(v, torch.Tensor)}
            }
            if negative_prompt is not None:
                neg_cond = conditioner.encode(
                    DiffusionRequest(prompt=negative_prompt),
                    device=target_dev,
                    dtype=torch.bfloat16,
                )
                result["negative"] = {k: v for k, v in neg_cond.positive.items() if isinstance(v, torch.Tensor)}
        if offload_after:
            self.offload_conditioner()
        return result

    def load_video_encoder(self) -> Any:
        """Load the LTX-2.5 3D Video VAE encoder and apply ``channels_last_3d``."""
        if self._video_encoder is not None:
            return self._video_encoder

        from worldfoundry.base_models.diffusion_model.loaders import (
            CheckpointSpec,
            ModuleLoadSpec,
            NativeModuleLoader,
            safetensors_json_metadata,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
            LTXVideoEncoderModule,
        )

        from .acceleration import (
            accelerate_ltx25_video_vae,
            convert_ltx25_conv_video_encoder_state_dict,
        )

        policy = self._resident_policy()
        encoder = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXVideoEncoderModule,
                config={},
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=partial(convert_ltx25_conv_video_encoder_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            CheckpointSpec(source=str(self.video_vae_path)),
            policy,
        )
        encoder.eval()
        accel_cfg = self._effective_accel_config()
        # Only convert layout on encoder (compile is reserved for decoder hot path).
        enc_cfg = replace(accel_cfg, vae_compile=False)
        self._video_encoder_accel_report = accelerate_ltx25_video_vae(encoder, enc_cfg)
        self._video_encoder = encoder
        return encoder

    def load_video_decoder(self) -> tuple[Any, Literal["conv", "diffvae"]]:
        """Load LTX-2.5 Conv VAE or DiffVAE video decoder with SOL-Engine acceleration."""
        if self._video_decoder is not None and self._video_decoder_kind is not None:
            return self._video_decoder, self._video_decoder_kind

        from safetensors import safe_open

        from worldfoundry.base_models.diffusion_model.loaders import (
            CheckpointSpec,
            ModuleLoadSpec,
            NativeCheckpointResolver,
            NativeModuleLoader,
            safetensors_json_metadata,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
            LTXDiffusionVideoDecoderModule,
            LTXVideoDecoderModule,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.state_dict import (
            convert_ltx_diffusion_decoder_state_dict,
        )

        ckpt = CheckpointSpec(source=str(self.video_vae_path))
        materialized = NativeCheckpointResolver().materialize(ckpt)
        meta_cfg = dict(safetensors_json_metadata(materialized))
        vae_cls = str((meta_cfg.get("vae") or {}).get("_class_name", ""))
        policy = self._resident_policy()
        accel_cfg = self._effective_accel_config()

        if vae_cls == "CausalDiffusionVAE":
            mode = accel_cfg.diffvae_optimization
            model_bytes = 0
            with safe_open(materialized.paths[0], framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118
                    if key.startswith(("vae.decoder.", "decoder.")) or "per_channel_statistics" in key:
                        model_bytes += math.prod(handle.get_slice(key).get_shape()) * 2
            decoder = NativeModuleLoader().load(
                ModuleLoadSpec(
                    module_class=LTXDiffusionVideoDecoderModule,
                    config={"mode": mode},
                    config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                    state_dict_converter=partial(convert_ltx_diffusion_decoder_state_dict, mode=mode),
                    lazy_state_dict=True,
                    buffer_aware_offload=True,
                ),
                ckpt,
                policy,
            )
            decoder.eval()
            self._video_decoder = decoder
            self._video_decoder_kind = "diffvae"
            self._video_vae_config = dict(meta_cfg.get("vae") or {})
            self._video_vae_model_bytes = model_bytes
            self._video_vae_accel_report = {
                "variant": "diffvae",
                "diffvae_mode": mode,
                "tiled": "auto",
            }
            return decoder, "diffvae"

        # Fast CausalVideoAutoencoder (Conv VAE)
        from .acceleration import (
            accelerate_ltx25_video_vae,
            convert_ltx25_conv_video_decoder_state_dict,
        )

        decoder = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXVideoDecoderModule,
                config={},
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=partial(convert_ltx25_conv_video_decoder_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            ckpt,
            policy,
        )
        decoder.eval()
        accel_report = accelerate_ltx25_video_vae(decoder, accel_cfg)
        self._video_decoder = decoder
        self._video_decoder_kind = "conv"
        self._video_vae_config = dict(meta_cfg.get("vae") or {})
        self._video_vae_accel_report = {"variant": "conv", **accel_report}
        return decoder, "conv"

    def load_audio_decoder(self) -> tuple[Any, Any]:
        """Load LTX-2.5 Audio VAE + BWE vocoder with optional compile & CUDA Graph."""
        if self._audio_decoder is not None and self._vocoder is not None:
            return self._audio_decoder, self._vocoder

        from worldfoundry.base_models.diffusion_model.loaders import (
            CheckpointSpec,
            ModuleLoadSpec,
            NativeModuleLoader,
            safetensors_json_metadata,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
            LTXAudioDecoderModule,
            LTXVocoderModule,
            convert_ltx_audio_decoder_state_dict,
            convert_ltx_vocoder_state_dict,
        )

        ckpt = CheckpointSpec(source=str(self.audio_vae_path))
        policy = self._resident_policy()
        audio = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXAudioDecoderModule,
                config={},
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=partial(convert_ltx_audio_decoder_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            ckpt,
            policy,
        )
        vocoder = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=LTXVocoderModule,
                config={},
                config_resolver=lambda checkpoint: {"checkpoint_config": safetensors_json_metadata(checkpoint)},
                state_dict_converter=partial(convert_ltx_vocoder_state_dict, lazy=True),
                lazy_state_dict=True,
                buffer_aware_offload=True,
            ),
            ckpt,
            policy,
        )
        audio.eval()
        vocoder.eval()
        self._audio_decoder = audio
        self._vocoder = vocoder
        accel_cfg = self._effective_accel_config()
        if accel_cfg.compile_vocoder and torch.device(self.device).type == "cuda":
            from worldfoundry.core.platform.compile_cache import configure_persistent_compile_cache

            configure_persistent_compile_cache(namespace="ltx25-vocoder")
            self._compiled_vocoder = torch.compile(vocoder.vocoder, mode="default")
        if accel_cfg.audio_cuda_graph and torch.device(self.device).type == "cuda":
            from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

            def _run_audio(lat: torch.Tensor) -> torch.Tensor:
                spec = audio.decoder(lat)
                voc = self._compiled_vocoder if self._compiled_vocoder is not None else vocoder.vocoder
                return voc(spec).squeeze(0).float()

            self._audio_graph_runner = InferenceCUDAGraphRunner(
                _run_audio,
                warmup=1,
                enabled=True,
                extra_key=f"ltx25_audio:{id(audio)}:{id(vocoder)}:{self._compiled_vocoder is not None}",
            )
        return audio, vocoder

    def load_spatial_processor(self) -> Any:
        """Load the LTX-2.5 x2 spatial latent upsampler + VAE statistics."""
        if self._spatial_processor is not None:
            return self._spatial_processor

        from worldfoundry.base_models.diffusion_model.components import (
            ComponentBuildContext,
            ComponentKey,
            ComponentKind,
        )
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.component import (
            build_ltx_spatial_latent_processor,
        )

        ctx = ComponentBuildContext(
            model_id="ltx-2.5",
            key=ComponentKey(ComponentKind.LATENT_PROCESSOR),
            policy=self._resident_policy(),
            checkpoints={
                "weights": CheckpointSpec(source=str(self.spatial_upsampler_path)),
                "statistics": CheckpointSpec(source=str(self.video_vae_path)),
            },
            component_options={"split_checkpoint": True},
        )
        self._spatial_processor = build_ltx_spatial_latent_processor(ctx)
        return self._spatial_processor

    def decode_video_latent(
        self,
        latent: torch.Tensor,
        *,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode ``[B, 128, F', H', W']`` LTX-2.5 latent to ``[F, H, W, 3]`` RGB in ``[0, 1]``."""
        decoder, kind = self.load_video_decoder()
        accel_cfg = self._effective_accel_config()
        dev = torch.device(self.device)
        latent = latent.to(device=dev, dtype=torch.bfloat16)

        with torch.inference_mode():
            if kind == "diffvae":
                from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_tiling import (
                    recommended_decode_tiling_config,
                )
                from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
                    diffvae_tiling_geometry_from_vae_config,
                )
                from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import (
                    DiffVAEMode,
                )
                from worldfoundry.core.vram.memory import activation_budget_bytes

                gen = generator or torch.Generator(device=dev).manual_seed(42)
                tiling = recommended_decode_tiling_config(
                    **diffvae_tiling_geometry_from_vae_config(dict(self._video_vae_config or {})),
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    mode=DiffVAEMode(accel_cfg.diffvae_optimization),
                    free_bytes=activation_budget_bytes(dev),
                    model_bytes=self._video_vae_model_bytes,
                )
                chunks = decoder(latent, tiling, gen)
                return torch.cat(chunks, dim=0)

            from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import (
                SpatialTilingConfig,
                TemporalTilingConfig,
                TilingConfig,
            )

            if accel_cfg.vae_tiled:
                tiling_cfg: TilingConfig | None = TilingConfig(
                    spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64),
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=80, tile_overlap_in_frames=24),
                )
            elif num_frames > 129:
                # Long clips (>5s, e.g. 10s=241f) use temporal-only tiling to cap 3D conv activation VRAM
                # without splitting spatial frames.
                tiling_cfg = TilingConfig(
                    spatial_config=None,
                    temporal_config=TemporalTilingConfig(tile_size_in_frames=128, tile_overlap_in_frames=24),
                )
            else:
                tiling_cfg = None
            chunks = list(decoder.decoder.decode_video(latent, tiling_config=tiling_cfg, generator=generator))
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def decode_audio_latent(self, audio_latent: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Decode ``[B, C, T, F]`` LTX-2.5 audio latent to ``(waveform, sampling_rate)``."""
        audio_mod, vocoder_mod = self.load_audio_decoder()
        dev = torch.device(self.device)
        lat = audio_latent.to(device=dev, dtype=torch.bfloat16)
        sampling_rate = int(getattr(vocoder_mod.vocoder, "output_sampling_rate", 48000))
        with torch.inference_mode():
            if self._audio_graph_runner is not None and lat.is_cuda:
                waveform = self._audio_graph_runner(lat)
                return waveform, sampling_rate
            spec = audio_mod.decoder(lat)
            voc = self._compiled_vocoder if self._compiled_vocoder is not None else vocoder_mod.vocoder
            waveform = voc(spec).squeeze(0).float()
            return waveform, sampling_rate

    def _get_audio_stream(self, device: torch.device) -> torch.cuda.Stream:
        idx = 0 if device.index is None else int(device.index)
        stream = self._audio_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._audio_streams[idx] = stream
        return stream

    def _decode_video_and_audio(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
        num_frames: int,
        decode_audio: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Decode video and optional audio, overlapping audio on a CUDA stream when safe."""
        if not decode_audio:
            video_frames = self.decode_video_latent(
                video_latent,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
            )
            return video_frames, None, 48000

        accel_cfg = self._effective_accel_config()
        dev = _normalize_device(self.device)
        # Only overlap audio on a secondary stream once audio decoder/graph are already warm,
        # avoiding concurrent torch.compile / global CUDA Graph capture races.
        audio_warm = (
            self._audio_decoder is not None
            and self._vocoder is not None
            and (self._audio_graph_runner is None or len(getattr(self._audio_graph_runner, "_graphs", ())) > 0)
        )
        if accel_cfg.parallel_audio and dev.type == "cuda" and audio_warm and not torch.is_grad_enabled():
            main_stream = torch.cuda.current_stream(dev)
            audio_stream = self._get_audio_stream(dev)
            audio_stream.wait_stream(main_stream)
            audio_in = audio_latent.to(device=dev, dtype=torch.bfloat16)
            audio_in.record_stream(audio_stream)
            audio_waveform: torch.Tensor | None = None
            audio_sr = 48000
            try:
                with torch.cuda.stream(audio_stream):
                    audio_waveform, audio_sr = self.decode_audio_latent(audio_in)
                video_frames = self.decode_video_latent(
                    video_latent,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=generator,
                )
            finally:
                main_stream.wait_stream(audio_stream)
                if audio_waveform is not None:
                    audio_waveform.record_stream(main_stream)
            return video_frames, audio_waveform, audio_sr

        video_frames = self.decode_video_latent(
            video_latent,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
        )
        audio_waveform, audio_sr = self.decode_audio_latent(audio_latent)
        return video_frames, audio_waveform, audio_sr

    @staticmethod
    def _build_modality(
        state: Any,
        *,
        sigma: torch.Tensor,
        context: torch.Tensor,
        uniform_denoise_mask: bool = True,
    ) -> Modality:
        from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality

        batch = int(state.latent.shape[0])
        sigma_vec = sigma.reshape(-1)
        if sigma_vec.numel() == 1:
            sigma_vec = sigma_vec.expand(batch)
        eff_mask = state.denoise_mask[:, :1] if uniform_denoise_mask else state.denoise_mask
        mask_sigma = sigma_vec.view(batch, *([1] * (eff_mask.ndim - 1)))
        return Modality(
            latent=state.latent,
            sigma=sigma_vec,
            timesteps=eff_mask * mask_sigma,
            positions=state.positions,
            context=context,
            context_mask=None,
            attention_mask=state.attention_mask,
            keyframes_mask=state.keyframes_mask,
        )

    @staticmethod
    def _create_stage_states(
        *,
        height: int,
        width: int,
        num_frames: int,
        fps: float,
        noise_scale: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.conditioning import add_ltx_noise
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.tools import (
            AudioLatentTools,
            VideoLatentTools,
        )
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import (
            AudioLatentShape,
            VideoLatentShape,
            VideoPixelShape,
        )

        pixels = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=fps)
        video_shape = VideoLatentShape.from_pixel_shape(pixels, latent_channels=128)
        video_tools = VideoLatentTools(VideoLatentPatchifier(1), video_shape, fps)
        video_state = video_tools.create_initial_state(device, dtype, initial_video_latent)
        keyframes_mask = torch.zeros_like(video_state.denoise_mask)
        frame_tokens = video_tools.patchifier.get_token_count(video_shape._replace(frames=1))
        keyframes_mask[:, :frame_tokens] = 1.0
        video_state = video_state.with_updates(keyframes_mask=keyframes_mask)
        video_state = add_ltx_noise(video_state, scale=noise_scale, generator=generator)

        audio_shape = AudioLatentShape.from_video_pixel_shape(pixels)
        audio_tools = AudioLatentTools(AudioPatchifier(1), audio_shape)
        audio_state = audio_tools.create_initial_state(device, dtype, initial_audio_latent)
        audio_state = add_ltx_noise(audio_state, scale=noise_scale, generator=generator)
        return video_state, audio_state, video_shape, audio_shape

    # ------------------------------------------------------------------ #
    # High-Level Accelerated Execution Modes: Distilled / Dev / 3-Step V2V
    # ------------------------------------------------------------------ #

    @_torch_inference_mode()
    def generate_dev(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 768,
        num_frames: int = 65,
        fps: float = 24.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        seed: int = 42,
        prompt_embeds: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        decode_audio: bool = True,
        offload_text_encoder: bool = True,
        cfg_group: Any = None,
        cfg_rank: int | None = None,
        cfg_world_size: int | None = None,
        sp_group: Any = None,
        sp_rank: int = 0,
    ) -> dict[str, Any]:
        """Run multi-step CFG LTX-2.5 22B Dev generation through the SOL-Engine stack."""
        from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )
        from worldfoundry.base_models.diffusion_model.schedulers.ltx import LTXProjectedEulerScheduler
        from worldfoundry.base_models.diffusion_model.schedulers.ltx_sigma import LTX2Scheduler
        from worldfoundry.core.nn.transformer.dit import velocity_to_denoised

        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(f"LTX-2.5 height and width must be positive multiples of 32, got {width}x{height}")
        if num_frames < 1 or (num_frames - 1) % 8 != 0:
            raise ValueError(f"LTX-2.5 num_frames must satisfy 8 * k + 1, got {num_frames}")
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")

        dev = torch.device(self.device)
        t_text = 0.0
        if prompt_embeds is None:
            _cuda_sync(dev)
            t0 = time.perf_counter()
            cond = self.encode_prompt(
                prompt,
                negative_prompt=negative_prompt if float(guidance_scale) != 1.0 else None,
                offload_after=offload_text_encoder,
            )
            _cuda_sync(dev)
            t_text = time.perf_counter() - t0
        else:
            cond = {
                branch: {k: v.to(device=dev, dtype=torch.bfloat16) for k, v in mapping.items()}
                for branch, mapping in prompt_embeds.items()
            }

        executor = self.load_transformer_executor(sp_group=sp_group, sp_rank=sp_rank)
        executor.begin_request()

        from .acceleration import LTX25CFGParallelRunner

        accel_cfg = self._effective_accel_config()
        if accel_cfg.parallel.cfg_parallel == 2:
            cfg_runner = LTX25CFGParallelRunner(
                cfg_group=cfg_group,
                cfg_rank=cfg_rank,
                cfg_world_size=cfg_world_size,
            )
            if cfg_runner.world_size not in (1, 2):
                raise ValueError(
                    f"LTX-2.5 cfg_parallel=2 requires a 2-rank cfg_group (got world_size={cfg_runner.world_size}); "
                    "pass an explicit 2-rank cfg_group when running 2D CFG+SP parallelism"
                )
        else:
            cfg_runner = LTX25CFGParallelRunner(
                cfg_group=None,
                cfg_rank=0,
                cfg_world_size=1,
            )

        gen = torch.Generator(device=dev).manual_seed(int(seed))
        latent_f = 1 + (num_frames - 1) // 8
        latent_h = height // 32
        latent_w = width // 32
        dummy_shape_tensor = torch.empty((1, 128, latent_f, latent_h, latent_w), device="meta")
        sigmas_tensor = LTX2Scheduler().execute(steps=int(num_inference_steps), latent=dummy_shape_tensor)
        sigmas = tuple(float(v) for v in sigmas_tensor.tolist())
        scheduler = LTXProjectedEulerScheduler(sigmas)
        schedule = scheduler.schedule(
            SamplingConfig(num_inference_steps=len(sigmas) - 1), device=dev, dtype=torch.bfloat16
        )

        video_state, audio_state, video_shape, audio_shape = self._create_stage_states(
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            noise_scale=float(schedule[0].timestep.item()),
            generator=gen,
            device=dev,
            dtype=torch.bfloat16,
        )

        pos_v_ctx = cond["positive"]["video_context"]
        pos_a_ctx = cond["positive"]["audio_context"]
        neg_v_ctx = cond.get("negative", {}).get("video_context", pos_v_ctx)
        neg_a_ctx = cond.get("negative", {}).get("audio_context", pos_a_ctx)

        _cuda_sync(dev)
        t_dit_start = time.perf_counter()
        total_steps = len(schedule)
        for step in schedule:
            v_pos_mod = self._build_modality(
                video_state, sigma=step.timestep, context=pos_v_ctx, uniform_denoise_mask=True
            )
            a_pos_mod = self._build_modality(
                audio_state, sigma=step.timestep, context=pos_a_ctx, uniform_denoise_mask=True
            )
            v_neg_mod = self._build_modality(
                video_state, sigma=step.timestep, context=neg_v_ctx, uniform_denoise_mask=True
            )
            a_neg_mod = self._build_modality(
                audio_state, sigma=step.timestep, context=neg_a_ctx, uniform_denoise_mask=True
            )

            def _run_branch(
                v_mod: Modality,
                a_mod: Modality,
                bname: str,
                step_idx: int = step.index,
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                vx, ax = executor.forward_velocity(
                    v_mod,
                    a_mod,
                    step_index=step_idx,
                    total_steps=total_steps,
                    stage_index=0,
                    branch=bname,
                )
                assert vx is not None and ax is not None
                return (
                    velocity_to_denoised(v_mod.latent, vx, v_mod.timesteps),
                    velocity_to_denoised(a_mod.latent, ax, a_mod.timesteps),
                )

            v_pred, a_pred = cfg_runner.run_cfg_step(
                run_positive=partial(_run_branch, v_pos_mod, a_pos_mod, "positive"),
                run_negative=partial(_run_branch, v_neg_mod, a_neg_mod, "negative"),
                guidance_scale=float(guidance_scale),
            )
            assert v_pred is not None and a_pred is not None
            video_state = video_state.with_updates(
                latent=scheduler.step_state(v_pred, step, video_state, generator=gen)
            )
            audio_state = audio_state.with_updates(
                latent=scheduler.step_state(a_pred, step, audio_state, generator=gen)
            )
        _cuda_sync(dev)
        t_dit = time.perf_counter() - t_dit_start
        executor.finalize_request_caches()

        video_latent = VideoLatentPatchifier(1).unpatchify(video_state.latent, video_shape)
        audio_latent = AudioPatchifier(1).unpatchify(audio_state.latent, audio_shape)

        _cuda_sync(dev)
        t_vae_start = time.perf_counter()
        video_frames, audio_waveform, audio_sr = self._decode_video_and_audio(
            video_latent,
            audio_latent,
            height=height,
            width=width,
            num_frames=num_frames,
            decode_audio=decode_audio,
            generator=gen,
        )
        _cuda_sync(dev)
        t_vae = time.perf_counter() - t_vae_start

        tx_report = executor.acceleration_report()
        report = {
            "model_variant": "dev",
            "video_vae_variant": self._video_decoder_kind,
            "transformer": tx_report,
            "cfg_parallel": cfg_runner.report(),
            "video_vae_decoder": dict(self._video_vae_accel_report or {}),
            "zero_fallbacks_verified": bool(tx_report.get("zero_fallbacks_verified", False)),
        }
        self._last_acceleration_report = report

        return {
            "video": video_frames,
            "video_latent": video_latent,
            "audio": audio_waveform,
            "audio_sampling_rate": audio_sr,
            "timings": {
                "text_encode_s": round(t_text, 4),
                "dit_s": round(t_dit, 4),
                "dit_ms_per_step": round(1000.0 * t_dit / total_steps, 2),
                "vae_decode_s": round(t_vae, 4),
                "e2e_s": round(t_dit + t_vae, 4),
                "e2e_with_text_s": round(t_text + t_dit + t_vae, 4),
            },
            "acceleration_report": report,
        }

    @_torch_inference_mode()
    def generate_distilled(
        self,
        *,
        prompt: str,
        height: int = 768,
        width: int = 1280,
        num_frames: int = 65,
        fps: float = 24.0,
        seed: int = 42,
        stage1_sigmas: tuple[float, ...] = LTX25_STAGE_1_SIGMAS,
        stage2_sigmas: tuple[float, ...] = LTX25_STAGE_2_SIGMAS,
        prompt_embeds: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        decode_audio: bool = True,
        offload_text_encoder: bool = True,
        sp_group: Any = None,
        sp_rank: int = 0,
    ) -> dict[str, Any]:
        """Run two-stage (8+3 step) distilled LTX-2.5 22B through the SOL-Engine stack."""
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )
        from worldfoundry.base_models.diffusion_model.schedulers.ltx import (
            LTXProjectedAncestralEulerScheduler,
            LTXProjectedEulerScheduler,
        )
        from worldfoundry.core.nn.transformer.dit import velocity_to_denoised

        if height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError(f"Two-stage LTX-2.5 height and width must be multiples of 64, got {width}x{height}")
        if num_frames < 1 or (num_frames - 1) % 8 != 0:
            raise ValueError(f"LTX-2.5 num_frames must satisfy 8 * k + 1, got {num_frames}")

        dev = torch.device(self.device)
        t_text = 0.0
        if prompt_embeds is None:
            _cuda_sync(dev)
            t0 = time.perf_counter()
            cond = self.encode_prompt(prompt, negative_prompt=None, offload_after=offload_text_encoder)
            _cuda_sync(dev)
            t_text = time.perf_counter() - t0
        else:
            cond = {
                branch: {k: v.to(device=dev, dtype=torch.bfloat16) for k, v in mapping.items()}
                for branch, mapping in prompt_embeds.items()
            }

        executor = self.load_transformer_executor(sp_group=sp_group, sp_rank=sp_rank)
        processor = self.load_spatial_processor()
        executor.begin_request()

        gen = torch.Generator(device=dev).manual_seed(int(seed))
        step_gen_s1 = torch.Generator(device=dev).manual_seed(int(seed) + 10000)
        step_gen_s2 = torch.Generator(device=dev).manual_seed(int(seed) + 10000)

        pos_v_ctx = cond["positive"]["video_context"]
        pos_a_ctx = cond["positive"]["audio_context"]

        # Stage 1: 8-step half-res ancestral Euler.
        sched1 = LTXProjectedAncestralEulerScheduler(stage1_sigmas)
        s1_steps = sched1.schedule(
            SamplingConfig(num_inference_steps=len(stage1_sigmas) - 1),
            device=dev,
            dtype=torch.bfloat16,
        )
        v_state, a_state, _, _ = self._create_stage_states(
            height=height // 2,
            width=width // 2,
            num_frames=num_frames,
            fps=fps,
            noise_scale=float(s1_steps[0].timestep.item()),
            generator=gen,
            device=dev,
            dtype=torch.bfloat16,
        )

        _cuda_sync(dev)
        t_s1_start = time.perf_counter()
        for step in s1_steps:
            v_mod = self._build_modality(v_state, sigma=step.timestep, context=pos_v_ctx, uniform_denoise_mask=True)
            a_mod = self._build_modality(a_state, sigma=step.timestep, context=pos_a_ctx, uniform_denoise_mask=True)
            vx, ax = executor.forward_velocity(
                v_mod,
                a_mod,
                step_index=step.index,
                total_steps=len(s1_steps),
                stage_index=0,
                branch="positive",
            )
            assert vx is not None and ax is not None
            v_pred = velocity_to_denoised(v_mod.latent, vx, v_mod.timesteps)
            a_pred = velocity_to_denoised(a_mod.latent, ax, a_mod.timesteps)
            v_state = v_state.with_updates(latent=sched1.step_state(v_pred, step, v_state, generator=step_gen_s1))
            a_state = a_state.with_updates(latent=sched1.step_state(a_pred, step, a_state, generator=step_gen_s1))
        _cuda_sync(dev)
        t_s1 = time.perf_counter() - t_s1_start

        # Latent x2 spatial upsampler.
        _cuda_sync(dev)
        t_up_start = time.perf_counter()
        req_stub = DiffusionRequest(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            inputs={"frame_rate": fps},
        )
        upsampled = processor.process({"video": v_state, "audio": a_state}, req_stub)
        _cuda_sync(dev)
        t_up = time.perf_counter() - t_up_start

        # Stage 2: 3-step full-res deterministic Euler refinement.
        sched2 = LTXProjectedEulerScheduler(stage2_sigmas)
        s2_steps = sched2.schedule(
            SamplingConfig(num_inference_steps=len(stage2_sigmas) - 1),
            device=dev,
            dtype=torch.bfloat16,
        )
        v_state2, a_state2, v_shape2, a_shape2 = self._create_stage_states(
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            noise_scale=float(s2_steps[0].timestep.item()),
            generator=gen,
            device=dev,
            dtype=torch.bfloat16,
            initial_video_latent=upsampled["video"],
            initial_audio_latent=upsampled["audio"],
        )

        _cuda_sync(dev)
        t_s2_start = time.perf_counter()
        for step in s2_steps:
            v_mod = self._build_modality(v_state2, sigma=step.timestep, context=pos_v_ctx, uniform_denoise_mask=True)
            a_mod = self._build_modality(a_state2, sigma=step.timestep, context=pos_a_ctx, uniform_denoise_mask=True)
            vx, ax = executor.forward_velocity(
                v_mod,
                a_mod,
                step_index=step.index,
                total_steps=len(s2_steps),
                stage_index=1,
                branch="positive",
            )
            assert vx is not None and ax is not None
            v_pred = velocity_to_denoised(v_mod.latent, vx, v_mod.timesteps)
            a_pred = velocity_to_denoised(a_mod.latent, ax, a_mod.timesteps)
            v_state2 = v_state2.with_updates(latent=sched2.step_state(v_pred, step, v_state2, generator=step_gen_s2))
            a_state2 = a_state2.with_updates(latent=sched2.step_state(a_pred, step, a_state2, generator=step_gen_s2))
        _cuda_sync(dev)
        t_s2 = time.perf_counter() - t_s2_start
        executor.finalize_request_caches()

        video_latent = VideoLatentPatchifier(1).unpatchify(v_state2.latent, v_shape2)
        audio_latent = AudioPatchifier(1).unpatchify(a_state2.latent, a_shape2)

        _cuda_sync(dev)
        t_vae_start = time.perf_counter()
        video_frames, audio_waveform, audio_sr = self._decode_video_and_audio(
            video_latent,
            audio_latent,
            height=height,
            width=width,
            num_frames=num_frames,
            decode_audio=decode_audio,
            generator=gen,
        )
        _cuda_sync(dev)
        t_vae = time.perf_counter() - t_vae_start

        tx_report = executor.acceleration_report()
        report = {
            "model_variant": "distilled",
            "video_vae_variant": self._video_decoder_kind,
            "transformer": tx_report,
            "video_vae_decoder": dict(self._video_vae_accel_report or {}),
            "zero_fallbacks_verified": bool(tx_report.get("zero_fallbacks_verified", False)),
        }
        self._last_acceleration_report = report

        return {
            "video": video_frames,
            "video_latent": video_latent,
            "audio": audio_waveform,
            "audio_sampling_rate": audio_sr,
            "timings": {
                "text_encode_s": round(t_text, 4),
                "stage1_dit_s": round(t_s1, 4),
                "upsampler_s": round(t_up, 4),
                "stage2_dit_s": round(t_s2, 4),
                "dit_total_s": round(t_s1 + t_s2, 4),
                "vae_decode_s": round(t_vae, 4),
                "e2e_s": round(t_s1 + t_up + t_s2 + t_vae, 4),
                "e2e_with_text_s": round(t_text + t_s1 + t_up + t_s2 + t_vae, 4),
            },
            "acceleration_report": report,
        }

    @_torch_inference_mode()
    def refine_video_3step(
        self,
        draft_video: torch.Tensor,
        *,
        prompt: str,
        target_height: int,
        target_width: int,
        target_num_frames: int | None = None,
        fps: float = 24.0,
        sigmas: tuple[float, ...] = LTX25_STAGE_2_SIGMAS,
        seed: int = 42,
        upscale_mode: Literal["bicubic", "bilinear"] = "bicubic",
        prompt_embeds: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        decode_audio: bool = False,
        offload_text_encoder: bool = True,
        sp_group: Any = None,
        sp_rank: int = 0,
    ) -> dict[str, Any]:
        """High-resolution 3-step distilled LTX-2.5 Video-to-Video refinement.

        Takes a low-resolution draft video (e.g. from Stage-1 4-step MiniMax-H3),
        upscales & encodes it into LTX-2.5 high-res latent space, injects noise at
        ``sigmas[0]``, runs ``len(sigmas) - 1`` distilled Euler refinement steps
        through the accelerated LTX-2.5 22B transformer, and decodes crisp high-res
        RGB video frames via the accelerated LTX-2.5 VideoVAE.
        """
        from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
        from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
        )
        from worldfoundry.base_models.diffusion_model.schedulers.ltx import LTXProjectedEulerScheduler
        from worldfoundry.core.nn.transformer.dit import velocity_to_denoised

        if len(sigmas) < 2:
            raise ValueError("refinement sigmas must contain at least 2 entries")
        src_frames = int(_normalize_video_bcthw(draft_video).shape[2])
        resolved_frames = _align_ltx25_num_frames(src_frames if target_num_frames is None else int(target_num_frames))
        if target_height <= 0 or target_width <= 0 or target_height % 32 or target_width % 32:
            raise ValueError(f"target height/width must be multiples of 32, got {target_width}x{target_height}")

        dev = torch.device(self.device)
        t_text = 0.0
        if prompt_embeds is None:
            _cuda_sync(dev)
            t0 = time.perf_counter()
            cond = self.encode_prompt(prompt, negative_prompt=None, offload_after=offload_text_encoder)
            _cuda_sync(dev)
            t_text = time.perf_counter() - t0
        else:
            cond = {
                branch: {k: v.to(device=dev, dtype=torch.bfloat16) for k, v in mapping.items()}
                for branch, mapping in prompt_embeds.items()
            }

        encoder = self.load_video_encoder()
        executor = self.load_transformer_executor(sp_group=sp_group, sp_rank=sp_rank)
        executor.begin_request()

        # 1. Upscale & align draft frames into [-1, 1] BCTHW.
        _cuda_sync(dev)
        t_up_start = time.perf_counter()
        pixels_hi = resample_video_for_ltx25(
            draft_video.to(device=dev),
            target_frames=resolved_frames,
            target_height=target_height,
            target_width=target_width,
            mode=upscale_mode,
        ).to(dtype=torch.bfloat16)
        _cuda_sync(dev)
        t_up = time.perf_counter() - t_up_start

        # 2. Encode into LTX-2.5 high-res latent space (auto temporal tiling for >5s clips to bound VRAM).
        _cuda_sync(dev)
        t_enc_start = time.perf_counter()
        inner_enc = getattr(encoder, "encoder", None)
        if resolved_frames > 129 and inner_enc is not None and hasattr(inner_enc, "tiled_encode"):
            from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import (
                TemporalTilingConfig,
                TilingConfig,
            )

            enc_tiling = TilingConfig(
                spatial_config=None,
                temporal_config=TemporalTilingConfig(tile_size_in_frames=128, tile_overlap_in_frames=24),
            )
            draft_latent = inner_enc.tiled_encode(pixels_hi, tiling_config=enc_tiling)
        else:
            draft_latent = encoder(pixels_hi)
        _cuda_sync(dev)
        t_enc = time.perf_counter() - t_enc_start

        # 3. Noise-inject & run 3-step distilled LTX-2.5 refinement.
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        sched = LTXProjectedEulerScheduler(sigmas)
        steps = sched.schedule(SamplingConfig(num_inference_steps=len(sigmas) - 1), device=dev, dtype=torch.bfloat16)
        v_state, a_state, v_shape, a_shape = self._create_stage_states(
            height=target_height,
            width=target_width,
            num_frames=resolved_frames,
            fps=fps,
            noise_scale=float(steps[0].timestep.item()),
            generator=gen,
            device=dev,
            dtype=torch.bfloat16,
            initial_video_latent=draft_latent,
            initial_audio_latent=None,
        )

        pos_v_ctx = cond["positive"]["video_context"]
        pos_a_ctx = cond["positive"]["audio_context"]

        _cuda_sync(dev)
        t_dit_start = time.perf_counter()
        for step in steps:
            v_mod = self._build_modality(v_state, sigma=step.timestep, context=pos_v_ctx, uniform_denoise_mask=True)
            a_mod = self._build_modality(a_state, sigma=step.timestep, context=pos_a_ctx, uniform_denoise_mask=True)
            vx, ax = executor.forward_velocity(
                v_mod,
                a_mod,
                step_index=step.index,
                total_steps=len(steps),
                stage_index=1,
                branch="positive",
            )
            assert vx is not None and ax is not None
            v_pred = velocity_to_denoised(v_mod.latent, vx, v_mod.timesteps)
            a_pred = velocity_to_denoised(a_mod.latent, ax, a_mod.timesteps)
            v_state = v_state.with_updates(latent=sched.step_state(v_pred, step, v_state, generator=gen))
            a_state = a_state.with_updates(latent=sched.step_state(a_pred, step, a_state, generator=gen))
        _cuda_sync(dev)
        t_dit = time.perf_counter() - t_dit_start
        executor.finalize_request_caches()

        video_latent = VideoLatentPatchifier(1).unpatchify(v_state.latent, v_shape)
        audio_latent = AudioPatchifier(1).unpatchify(a_state.latent, a_shape)

        # 4. Decode crisp high-res video via accelerated LTX-2.5 VideoVAE.
        _cuda_sync(dev)
        t_dec_start = time.perf_counter()
        refined_frames = self.decode_video_latent(
            video_latent,
            height=target_height,
            width=target_width,
            num_frames=resolved_frames,
            generator=gen,
        )
        audio_waveform = None
        audio_sr = 48000
        if decode_audio:
            audio_waveform, audio_sr = self.decode_audio_latent(audio_latent)
        _cuda_sync(dev)
        t_dec = time.perf_counter() - t_dec_start

        tx_report = executor.acceleration_report()
        report = {
            "model_variant": self.model_variant,
            "mode": "v2v_3step_refinement",
            "video_vae_variant": self._video_decoder_kind,
            "transformer": tx_report,
            "video_vae_encoder": dict(self._video_encoder_accel_report or {}),
            "video_vae_decoder": dict(self._video_vae_accel_report or {}),
            "zero_fallbacks_verified": bool(tx_report.get("zero_fallbacks_verified", False)),
        }
        self._last_acceleration_report = report

        return {
            "video": refined_frames,
            "video_latent": video_latent,
            "audio": audio_waveform,
            "audio_sampling_rate": audio_sr,
            "resolved_frames": resolved_frames,
            "sigmas": list(sigmas),
            "timings": {
                "text_encode_s": round(t_text, 4),
                "upscale_s": round(t_up, 4),
                "vae_encode_s": round(t_enc, 4),
                "dit_refine_s": round(t_dit, 4),
                "dit_ms_per_step": round(1000.0 * t_dit / len(steps), 2),
                "vae_decode_s": round(t_dec, 4),
                "total_s": round(t_up + t_enc + t_dit + t_dec, 4),
                "total_with_text_s": round(t_text + t_up + t_enc + t_dit + t_dec, 4),
            },
            "acceleration_report": report,
        }

    def run_plan(self, plan: LTX25RuntimePlan, *, timeout_seconds: float = 7200) -> dict[str, Any]:
        """Run the native (or SOL-Engine accelerated) recipe and write its artifact in-process."""
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
        from worldfoundry.core.media.codecs.pyav import save_audio_video_h264

        started = time.monotonic()
        deadline = started + timeout_seconds
        generated: list[str] = []
        error = None
        accel_report: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        try:
            check_deadline(deadline, stage="LTX-2.5 component loading")
            values = plan.request
            if (
                self.acceleration is not None
                and not values.get("images")
                and values.get("auto_duration") is None
                and not values.get("enhance_prompt")
                and int(values.get("num_generated_keyframes") or 0) == 0
            ):
                if self.model_variant == "dev":
                    out = self.generate_dev(
                        prompt=values["prompt"],
                        negative_prompt=str(values.get("negative_prompt") or ""),
                        height=values["height"],
                        width=values["width"],
                        num_frames=values["num_frames"] or 65,
                        fps=values["frame_rate"],
                        num_inference_steps=int(values.get("num_inference_steps") or 30),
                        guidance_scale=float(
                            values["guidance_scale"] if values.get("guidance_scale") is not None else 3.0
                        ),
                        seed=values["seed"],
                    )
                else:
                    out = self.generate_distilled(
                        prompt=values["prompt"],
                        height=values["height"],
                        width=values["width"],
                        num_frames=values["num_frames"] or 121,
                        fps=values["frame_rate"],
                        seed=values["seed"],
                    )
                accel_report = out["acceleration_report"]
                timings = out["timings"]
                check_deadline(deadline, stage="LTX-2.5 artifact encoding")
                save_audio_video_h264(
                    out["video"],
                    plan.output_path,
                    fps=values["frame_rate"],
                    waveform=out["audio"],
                    sample_rate=int(out["audio_sampling_rate"]),
                    deadline_monotonic=deadline,
                )
                generated = [plan.output_path]
            else:
                pipeline = self._native_pipeline(plan)
                output = pipeline(
                    DiffusionRequest(
                        prompt=values["prompt"],
                        height=values["height"],
                        width=values["width"],
                        num_frames=values["num_frames"] or 1,
                        sampling=SamplingConfig(num_inference_steps=11, seed=values["seed"], guidance_scale=1.0),
                        inputs={
                            "frame_rate": values["frame_rate"],
                            "images": values["images"],
                            "auto_duration": values["auto_duration"],
                            "enhance_prompt": values["enhance_prompt"],
                            "num_generated_keyframes": values["num_generated_keyframes"],
                            "deadline_monotonic": deadline,
                        },
                    )
                )
                check_deadline(deadline, stage="LTX-2.5 artifact encoding")
                save_audio_video_h264(
                    output.sample,
                    plan.output_path,
                    fps=values["frame_rate"],
                    waveform=output.artifacts["audio"],
                    sample_rate=int(output.artifacts["audio_sampling_rate"]),
                    deadline_monotonic=deadline,
                )
                generated = [plan.output_path]
        except TimeoutError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        payload: dict[str, Any] = {
            "ok": error is None,
            "status": "success" if error is None else "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_files": generated,
            "error": error,
            "runtime_plan": plan.to_dict(),
        }
        if timings is not None:
            payload["timings"] = timings
        if accel_report is not None:
            payload["acceleration_report"] = accel_report
        return payload


__all__ = [
    "CHECKPOINT_REVISION",
    "CONV_VIDEO_VAE_RELPATH",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_COMPONENTS",
    "DEV_TRANSFORMER_RELPATH",
    "DIFF_VIDEO_VAE_RELPATH",
    "LTX25_STAGE_1_SIGMAS",
    "LTX25_STAGE_2_SIGMAS",
    "UPSTREAM_REVISION",
    "LTX25DistilledRuntime",
    "LTX25RuntimePlan",
    "resample_video_for_ltx25",
]
