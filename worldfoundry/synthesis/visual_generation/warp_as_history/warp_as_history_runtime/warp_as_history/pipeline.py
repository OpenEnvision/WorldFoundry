from __future__ import annotations

import math
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file, save_file

from helios.diffusers_version.pipeline_helios_diffusers import (
    XLA_AVAILABLE,
    HeliosPipelineOutput,
    HeliosPipeline,
    calculate_shift,
    optimized_scale,
)

from .camera_warp import (
    CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
    CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
    CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
    CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY,
    CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
    CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH,
    CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
    CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
    CAMERA_CONTROL_PROMPT_TRIGGER,
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
)
from .defaults import (
    LORA_DISABLED_VALUES,
    WAH_HISTORY_SIZES,
    WAH_INVISIBLE_FILL_MODES,
    WAH_NEGATIVE_PROMPT,
    WAH_NUM_FRAMES,
    WAH_NUM_LATENT_FRAMES_PER_CHUNK,
    WAH_PREV_CHUNK_HISTORY_SIZES,
    WAH_PROMPT_TRIGGER,
    WAH_PYRAMID_NUM_STAGES,
    WAH_PYRAMID_STEPS,
    WAH_VISIBLE_TOKEN_THRESHOLD,
)


_PEFT_TORCHAO_COMPAT_LOCK = threading.RLock()


if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


@dataclass
class WarpAsHistoryPipelineOutput(HeliosPipelineOutput):
    """Warp-as-History output with optional warp debug tensors."""

    warp_debug: dict[str, Any] | None = None


def _normalize_optional_lora_path(lora_path: str | Path | None) -> str | None:
    if lora_path is None:
        return None
    lora_path_str = str(lora_path)
    if lora_path_str.strip().lower() in LORA_DISABLED_VALUES:
        return None
    return lora_path_str


def _diffusers_lora_source(lora_path: str | Path) -> tuple[str, str | None]:
    """Return the directory and optional weight name expected by Diffusers.

    Recent Diffusers releases reject a direct local file while offline unless
    ``weight_name`` is explicit.  Warp-as-History accepts both released LoRA
    files and adapter directories, so normalize the file case here.
    """

    path = Path(lora_path).expanduser().resolve()
    if path.is_file():
        return str(path.parent), path.name
    return str(path), None


@contextmanager
def _ignore_incompatible_optional_torchao():
    """Keep ordinary PEFT LoRA loading independent of an old optional torchao.

    PEFT probes torchao for every target layer.  Versions below its supported
    floor raise instead of returning ``False``, even when the base layer is not
    torchao-quantized.  Scope the compatibility override to adapter loading and
    restore PEFT immediately afterwards.
    """

    with _PEFT_TORCHAO_COMPAT_LOCK:
        module = None
        original = None
        try:
            try:
                from peft.tuners.lora import torchao as peft_torchao
            except (ImportError, ModuleNotFoundError):
                yield
                return

            module = peft_torchao
            original = module.is_torchao_available
            try:
                original()
            except ImportError:
                module.is_torchao_available = lambda: False
            yield
        finally:
            if module is not None and original is not None:
                module.is_torchao_available = original


def _optional_to_dtype(tensor: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(dtype=dtype)


def _display_boundary_frame(frame: torch.Tensor) -> torch.Tensor:
    frame01 = (frame.detach().float().cpu() / 2.0 + 0.5).clamp(0.0, 1.0)
    frame01 = (frame01 * 255.0).round() / 255.0
    return frame01 * 2.0 - 1.0


def _normalize_history_sizes(history_sizes: list[int] | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    normalized = tuple(int(x) for x in history_sizes)
    if len(normalized) != 3 or any(x < 0 for x in normalized):
        raise ValueError(f"{name} must contain exactly three non-negative integers, got {history_sizes!r}.")
    return normalized


def _checkpoint_model_path(value: str | Path, *, label: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoints_root = Path(str((repo_root / "checkpoints").absolute()))
    path = Path(str(value)).expanduser()
    if path.is_absolute() and path.is_dir():
        return str(path)
    if not path.is_absolute():
        path = repo_root / path
    path = Path(str(path.absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"{label} must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}")
    return str(path)


class WarpAsHistoryPipeline(HeliosPipeline):
    """Minimal Warp-as-History inference pipeline.

    Pass `warp_video` when geometry preprocessing was done externally. If
    `warp_video` is omitted, pass `camera_poses` and the pipeline will estimate
    first-frame Pi3X geometry and render the warp condition internally.
    """

    _wah_adapter_name = "wah"
    _camera_warp_renderer: Pi3XWarpRenderer | None = None
    _camera_warp_renderer_key: tuple[Any, ...] | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any):
        model_path = _checkpoint_model_path(
            pretrained_model_name_or_path,
            label="pretrained_model_name_or_path",
        )
        return super().from_pretrained(model_path, *args, **kwargs)

    @staticmethod
    def _add_prompt_trigger(
        prompt: str | list[str] | None,
        trigger: str | None = WAH_PROMPT_TRIGGER,
    ) -> str | list[str] | None:
        trigger_text = "" if trigger is None else str(trigger).strip()
        if not trigger_text:
            return prompt

        def add_one(text: str) -> str:
            stripped = text.lstrip()
            if stripped.startswith(trigger_text):
                return text
            return f"{trigger_text} {text}"

        if prompt is None:
            return None
        if isinstance(prompt, list):
            return [add_one(item) for item in prompt]
        return add_one(prompt)

    @staticmethod
    def _coerce_visibility_mask(mask: Any) -> torch.Tensor | None:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            tensor = mask.detach()
        elif isinstance(mask, np.ndarray):
            tensor = torch.from_numpy(mask)
        elif isinstance(mask, (list, tuple)):
            if len(mask) == 0:
                raise ValueError("warp_visibility_mask cannot be an empty list.")
            if all(torch.is_tensor(item) for item in mask):
                tensor = torch.stack([item.detach() for item in mask], dim=0)
            else:
                tensor = torch.from_numpy(np.stack([np.asarray(item) for item in mask], axis=0))
        else:
            tensor = torch.from_numpy(np.asarray(mask))

        tensor = tensor.float()
        if tensor.ndim == 5:
            if tensor.shape[1] in {1, 3, 4}:
                pass
            elif tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(0, 4, 1, 2, 3)
            else:
                raise ValueError(
                    "warp_visibility_mask must be [B,1,T,H,W], [B,T,H,W,1], or a compatible mask tensor."
                )
        elif tensor.ndim == 4:
            if tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"warp_visibility_mask must be 2D-5D, got shape {tuple(tensor.shape)}.")

        if tensor.shape[1] != 1:
            tensor = tensor[:, :3].mean(dim=1, keepdim=True)
        if tensor.numel() and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _resize_visibility_mask(
        mask: torch.Tensor,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.float32)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1, -1)
        if mask.shape[0] != batch_size:
            raise ValueError(f"warp_visibility_mask batch size must be 1 or {batch_size}, got {mask.shape[0]}.")
        if mask.shape[2:] != (num_frames, height, width):
            mask = F.interpolate(mask, size=(num_frames, height, width), mode="trilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _coerce_warp_video_tensor(warp_video: Any, height: int, width: int, device: torch.device) -> torch.Tensor:
        if isinstance(warp_video, (list, tuple)):
            if len(warp_video) == 0:
                raise ValueError("warp_video cannot be an empty list.")
            frames = []
            for frame in warp_video:
                if isinstance(frame, Image.Image):
                    image = frame.convert("RGB")
                else:
                    array = np.asarray(frame)
                    if array.ndim == 2:
                        array = np.repeat(array[..., None], 3, axis=-1)
                    array = array[..., :3]
                    if array.dtype != np.uint8:
                        array = array.astype(np.float32, copy=False)
                        if array.size and float(np.nanmax(array)) <= 1.5:
                            array = array * 255.0
                        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
                    image = Image.fromarray(array).convert("RGB")
                if image.size != (int(width), int(height)):
                    image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
                frames.append(np.asarray(image, dtype=np.uint8))
            video = torch.from_numpy(np.stack(frames, axis=0)).float() / 255.0
            video = video.permute(0, 3, 1, 2)
            video = F.interpolate(video, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            tensor = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            return (tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)
        elif isinstance(warp_video, np.ndarray):
            tensor = torch.from_numpy(warp_video)
        elif torch.is_tensor(warp_video):
            tensor = warp_video.detach()
        else:
            raise ValueError(
                "warp_video must be a tensor, numpy array, or a list of PIL/numpy/torch image frames."
            )

        tensor = tensor.float()
        if tensor.ndim == 5:
            if tensor.shape[1] in {1, 3, 4}:  # [B, C, T, H, W]
                pass
            elif tensor.shape[2] in {1, 3, 4}:  # [B, T, C, H, W]
                tensor = tensor.permute(0, 2, 1, 3, 4)
            elif tensor.shape[-1] in {1, 3, 4}:  # [B, T, H, W, C]
                tensor = tensor.permute(0, 4, 1, 2, 3)
            else:
                raise ValueError(f"Unsupported 5D warp_video shape: {tuple(tensor.shape)}.")
        elif tensor.ndim == 4:
            if tensor.shape[0] in {1, 3, 4}:  # [C, T, H, W]
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[1] in {1, 3, 4}:  # [T, C, H, W]
                tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
            elif tensor.shape[-1] in {1, 3, 4}:  # [T, H, W, C]
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported 4D warp_video shape: {tuple(tensor.shape)}.")
        else:
            raise ValueError(f"warp_video must be 4D or 5D after conversion, got {tuple(tensor.shape)}.")

        if tensor.shape[1] == 1:
            tensor = tensor.expand(-1, 3, -1, -1, -1)
        elif tensor.shape[1] > 3:
            tensor = tensor[:, :3]

        if tensor.numel():
            if float(tensor.max()) > 2.0:
                tensor = tensor / 255.0
            if float(tensor.min()) >= 0.0:
                tensor = tensor * 2.0 - 1.0

        if tensor.shape[-2:] != (int(height), int(width)):
            batch_size, channels, num_frames, old_height, old_width = tensor.shape
            tensor = tensor.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, old_height, old_width)
            tensor = F.interpolate(tensor, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            tensor = tensor.reshape(batch_size, num_frames, channels, int(height), int(width)).permute(0, 2, 1, 3, 4)

        return tensor.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)

    @staticmethod
    def _visibility_mask_to_history_latents(
        mask: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        temporal_scale: int,
    ) -> torch.Tensor:
        sample_ids = torch.arange(int(latent_frames), dtype=torch.long, device=mask.device) * int(
            max(1, temporal_scale)
        )
        sample_ids = sample_ids.clamp(max=mask.shape[2] - 1)
        sampled = mask.index_select(2, sample_ids)
        sampled = F.interpolate(
            sampled,
            size=(int(latent_frames), int(latent_height), int(latent_width)),
            mode="trilinear",
            align_corners=False,
        )
        return sampled.clamp(0.0, 1.0)

    def _latent_stats(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, self.vae.dtype)
        return latents_mean, latents_std

    @staticmethod
    def _slice_frame_sequence(
        value: torch.Tensor | np.ndarray | None,
        start: int,
        frame_count: int,
        name: str,
    ) -> torch.Tensor | np.ndarray | None:
        if value is None:
            return None
        start = max(0, int(start))
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise ValueError(f"{name} frame_count must be positive, got {frame_count}.")
        if torch.is_tensor(value):
            if value.shape[0] <= 0:
                raise ValueError(f"{name} must contain at least one frame.")
            start = min(start, int(value.shape[0]) - 1)
            sliced = value.detach()[start : start + frame_count]
            if sliced.shape[0] < frame_count:
                pad = sliced[-1:].repeat(frame_count - int(sliced.shape[0]), *([1] * (sliced.ndim - 1)))
                sliced = torch.cat([sliced, pad], dim=0)
            return sliced

        array = np.asarray(value)
        if array.shape[0] <= 0:
            raise ValueError(f"{name} must contain at least one frame.")
        start = min(start, int(array.shape[0]) - 1)
        sliced = array[start : start + frame_count]
        if sliced.shape[0] < frame_count:
            pad = np.repeat(sliced[-1:], frame_count - int(sliced.shape[0]), axis=0)
            sliced = np.concatenate([sliced, pad], axis=0)
        return sliced

    @staticmethod
    def _frame_sequence_length(value: torch.Tensor | np.ndarray | None) -> int:
        if value is None:
            return 0
        return int(value.shape[0] if torch.is_tensor(value) else np.asarray(value).shape[0])

    @staticmethod
    def _last_frame_sequence(value: torch.Tensor | np.ndarray | None) -> torch.Tensor | np.ndarray | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach()[-1:].clone()
        return np.asarray(value)[-1:].copy()

    @staticmethod
    def _prepend_frame_sequence(
        prefix: torch.Tensor | np.ndarray,
        value: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        if torch.is_tensor(value):
            if torch.is_tensor(prefix):
                prefix_tensor = prefix.detach().to(device=value.device, dtype=value.dtype)
            else:
                prefix_tensor = torch.from_numpy(np.asarray(prefix)).to(device=value.device, dtype=value.dtype)
            return torch.cat([prefix_tensor, value], dim=0)

        array = np.asarray(value)
        if torch.is_tensor(prefix):
            prefix_array = prefix.detach().cpu().numpy().astype(array.dtype, copy=False)
        else:
            prefix_array = np.asarray(prefix, dtype=array.dtype)
        return np.concatenate([prefix_array, array], axis=0)

    def _trim_decoded_video(self, video: torch.Tensor) -> torch.Tensor:
        generated_frames = int(video.size(2))
        generated_frames = (
            generated_frames - 1
        ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        return video[:, :, :generated_frames]

    @staticmethod
    def _should_capture_warp_debug(state: dict[str, Any]) -> bool:
        return bool(state.get("return_warp_debug")) or state.get("warp_debug_dir") is not None

    @staticmethod
    def _detach_debug_video(video: torch.Tensor) -> torch.Tensor:
        return video.detach().float().cpu().clone()

    def _record_warp_debug_chunk(
        self,
        state: dict[str, Any],
        chunk_index: int,
        warp_video: torch.Tensor,
        *,
        drop_first_frame_for_rollout: bool,
    ) -> None:
        if not self._should_capture_warp_debug(state):
            return
        state.setdefault("warp_debug_chunks", {})[int(chunk_index)] = {
            "warp_video": self._detach_debug_video(warp_video),
            "drop_first_frame_for_rollout": bool(drop_first_frame_for_rollout),
        }

    @staticmethod
    def _warp_debug_frames_uint8(warp_video: torch.Tensor) -> list[np.ndarray]:
        if not torch.is_tensor(warp_video) or warp_video.ndim != 5:
            raise ValueError(f"warp debug video must be a [B,C,T,H,W] tensor, got {type(warp_video)!r}.")
        video01 = (warp_video[:1].detach().float().cpu() / 2.0 + 0.5).clamp(0.0, 1.0)
        if video01.shape[1] == 1:
            video01 = video01.expand(-1, 3, -1, -1, -1)
        elif video01.shape[1] > 3:
            video01 = video01[:, :3]
        frames = video01[0].permute(1, 2, 3, 0).numpy()
        return [(frame * 255.0).round().astype(np.uint8) for frame in frames]

    @staticmethod
    def _write_warp_debug_video(path: Path, warp_video: torch.Tensor, fps: int) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        frames = WarpAsHistoryPipeline._warp_debug_frames_uint8(warp_video)
        with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)

    def collect_warp_debug(
        self,
        state: dict[str, Any],
        *,
        save_dir: str | Path | None = None,
        fps: int = 16,
    ) -> dict[str, Any]:
        """Collect recorded warp conditioning frames and optionally save only warp.mp4."""
        chunks = state.get("warp_debug_chunks")
        if not chunks:
            chunks = {}
            for chunk_index, rendered in sorted(state.get("camera_warp_chunks", {}).items()):
                if isinstance(rendered, dict) and torch.is_tensor(rendered.get("warp_video")):
                    chunks[int(chunk_index)] = {
                        "warp_video": self._detach_debug_video(rendered["warp_video"]),
                        "drop_first_frame_for_rollout": int(chunk_index) > 0,
                    }
        ordered_chunks = [chunks[index] for index in sorted(chunks)]
        rollout_parts = []
        for chunk in ordered_chunks:
            warp_video = chunk["warp_video"]
            if bool(chunk.get("drop_first_frame_for_rollout")) and int(warp_video.shape[2]) > 1:
                warp_video = warp_video[:, :, 1:]
            rollout_parts.append(warp_video)
        warp_rollout = torch.cat(rollout_parts, dim=2) if rollout_parts else None

        debug = {
            "warp_video": warp_rollout,
            "chunks": chunks,
            "debug_dir": None,
        }
        if save_dir is not None:
            if warp_rollout is None:
                raise RuntimeError("No warp debug frames were recorded.")
            debug_dir = Path(save_dir).expanduser()
            self._write_warp_debug_video(debug_dir / "warp.mp4", warp_rollout, fps=int(fps))
            debug["debug_dir"] = str(debug_dir)
        return debug

    def _prepare_warp_state(
        self,
        image: PipelineImageInput,
        warp_video: Any | None,
        warp_visibility_mask: Any,
        height: int,
        width: int,
        num_frames: int,
        warp_invisible_fill: str,
        visible_token_drop: bool,
        rope_alignment: bool,
        prev_chunk_history_sizes: list[int] | tuple[int, int, int],
        add_noise_to_warp_latents: bool,
        warp_noise_sigma_min: float,
        warp_noise_sigma_max: float,
        image_history_prefix_noised: bool,
        generator: torch.Generator | list[torch.Generator] | None,
        lora_active: bool,
    ) -> dict[str, Any]:
        if warp_invisible_fill not in WAH_INVISIBLE_FILL_MODES:
            raise ValueError(f"warp_invisible_fill must be one of {sorted(WAH_INVISIBLE_FILL_MODES)}.")

        device = self._execution_device
        window_num_frames = (WAH_NUM_LATENT_FRAMES_PER_CHUNK - 1) * self.vae_scale_factor_temporal + 1
        num_warp_chunks = max(1, (max(int(num_frames), 1) + window_num_frames - 1) // window_num_frames)
        total_warp_frames = num_warp_chunks * window_num_frames
        prev_chunk_history_sizes = _normalize_history_sizes(prev_chunk_history_sizes, "prev_chunk_history_sizes")
        history_capacity = int(sum(WAH_HISTORY_SIZES))
        if sum(prev_chunk_history_sizes) > history_capacity:
            raise ValueError(
                "sum(prev_chunk_history_sizes) must be <= "
                f"{history_capacity}, got {prev_chunk_history_sizes!r}."
            )
        if any(size > cap for size, cap in zip(prev_chunk_history_sizes, WAH_HISTORY_SIZES)):
            raise ValueError(
                "prev_chunk_history_sizes cannot exceed the official WAH history slots "
                f"{WAH_HISTORY_SIZES!r}, got {prev_chunk_history_sizes!r}."
            )
        warp_noise_sigma_min = float(warp_noise_sigma_min)
        warp_noise_sigma_max = float(warp_noise_sigma_max)
        if warp_noise_sigma_min < 0.0 or warp_noise_sigma_max < warp_noise_sigma_min:
            raise ValueError(
                "warp noise sigma range must satisfy 0 <= min <= max, got "
                f"[{warp_noise_sigma_min}, {warp_noise_sigma_max}]."
            )

        source_image = self.video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=torch.float32
        )
        warp_video_tensor = None
        visibility_mask = None
        if source_image.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
        if warp_video is not None:
            warp_video_tensor = self._coerce_warp_video_tensor(warp_video, height=height, width=width, device=device)
            if warp_video_tensor.shape[0] != 1:
                raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
            if warp_video_tensor.shape[2] != total_warp_frames:
                raise ValueError(
                    "warp_video must contain exactly one full warp rollout for the requested frame count: "
                    f"{total_warp_frames} frames for num_frames={int(num_frames)} "
                    f"({num_warp_chunks} chunks x {window_num_frames} frames)."
                )

            visibility_mask = self._coerce_visibility_mask(warp_visibility_mask)
            if visibility_mask is None:
                visibility_mask = torch.ones(
                    1,
                    1,
                    total_warp_frames,
                    height,
                    width,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                visibility_mask = self._resize_visibility_mask(
                    visibility_mask,
                    batch_size=warp_video_tensor.shape[0],
                    num_frames=total_warp_frames,
                    height=height,
                    width=width,
                    device=device,
                )

        return {
            "chunk_index": 0,
            "height": int(height),
            "num_warp_chunks": num_warp_chunks,
            "prev_chunk_history_sizes": prev_chunk_history_sizes,
            "prev_history_latent_window": None,
            "prev_chunk_last_frame": None,
            "warp_history_prefix_latent": None,
            "warp_latents_tensor": None,
            "source_image": source_image,
            "visibility_mask": visibility_mask,
            "warp_invisible_fill": str(warp_invisible_fill),
            "warp_video_tensor": warp_video_tensor,
            "width": int(width),
            "window_num_frames": window_num_frames,
            "visible_token_drop": bool(visible_token_drop),
            "rope_alignment": bool(rope_alignment),
            "add_noise_to_warp_latents": bool(add_noise_to_warp_latents),
            "warp_noise_sigma_min": warp_noise_sigma_min,
            "warp_noise_sigma_max": warp_noise_sigma_max,
            "image_history_prefix_noised": bool(image_history_prefix_noised),
            "lora_active": bool(lora_active),
            "using_camera_warp": False,
        }

    def _render_camera_warp_chunk(
        self,
        state: dict[str, Any],
        chunk_index: int,
        source_frame: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rendered = state.get("camera_first_chunk_rendered") if int(chunk_index) == 0 else None
        if rendered is None:
            window_num_frames = int(state["window_num_frames"])
            online_camera_poses = state.get("online_camera_poses")
            online_target_intrinsics = state.get("online_target_intrinsics")
            if int(chunk_index) == 0:
                source_pose_index = 0
                frame_count = window_num_frames
                translation_scale = float(state["camera_control_translation_scale"])
                use_depth_scale = bool(state["camera_control_translation_scale_use_first_frame_depth"])
            else:
                source_pose_index = int(chunk_index) * window_num_frames - 1
                frame_count = window_num_frames + 1
                translation_scale = float(
                    state.get("camera_translation_effective_scale", state["camera_control_translation_scale"])
                )
                use_depth_scale = False

            if online_camera_poses is None:
                camera_poses = self._slice_frame_sequence(
                    state["camera_poses"],
                    source_pose_index,
                    frame_count,
                    "camera_poses",
                )
                target_intrinsics = self._slice_frame_sequence(
                    state.get("target_intrinsics"),
                    source_pose_index,
                    frame_count,
                    "target_intrinsics",
                )
            else:
                camera_poses = self._slice_frame_sequence(
                    online_camera_poses,
                    0,
                    frame_count,
                    "camera_poses",
                )
                target_intrinsics = self._slice_frame_sequence(
                    online_target_intrinsics,
                    0,
                    frame_count,
                    "target_intrinsics",
                )
            geometry = None
            pi3x_keyframe_images = state.get("pi3x_keyframe_images")
            if (
                int(chunk_index) > 0
                and pi3x_keyframe_images is not None
                and len(pi3x_keyframe_images) > 1
            ):
                geometry = state["camera_renderer"].estimate_keyframe_geometry(
                    image_tensors=list(pi3x_keyframe_images),
                    device=device,
                    scale_reference_geometry=state.get("camera_first_frame_geometry"),
                )
                state.setdefault("pi3x_keyframe_counts", []).append(int(geometry["keyframe_count"]))
                state.setdefault("pi3x_keyframe_scale_alignment_stats", []).append(
                    geometry.get("scale_alignment_stats", {})
                )
                state.setdefault("pi3x_keyframe_intrinsic_smoothing_stats", []).append(
                    geometry.get("intrinsic_smoothing_stats", {})
                )
            rendered = state["camera_renderer"].render(
                image_tensor=source_frame.to(device=device, dtype=torch.float32),
                camera_poses=camera_poses,
                height=int(state["height"]),
                width=int(state["width"]),
                num_frames=frame_count,
                device=device,
                geometry=geometry,
                target_intrinsics=target_intrinsics,
                chunk_index=int(chunk_index),
                translation_scale=translation_scale,
                translation_scale_use_first_frame_depth=use_depth_scale,
                invisible_fill_mode=str(state["camera_control_warp_invisible_fill"]),
                render_mode=str(state["camera_control_warp_render_mode"]),
                target_fill_radius=int(state["camera_control_warp_target_fill_radius"]),
                target_fill_min_neighbors=int(state["camera_control_warp_target_fill_min_neighbors"]),
                mesh_break_mode=str(state["camera_control_mesh_break_mode"]),
            )

        if int(chunk_index) == 0:
            if isinstance(rendered, dict) and "geometry" in rendered:
                state["camera_first_frame_geometry"] = rendered["geometry"]
            state["camera_translation_effective_scale"] = float(
                rendered.get("camera_translation_effective_scale", state["camera_control_translation_scale"])
            )
        state.setdefault("camera_warp_chunks", {})[int(chunk_index)] = rendered
        self._last_camera_warp = rendered

        warp_video = rendered["warp_video"].to(device=device, dtype=torch.float32)
        visibility_mask = rendered.get("warp_visibility_mask")
        if visibility_mask is None:
            visibility_mask = self._coerce_visibility_mask(rendered["visibility_frames"])
        visibility_mask = visibility_mask.to(device=device, dtype=torch.float32)
        self._record_warp_debug_chunk(
            state,
            int(chunk_index),
            warp_video,
            drop_first_frame_for_rollout=int(chunk_index) > 0,
        )
        return warp_video, visibility_mask

    def _check_minimal_inputs(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        num_videos_per_prompt: int,
    ) -> None:
        if num_videos_per_prompt != 1:
            raise ValueError("WarpAsHistoryPipeline requires num_videos_per_prompt == 1.")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports a single prompt.")
        if isinstance(negative_prompt, list) and len(negative_prompt) != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports a single negative prompt.")
        if prompt_embeds is not None and prompt_embeds.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports prompt_embeds batch size 1.")
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports negative_prompt_embeds batch size 1.")

    def _wah_has_loaded_adapters(self) -> bool:
        transformer = getattr(self, "transformer", None)
        peft_config = getattr(transformer, "peft_config", None)
        if isinstance(peft_config, dict) and len(peft_config) > 0:
            return True
        return getattr(self, "_wah_loaded_lora_path", None) is not None

    def _delete_wah_adapter(self) -> None:
        for module in (self, getattr(self, "transformer", None)):
            if module is not None and hasattr(module, "delete_adapters"):
                try:
                    module.delete_adapters(self._wah_adapter_name)
                    return
                except Exception:
                    pass

    def _set_wah_lora_enabled(self, enabled: bool) -> None:
        transformer = getattr(self, "transformer", None)
        if transformer is None or not self._wah_has_loaded_adapters():
            return
        if not enabled:
            self._unfuse_wah_lora()
        if enabled and hasattr(transformer, "enable_adapters"):
            transformer.enable_adapters()
        elif not enabled and hasattr(transformer, "disable_adapters"):
            transformer.disable_adapters()

    def _wah_lora_is_fused(self) -> bool:
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return False
        for module in transformer.modules():
            if bool(getattr(module, "merged", False)):
                return True
        return False

    def _fuse_wah_lora(self) -> bool:
        transformer = getattr(self, "transformer", None)
        if transformer is None or not self._wah_has_loaded_adapters():
            return False
        if self._wah_lora_is_fused():
            return True

        self._set_wah_lora_enabled(True)
        adapter_names = [self._wah_adapter_name]
        if hasattr(self, "fuse_lora"):
            try:
                self.fuse_lora(components=["transformer"], adapter_names=adapter_names)
            except TypeError:
                self.fuse_lora(adapter_names=adapter_names)
            return self._wah_lora_is_fused()
        if hasattr(transformer, "fuse_lora"):
            transformer.fuse_lora(adapter_names=adapter_names)
            return self._wah_lora_is_fused()
        return False

    def _unfuse_wah_lora(self) -> None:
        if not self._wah_lora_is_fused():
            return
        transformer = getattr(self, "transformer", None)
        if hasattr(self, "unfuse_lora"):
            try:
                self.unfuse_lora(components=["transformer"])
            except TypeError:
                self.unfuse_lora()
            return
        if transformer is not None and hasattr(transformer, "unfuse_lora"):
            transformer.unfuse_lora()

    def _configure_wah_lora(self, lora_path: str | None) -> bool:
        if lora_path is None:
            self._set_wah_lora_enabled(False)
            return False

        transformer = getattr(self, "transformer", None)
        transformer_name = transformer.__class__.__name__ if transformer is not None else ""
        if transformer_name:
            try:
                from diffusers.loaders import peft as diffusers_peft

                scale_mapping = getattr(diffusers_peft, "_SET_ADAPTER_SCALE_FN_MAPPING", None)
                if scale_mapping is not None and transformer_name not in scale_mapping:
                    scale_mapping[transformer_name] = lambda model_cls, weights: weights
            except Exception:
                pass

        adapter_path = self._materialize_wah_lora_path(lora_path, self._wah_adapter_name)
        loaded_path = getattr(self, "_wah_loaded_lora_path", None)
        if loaded_path != lora_path:
            if loaded_path is not None:
                self._delete_wah_adapter()
            transformer_config = getattr(transformer, "config", None)
            if transformer_config is not None and not hasattr(transformer_config, "image_dim"):
                if hasattr(transformer, "register_to_config"):
                    transformer.register_to_config(image_dim=None)
                else:
                    try:
                        setattr(transformer_config, "image_dim", None)
                    except Exception:
                        pass
            lora_source, weight_name = _diffusers_lora_source(adapter_path)
            load_kwargs = {"adapter_name": self._wah_adapter_name}
            if weight_name is not None:
                load_kwargs["weight_name"] = weight_name
            with _ignore_incompatible_optional_torchao():
                self.load_lora_weights(lora_source, **load_kwargs)
            self._wah_loaded_lora_path = lora_path

        if hasattr(self, "set_adapters"):
            self.set_adapters([self._wah_adapter_name], adapter_weights=[1.0])
        self._set_wah_lora_enabled(True)
        return True

    def _get_wah_runtime_lora_temp_dir(self) -> Path:
        temp_dir = getattr(self, "_wah_runtime_lora_temp_dir", None)
        if temp_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="wah_runtime_lora_")
            self._wah_runtime_lora_temp_dir = temp_dir
        return Path(temp_dir.name)

    def _materialize_wah_lora_path(self, lora_path: str | Path, adapter_name: str) -> str:
        path = Path(lora_path).expanduser().resolve()
        if path.is_dir():
            return str(path)

        safe_adapter_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in adapter_name)
        if path.suffix == ".safetensors":
            peft_state = load_file(str(path))
            if any(key.startswith("transformer.") for key in peft_state):
                return str(path)
            materialized = self._get_wah_runtime_lora_temp_dir() / f"materialized_prefixed_{safe_adapter_name}.safetensors"
            save_file(
                {f"transformer.{key}": value.detach().contiguous().cpu() for key, value in peft_state.items()},
                str(materialized),
            )
            return str(materialized)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError(f"Unsupported LoRA checkpoint type from {path}: {type(loaded)}")
        peft_state = loaded.get("peft", loaded)
        if not isinstance(peft_state, dict) or not peft_state:
            raise ValueError(f"No PEFT LoRA tensors found in {path}")
        if not all(torch.is_tensor(value) for value in peft_state.values()):
            raise TypeError(f"LoRA checkpoint {path} contains non-tensor entries in PEFT state.")
        if not any(key.startswith("transformer.") for key in peft_state):
            peft_state = {f"transformer.{key}": value for key, value in peft_state.items()}

        materialized = self._get_wah_runtime_lora_temp_dir() / f"materialized_{safe_adapter_name}.safetensors"
        save_file(
            {key: value.detach().contiguous().cpu() for key, value in peft_state.items()},
            str(materialized),
        )
        return str(materialized)

    def _get_camera_warp_renderer(
        self,
        *,
        camera_control_warp_render_mode: str,
        camera_control_warp_target_fill_radius: int,
        camera_control_warp_target_fill_min_neighbors: int,
        camera_control_mesh_break_mode: str,
        camera_control_mesh_depth_rtol: float,
        camera_control_mesh_normal_tol_deg: float,
    ) -> Pi3XWarpRenderer:
        key = (
            str(camera_control_warp_render_mode),
            int(camera_control_warp_target_fill_radius),
            int(camera_control_warp_target_fill_min_neighbors),
            str(camera_control_mesh_break_mode),
            float(camera_control_mesh_depth_rtol),
            float(camera_control_mesh_normal_tol_deg),
        )
        if self._camera_warp_renderer is None or self._camera_warp_renderer_key != key:
            config = Pi3XWarpRendererConfig(
                render_mode=str(camera_control_warp_render_mode),
                target_fill_radius=int(camera_control_warp_target_fill_radius),
                target_fill_min_neighbors=int(camera_control_warp_target_fill_min_neighbors),
                mesh_break_mode=str(camera_control_mesh_break_mode),
                mesh_depth_rtol=float(camera_control_mesh_depth_rtol),
                mesh_normal_tol_deg=float(camera_control_mesh_normal_tol_deg),
            )
            self._camera_warp_renderer = Pi3XWarpRenderer(config=config)
            self._camera_warp_renderer_key = key
        return self._camera_warp_renderer

    def _wah_execution_device(self) -> torch.device:
        device = getattr(self, "_execution_device", None)
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _add_noise_to_warp_history_latents(
        self,
        state: dict[str, Any],
        warp_latents: torch.Tensor,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        if not bool(state.get("add_noise_to_warp_latents", True)):
            return warp_latents
        noise_sigma_min = float(state.get("warp_noise_sigma_min", 0.111))
        noise_sigma_max = float(state.get("warp_noise_sigma_max", 0.135))
        rand_generator = generator[0] if isinstance(generator, list) else generator
        noisy_chunks = []
        chunk_size = int(WAH_NUM_LATENT_FRAMES_PER_CHUNK)
        for chunk_start in range(0, int(warp_latents.shape[2]), chunk_size):
            latent_chunk = warp_latents[:, :, chunk_start : chunk_start + chunk_size]
            chunk_frames = int(latent_chunk.shape[2])
            frame_sigmas = (
                torch.rand(chunk_frames, device=device, generator=rand_generator)
                * (noise_sigma_max - noise_sigma_min)
                + noise_sigma_min
            ).to(dtype=latent_chunk.dtype)
            frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
            noisy_chunks.append(
                frame_sigmas
                * randn_tensor(latent_chunk.shape, generator=generator, device=device, dtype=latent_chunk.dtype)
                + (1 - frame_sigmas) * latent_chunk
            )
        return torch.cat(noisy_chunks, dim=2)

    def _prepare_external_warp_latents(
        self,
        state: dict[str, Any],
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        cached = state.get("warp_latents_tensor")
        if cached is not None:
            return cached.to(device=device, dtype=torch.float32)
        warp_video_tensor = state["warp_video_tensor"]
        if warp_video_tensor is None:
            raise RuntimeError("warp_as_history rollout is missing external warp video.")
        latents_mean, latents_std = self._latent_stats(device)
        _, warp_latents_tensor = self.prepare_video_latents(
            warp_video_tensor.to(device=device, dtype=torch.float32),
            latents_mean=latents_mean,
            latents_std=latents_std,
            num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        warp_latents_tensor = self._add_noise_to_warp_history_latents(
            state=state,
            warp_latents=warp_latents_tensor,
            device=device,
            generator=generator,
        )
        state["warp_latents_tensor"] = warp_latents_tensor.detach()
        return warp_latents_tensor

    def _build_pyramid_base_histories(
        self,
        state: dict[str, Any],
        device: torch.device,
        history_dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None,
        base_latents_history_short: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        chunk_index = int(state.get("chunk_index", 0))
        frame_start = chunk_index * int(state["window_num_frames"])
        frame_end = frame_start + int(state["window_num_frames"])
        if chunk_index == 0:
            source_frame = state["source_image"].to(device=device, dtype=torch.float32)
        else:
            source_frame = state.get("prev_chunk_last_frame")
            if source_frame is None:
                raise RuntimeError("warp_as_history rollout is missing the previous chunk boundary frame.")
            source_frame = source_frame.to(device=device, dtype=torch.float32)

        online_conditioning_type = state.get("online_conditioning_type")
        if state.get("using_camera_warp", False):
            warp_video_chunk, visibility_chunk = self._render_camera_warp_chunk(
                state=state,
                chunk_index=chunk_index,
                source_frame=source_frame,
                device=device,
            )
        elif online_conditioning_type == "warp":
            warp_video_chunk = state.get("online_warp_video_tensor")
            visibility_chunk = state.get("online_visibility_mask")
            if warp_video_chunk is None or visibility_chunk is None:
                raise RuntimeError("warp_as_history rollout is missing online warp conditioning.")
            warp_video_chunk = warp_video_chunk.to(device=device, dtype=torch.float32)
            visibility_chunk = visibility_chunk.to(device=device, dtype=torch.float32)
        else:
            warp_video_tensor = state["warp_video_tensor"]
            visibility_mask = state["visibility_mask"]
            if warp_video_tensor is None or visibility_mask is None:
                raise RuntimeError("warp_as_history rollout is missing external warp state.")
            visibility_mask = visibility_mask.to(device=device, dtype=torch.float32)
            if frame_end > warp_video_tensor.shape[2]:
                raise RuntimeError(
                    "warp_as_history rollout is missing warp frames for chunk "
                    f"{chunk_index}: need frames [{frame_start}, {frame_end}), got {warp_video_tensor.shape[2]}."
                )
            visibility_chunk = visibility_mask[:, :, frame_start:frame_end].clone()

        if base_latents_history_short is None or base_latents_history_short.shape[2] < 1:
            raise RuntimeError("warp_as_history rollout is missing the official first-frame history prefix.")
        prefix_latent = base_latents_history_short[:, :, :1].detach().to(device=device, dtype=torch.float32)
        if bool(state.get("add_noise_to_warp_latents", True)) and not bool(
            state.get("image_history_prefix_noised", False)
        ):
            cached_prefix = state.get("warp_history_prefix_latent")
            if cached_prefix is None:
                noise_sigma_min = float(state.get("warp_noise_sigma_min", 0.111))
                noise_sigma_max = float(state.get("warp_noise_sigma_max", 0.135))
                rand_generator = generator[0] if isinstance(generator, list) else generator
                prefix_sigma = (
                    torch.rand(1, device=device, generator=rand_generator) * (noise_sigma_max - noise_sigma_min)
                    + noise_sigma_min
                ).to(dtype=prefix_latent.dtype)
                prefix_sigma = prefix_sigma.view(1, 1, 1, 1, 1)
                prefix_latent = (
                    prefix_sigma
                    * randn_tensor(prefix_latent.shape, generator=generator, device=device, dtype=prefix_latent.dtype)
                    + (1 - prefix_sigma) * prefix_latent
                )
                state["warp_history_prefix_latent"] = prefix_latent.detach()
            else:
                prefix_latent = cached_prefix.to(device=device, dtype=prefix_latent.dtype)

        if state.get("using_camera_warp", False):
            latents_mean, latents_std = self._latent_stats(device)
            _, warp_latents = self.prepare_video_latents(
                warp_video_chunk,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            warp_latents = self._add_noise_to_warp_history_latents(
                state=state,
                warp_latents=warp_latents,
                device=device,
                generator=generator,
            )
        elif online_conditioning_type == "warp":
            latents_mean, latents_std = self._latent_stats(device)
            _, warp_latents = self.prepare_video_latents(
                warp_video_chunk,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            warp_latents = self._add_noise_to_warp_history_latents(
                state=state,
                warp_latents=warp_latents,
                device=device,
                generator=generator,
            )
        else:
            warp_latents_tensor = self._prepare_external_warp_latents(
                state=state,
                device=device,
                generator=generator,
            )
            latent_start = chunk_index * WAH_NUM_LATENT_FRAMES_PER_CHUNK
            latent_end = latent_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK
            if latent_end > warp_latents_tensor.shape[2]:
                raise RuntimeError(
                    "warp_as_history rollout is missing warp latent frames for chunk "
                    f"{chunk_index}: need latents [{latent_start}, {latent_end}), got {warp_latents_tensor.shape[2]}."
                )
            warp_latents = warp_latents_tensor[:, :, latent_start:latent_end].clone()

        prefix_latent = prefix_latent.to(dtype=warp_latents.dtype)
        visibility_latents = self._visibility_mask_to_history_latents(
            visibility_chunk,
            latent_frames=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            latent_height=int(state["height"] // self.vae_scale_factor_spatial),
            latent_width=int(state["width"] // self.vae_scale_factor_spatial),
            temporal_scale=int(self.vae_scale_factor_temporal),
        )

        prev_chunk_history_sizes = tuple(int(x) for x in state.get("prev_chunk_history_sizes", (0, 0, 0)))
        total_prev_history = sum(prev_chunk_history_sizes)
        long_size, mid_size, short_size = prev_chunk_history_sizes
        official_target_start = 1 + int(sum(WAH_HISTORY_SIZES))
        hidden_start = official_target_start
        if not state["rope_alignment"]:
            hidden_start += WAH_NUM_LATENT_FRAMES_PER_CHUNK
        indices_hidden_states = torch.arange(
            hidden_start,
            hidden_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        warp_indices = torch.arange(
            official_target_start,
            official_target_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            device=device,
            dtype=torch.long,
        )

        if total_prev_history > 0:
            prev_window = warp_latents.new_zeros(
                warp_latents.shape[0],
                warp_latents.shape[1],
                total_prev_history,
                warp_latents.shape[3],
                warp_latents.shape[4],
            )
            prev_visible_window = torch.zeros(
                warp_latents.shape[0],
                1,
                total_prev_history,
                warp_latents.shape[3],
                warp_latents.shape[4],
                device=device,
                dtype=torch.float32,
            )
            prev_history_latent_window = state.get("prev_history_latent_window")
            if chunk_index > 0:
                if prev_history_latent_window is None:
                    raise RuntimeError("warp_as_history rollout is missing previous latent history.")
                prev_history_latent_window = prev_history_latent_window.to(device=device, dtype=prev_window.dtype)
                available_history = min(int(prev_history_latent_window.shape[2]), int(total_prev_history))
                if available_history > 0:
                    prev_window[:, :, -available_history:] = prev_history_latent_window[:, :, -available_history:]
                    prev_visible_window[:, :, -available_history:] = 1.0
            elif short_size > 0 and base_latents_history_short.shape[2] > 1:
                fake_count = min(int(short_size), int(base_latents_history_short.shape[2] - 1))
                fake_short = base_latents_history_short[:, :, 1 : 1 + fake_count].detach()
                fake_short = fake_short.to(device=device, dtype=prev_window.dtype)
                short_end = int(long_size + mid_size + short_size)
                short_start = short_end - fake_count
                prev_window[:, :, short_start:short_end] = fake_short[:, :, -fake_count:]
                prev_visible_window[:, :, short_start:short_end] = 1.0
            prev_long, prev_mid, prev_short = prev_window.split(prev_chunk_history_sizes, dim=2)
            prev_visible_long, prev_visible_mid, prev_visible_short = prev_visible_window.split(
                prev_chunk_history_sizes,
                dim=2,
            )
            history_start = official_target_start - total_prev_history
            prev_indices = torch.arange(
                history_start,
                history_start + total_prev_history,
                device=device,
                dtype=torch.long,
            )
            prev_long_indices, prev_mid_indices, prev_short_indices = prev_indices.split(prev_chunk_history_sizes, dim=0)
        else:
            prev_long = prev_mid = prev_short = None
            prev_visible_long = prev_visible_mid = prev_visible_short = None
            prev_long_indices = prev_mid_indices = prev_short_indices = None

        if prefix_latent.shape[0] != warp_latents.shape[0] or prefix_latent.shape[-2:] != warp_latents.shape[-2:]:
            raise RuntimeError(
                "warp_as_history first-frame prefix shape does not match warp history latents: "
                f"prefix={tuple(prefix_latent.shape)}, warp={tuple(warp_latents.shape)}."
            )

        short_indices_parts = [torch.zeros(1, device=device, dtype=torch.long)]
        short_history_parts = [prefix_latent]
        short_visible_parts = [] if state["visible_token_drop"] else None
        if short_visible_parts is not None:
            short_visible_parts.append(
                torch.ones(
                    warp_latents.shape[0],
                    1,
                    1,
                    warp_latents.shape[3],
                    warp_latents.shape[4],
                    device=device,
                    dtype=torch.float32,
                )
            )
        if short_size > 0:
            short_indices_parts.append(prev_short_indices)
            short_history_parts.append(prev_short)
            if short_visible_parts is not None:
                short_visible_parts.append(prev_visible_short)
        short_indices_parts.append(warp_indices)
        short_history_parts.append(warp_latents)
        if short_visible_parts is not None:
            short_visible_parts.append(visibility_latents)

        return {
            "indices_hidden_states": indices_hidden_states,
            "indices_latents_history_short": torch.cat(short_indices_parts, dim=0).unsqueeze(0),
            "indices_latents_history_mid": prev_mid_indices.unsqueeze(0) if mid_size > 0 else None,
            "indices_latents_history_long": prev_long_indices.unsqueeze(0) if long_size > 0 else None,
            "latents_history_short": torch.cat(short_history_parts, dim=2),
            "latents_history_mid": prev_mid if mid_size > 0 else None,
            "latents_history_long": prev_long if long_size > 0 else None,
            "history_visible_mask_short": None
            if short_visible_parts is None
            else torch.cat(short_visible_parts, dim=2),
            "history_visible_mask_mid": None
            if not state["visible_token_drop"] or mid_size <= 0
            else prev_visible_mid,
            "history_visible_mask_long": None
            if not state["visible_token_drop"] or long_size <= 0
            else prev_visible_long,
        }

    @torch.no_grad()
    def init_autoregressive_state(
        self,
        prompt: str | list[str] | None,
        image: PipelineImageInput,
        *,
        conditioning_type: str = "camera",
        lora_path: str | Path | None = None,
        visible_token_drop: bool = True,
        rope_alignment: bool = True,
        warp_invisible_fill: str = "mean_first_frame",
        height: int = 384,
        width: int = 640,
        num_frames: int = WAH_NUM_FRAMES,
        negative_prompt: str | list[str] | None = WAH_NEGATIVE_PROMPT,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        num_videos_per_prompt: int = 1,
        add_noise_to_image_latents: bool = False,
        add_noise_to_warp_latents: bool = True,
        warp_noise_sigma_min: float = 0.111,
        warp_noise_sigma_max: float = 0.135,
        is_amplify_first_chunk: bool = True,
        lora_prompt_trigger: str | None = None,
        prev_chunk_history_sizes: list[int] | tuple[int, int, int] = WAH_PREV_CHUNK_HISTORY_SIZES,
        camera_control_translation_scale: float = CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
        camera_control_translation_scale_use_first_frame_depth: bool = (
            CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH
        ),
        camera_control_warp_invisible_fill: str = CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
        camera_control_warp_render_mode: str = CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
        camera_control_warp_target_fill_radius: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
        camera_control_warp_target_fill_min_neighbors: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
        camera_control_mesh_break_mode: str = CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
        camera_control_mesh_depth_rtol: float = CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
        camera_control_mesh_normal_tol_deg: float = CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
        camera_control_pi3x_keyframe_memory: bool = CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY,
        return_warp_debug: bool = False,
        warp_debug_dir: str | Path | None = None,
        warp_debug_fps: int = 16,
    ) -> dict[str, Any]:
        conditioning_type = str(conditioning_type).strip().lower()
        if conditioning_type not in {"camera", "warp"}:
            raise ValueError("conditioning_type must be either 'camera' or 'warp'.")
        using_camera_warp = conditioning_type == "camera"
        if using_camera_warp:
            image = center_crop_resize_first_frame(image, height=int(height), width=int(width))
            warp_invisible_fill = str(camera_control_warp_invisible_fill)

        if lora_prompt_trigger is None:
            lora_prompt_trigger = CAMERA_CONTROL_PROMPT_TRIGGER if using_camera_warp else WAH_PROMPT_TRIGGER

        self._check_minimal_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        normalized_lora_path = _normalize_optional_lora_path(lora_path)
        lora_active = self._configure_wah_lora(normalized_lora_path)
        prompt_for_pipe = (
            self._add_prompt_trigger(prompt, lora_prompt_trigger) if lora_active and prompt_embeds is None else prompt
        )
        attention_kwargs = (
            {"history_visible_token_threshold": WAH_VISIBLE_TOKEN_THRESHOLD} if bool(visible_token_drop) else None
        )

        self._guidance_scale = 1.0
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        self.check_inputs(
            prompt_for_pipe,
            negative_prompt,
            int(height),
            int(width),
            prompt_embeds,
            negative_prompt_embeds,
            ["latents"],
            image,
            None,
            False,
            num_videos_per_prompt,
            [7, 7, 7],
            3,
            1.0,
        )

        device = self._execution_device
        vae_dtype = self.vae.dtype
        latents_mean, latents_std = self._latent_stats(device)

        if prompt_for_pipe is not None and isinstance(prompt_for_pipe, str):
            batch_size = 1
        elif prompt_for_pipe is not None and isinstance(prompt_for_pipe, list):
            batch_size = len(prompt_for_pipe)
        else:
            batch_size = int(prompt_embeds.shape[0])

        all_prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt_for_pipe,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=512,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        image_tensor = self.video_processor.preprocess(image, height=int(height), width=int(width))
        image_latents, fake_image_latents = self.prepare_image_latents(
            image_tensor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        first_frame_image_latents = image_latents

        if bool(add_noise_to_image_latents):
            image_noise_sigma = torch.rand(1, device=device, generator=generator) * (0.135 - 0.111) + 0.111
            image_latents = image_noise_sigma * randn_tensor(
                image_latents.shape,
                generator=generator,
                device=device,
            ) + (1 - image_noise_sigma) * image_latents
            fake_image_noise_sigma = torch.rand(1, device=device, generator=generator) * (0.135 - 0.111) + 0.111
            fake_image_latents = fake_image_noise_sigma * randn_tensor(
                fake_image_latents.shape,
                generator=generator,
                device=device,
            ) + (1 - fake_image_noise_sigma) * fake_image_latents

        history_sizes = sorted(list(WAH_HISTORY_SIZES), reverse=True)
        num_channels_latents = int(self.transformer.config.in_channels)
        window_num_frames = (WAH_NUM_LATENT_FRAMES_PER_CHUNK - 1) * self.vae_scale_factor_temporal + 1
        num_history_latent_frames = int(sum(history_sizes))
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        total_generated_latent_frames = 0
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents[:, :, :-1, :, :], fake_image_latents], dim=2)
            total_generated_latent_frames += 1

        indices = torch.arange(0, sum([1, *history_sizes, WAH_NUM_LATENT_FRAMES_PER_CHUNK]))
        (
            indices_prefix,
            indices_latents_history_long,
            indices_latents_history_mid,
            indices_latents_history_1x,
            indices_hidden_states,
        ) = indices.split([1, *history_sizes, WAH_NUM_LATENT_FRAMES_PER_CHUNK], dim=0)
        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

        state = self._prepare_warp_state(
            image=image,
            warp_video=None,
            warp_visibility_mask=None,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            warp_invisible_fill=str(warp_invisible_fill),
            visible_token_drop=bool(visible_token_drop),
            rope_alignment=bool(rope_alignment),
            prev_chunk_history_sizes=prev_chunk_history_sizes,
            add_noise_to_warp_latents=bool(add_noise_to_warp_latents),
            warp_noise_sigma_min=float(warp_noise_sigma_min),
            warp_noise_sigma_max=float(warp_noise_sigma_max),
            image_history_prefix_noised=bool(add_noise_to_image_latents),
            generator=generator,
            lora_active=bool(lora_active),
        )
        state.update(
            {
                "conditioning_type": conditioning_type,
                "online_conditioning_type": None,
                "generator": generator,
                "prompt_embeds": all_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "attention_kwargs": attention_kwargs,
                "batch_size": batch_size,
                "num_channels_latents": num_channels_latents,
                "history_sizes": history_sizes,
                "num_history_latent_frames": num_history_latent_frames,
                "history_latents": history_latents,
                "total_generated_latent_frames": total_generated_latent_frames,
                "history_video": None,
                "returned_frame_count": 0,
                "real_history_latents": None,
                "image_latents": image_latents,
                "first_frame_image_latents": first_frame_image_latents,
                "latents_mean": latents_mean,
                "latents_std": latents_std,
                "vae_dtype": vae_dtype,
                "transformer_dtype": transformer_dtype,
                "indices_hidden_states": indices_hidden_states.unsqueeze(0),
                "indices_latents_history_short": indices_latents_history_short.unsqueeze(0),
                "indices_latents_history_mid": indices_latents_history_mid.unsqueeze(0),
                "indices_latents_history_long": indices_latents_history_long.unsqueeze(0),
                "output_type": output_type,
                "keep_first_frame": True,
                "pyramid_num_stages": WAH_PYRAMID_NUM_STAGES,
                "pyramid_num_inference_steps_list": list(WAH_PYRAMID_STEPS),
                "guidance_scale": 1.0,
                "use_zero_init": False,
                "zero_steps": 0,
                "is_amplify_first_chunk": bool(is_amplify_first_chunk),
                "lora_active": bool(lora_active),
                "last_camera_pose": None,
                "last_target_intrinsic": None,
                "return_warp_debug": bool(return_warp_debug),
                "warp_debug_dir": str(Path(warp_debug_dir).expanduser()) if warp_debug_dir is not None else None,
                "warp_debug_fps": int(warp_debug_fps),
                "warp_debug_chunks": {},
            }
        )

        if using_camera_warp:
            renderer = self._get_camera_warp_renderer(
                camera_control_warp_render_mode=camera_control_warp_render_mode,
                camera_control_warp_target_fill_radius=camera_control_warp_target_fill_radius,
                camera_control_warp_target_fill_min_neighbors=camera_control_warp_target_fill_min_neighbors,
                camera_control_mesh_break_mode=camera_control_mesh_break_mode,
                camera_control_mesh_depth_rtol=float(camera_control_mesh_depth_rtol),
                camera_control_mesh_normal_tol_deg=float(camera_control_mesh_normal_tol_deg),
            )
            source_image_tensor = image_tensor.to(device=device, dtype=torch.float32)
            state.update(
                {
                    "using_camera_warp": True,
                    "camera_renderer": renderer,
                    "camera_control_translation_scale": float(camera_control_translation_scale),
                    "camera_translation_effective_scale": float(camera_control_translation_scale),
                    "camera_control_translation_scale_use_first_frame_depth": bool(
                        camera_control_translation_scale_use_first_frame_depth
                    ),
                    "camera_control_warp_invisible_fill": str(camera_control_warp_invisible_fill),
                    "camera_control_warp_render_mode": str(camera_control_warp_render_mode),
                    "camera_control_warp_target_fill_radius": int(camera_control_warp_target_fill_radius),
                    "camera_control_warp_target_fill_min_neighbors": int(
                        camera_control_warp_target_fill_min_neighbors
                    ),
                    "camera_control_mesh_break_mode": str(camera_control_mesh_break_mode),
                    "pi3x_keyframe_images": [source_image_tensor.detach().float().cpu()]
                    if bool(camera_control_pi3x_keyframe_memory)
                    else None,
                    "pi3x_keyframe_last_decoded_chunk": -1,
                    "pi3x_keyframe_memory_enabled": bool(camera_control_pi3x_keyframe_memory),
                    "pi3x_keyframe_counts": [],
                    "pi3x_keyframe_intrinsic_smoothing_stats": [],
                    "pi3x_keyframe_scale_alignment_stats": [],
                    "camera_warp_chunks": {},
                }
            )
        return state

    def _prepare_autoregressive_camera_chunk(
        self,
        state: dict[str, Any],
        camera_poses: torch.Tensor | np.ndarray,
        target_intrinsics: torch.Tensor | np.ndarray | None,
    ) -> None:
        if state.get("conditioning_type") != "camera":
            raise ValueError("This autoregressive state was initialized for warp_video chunks, not camera chunks.")
        chunk_index = int(state.get("chunk_index", 0))
        window_num_frames = int(state["window_num_frames"])
        if chunk_index == 0:
            prepared_camera_poses = self._slice_frame_sequence(
                camera_poses,
                0,
                window_num_frames,
                "camera_poses",
            )
        else:
            frame_count = window_num_frames + 1
            if self._frame_sequence_length(camera_poses) == frame_count:
                prepared_camera_poses = self._slice_frame_sequence(camera_poses, 0, frame_count, "camera_poses")
            else:
                target_camera_poses = self._slice_frame_sequence(
                    camera_poses,
                    0,
                    window_num_frames,
                    "camera_poses",
                )
                last_camera_pose = state.get("last_camera_pose")
                if last_camera_pose is None:
                    raise RuntimeError("autoregressive camera state is missing the previous boundary camera pose.")
                prepared_camera_poses = self._prepend_frame_sequence(last_camera_pose, target_camera_poses)

        prepared_target_intrinsics = None
        if target_intrinsics is not None:
            if chunk_index == 0:
                prepared_target_intrinsics = self._slice_frame_sequence(
                    target_intrinsics,
                    0,
                    window_num_frames,
                    "target_intrinsics",
                )
            else:
                frame_count = window_num_frames + 1
                if self._frame_sequence_length(target_intrinsics) == frame_count:
                    prepared_target_intrinsics = self._slice_frame_sequence(
                        target_intrinsics,
                        0,
                        frame_count,
                        "target_intrinsics",
                    )
                else:
                    target_intrinsics_chunk = self._slice_frame_sequence(
                        target_intrinsics,
                        0,
                        window_num_frames,
                        "target_intrinsics",
                    )
                    last_target_intrinsic = state.get("last_target_intrinsic")
                    if last_target_intrinsic is None:
                        raise RuntimeError(
                            "autoregressive target_intrinsics after the first chunk must either include the "
                            "previous boundary intrinsic or be provided from the first chunk so it can be cached."
                        )
                    prepared_target_intrinsics = self._prepend_frame_sequence(
                        last_target_intrinsic,
                        target_intrinsics_chunk,
                    )

        state["online_conditioning_type"] = "camera"
        state["using_camera_warp"] = True
        state["online_camera_poses"] = prepared_camera_poses
        state["online_target_intrinsics"] = prepared_target_intrinsics
        state["_pending_last_camera_pose"] = self._last_frame_sequence(prepared_camera_poses)
        state["_pending_last_target_intrinsic"] = self._last_frame_sequence(prepared_target_intrinsics)

    def _prepare_autoregressive_warp_chunk(
        self,
        state: dict[str, Any],
        warp_video: Any,
        warp_visibility_mask: Any | None,
    ) -> None:
        if state.get("conditioning_type") != "warp":
            raise ValueError("This autoregressive state was initialized for camera chunks, not warp_video chunks.")
        device = self._wah_execution_device()
        height = int(state["height"])
        width = int(state["width"])
        window_num_frames = int(state["window_num_frames"])
        warp_video_tensor = self._coerce_warp_video_tensor(
            warp_video,
            height=height,
            width=width,
            device=device,
        )
        if warp_video_tensor.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
        if int(warp_video_tensor.shape[2]) != window_num_frames:
            raise ValueError(
                "autoregressive warp_video chunks must contain exactly "
                f"{window_num_frames} frames, got {int(warp_video_tensor.shape[2])}."
            )

        visibility_mask = self._coerce_visibility_mask(warp_visibility_mask)
        if visibility_mask is None:
            visibility_mask = torch.ones(
                1,
                1,
                window_num_frames,
                height,
                width,
                device=device,
                dtype=torch.float32,
            )
        else:
            visibility_mask = self._resize_visibility_mask(
                visibility_mask,
                batch_size=warp_video_tensor.shape[0],
                num_frames=window_num_frames,
                height=height,
                width=width,
                device=device,
            )

        state["online_conditioning_type"] = "warp"
        state["using_camera_warp"] = False
        state["online_warp_video_tensor"] = warp_video_tensor
        state["online_visibility_mask"] = visibility_mask
        self._record_warp_debug_chunk(
            state,
            int(state.get("chunk_index", 0)),
            warp_video_tensor,
            drop_first_frame_for_rollout=False,
        )

    def _record_decoded_chunk_boundary(self, state: dict[str, Any], decoded_video: torch.Tensor) -> None:
        if not isinstance(decoded_video, torch.Tensor) or decoded_video.ndim != 5 or decoded_video.shape[2] < 1:
            return
        boundary_frame = _display_boundary_frame(decoded_video[:, :, -1])
        state["prev_chunk_last_frame"] = boundary_frame
        pi3x_keyframe_images = state.get("pi3x_keyframe_images")
        if pi3x_keyframe_images is not None:
            decoded_chunk_index = int(state.get("chunk_index", 0)) - 1
            last_decoded_chunk = int(state.get("pi3x_keyframe_last_decoded_chunk", -1))
            if decoded_chunk_index >= 0 and decoded_chunk_index > last_decoded_chunk:
                pi3x_keyframe_images.append(boundary_frame)
                state["pi3x_keyframe_last_decoded_chunk"] = decoded_chunk_index

    def _commit_autoregressive_conditioning(self, state: dict[str, Any]) -> None:
        if state.get("online_conditioning_type") != "camera":
            return
        state["last_camera_pose"] = state.get("_pending_last_camera_pose")
        state["last_target_intrinsic"] = state.get("_pending_last_target_intrinsic")

    def _generate_next_chunk_from_state(self, state: dict[str, Any]) -> torch.Tensor:
        device = self._execution_device
        self._guidance_scale = float(state.get("guidance_scale", 1.0))
        self._attention_kwargs = state.get("attention_kwargs")
        self._interrupt = False

        history_sizes = list(state["history_sizes"])
        num_history_latent_frames = int(state["num_history_latent_frames"])
        history_latents = state["history_latents"]
        latents_history_long, latents_history_mid, latents_history_1x = history_latents[
            :, :, -num_history_latent_frames:
        ].split(history_sizes, dim=2)

        chunk_index = int(state.get("chunk_index", 0))
        is_first_chunk = chunk_index == 0
        batch_size = int(state["batch_size"])
        num_channels_latents = int(state["num_channels_latents"])
        height = int(state["height"])
        width = int(state["width"])
        window_num_frames = int(state["window_num_frames"])

        image_latents = state.get("image_latents")
        if image_latents is None and is_first_chunk:
            latents_prefix = torch.zeros(
                (
                    batch_size,
                    num_channels_latents,
                    1,
                    latents_history_1x.shape[-2],
                    latents_history_1x.shape[-1],
                ),
                device=device,
                dtype=latents_history_1x.dtype,
            )
        else:
            latents_prefix = image_latents
        latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            window_num_frames,
            dtype=torch.float32,
            device=device,
            generator=state.get("generator"),
            latents=None,
        )

        pyramid_steps = list(state["pyramid_num_inference_steps_list"])
        num_inference_steps = (
            sum(pyramid_steps) * 2
            if bool(state.get("is_amplify_first_chunk", False)) and self.config.is_distilled and is_first_chunk
            else sum(pyramid_steps)
        )

        previous_wah_state = getattr(self, "_wah_state", None)
        self._wah_state = state
        try:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                latents = self.stage2_sample(
                    latents=latents,
                    pyramid_num_stages=int(state["pyramid_num_stages"]),
                    pyramid_num_inference_steps_list=pyramid_steps,
                    prompt_embeds=state["prompt_embeds"],
                    negative_prompt_embeds=state.get("negative_prompt_embeds"),
                    guidance_scale=float(state.get("guidance_scale", 1.0)),
                    indices_hidden_states=state["indices_hidden_states"],
                    indices_latents_history_short=state["indices_latents_history_short"],
                    indices_latents_history_mid=state["indices_latents_history_mid"],
                    indices_latents_history_long=state["indices_latents_history_long"],
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    attention_kwargs=state.get("attention_kwargs"),
                    device=device,
                    transformer_dtype=state["transformer_dtype"],
                    generator=state.get("generator"),
                    use_zero_init=bool(state.get("use_zero_init", False)),
                    zero_steps=int(state.get("zero_steps", 0)),
                    is_amplify_first_chunk=bool(state.get("is_amplify_first_chunk", False)) and is_first_chunk,
                    progress_bar=progress_bar,
                )
        finally:
            self._wah_state = previous_wah_state

        first_frame_image_latents = state.get("first_frame_image_latents")
        if bool(state.get("keep_first_frame", True)) and is_first_chunk and first_frame_image_latents is not None:
            latents[:, :, 0:1, :, :] = first_frame_image_latents.to(device=latents.device, dtype=latents.dtype)

        state["total_generated_latent_frames"] = int(state["total_generated_latent_frames"]) + int(latents.shape[2])
        state["history_latents"] = torch.cat([state["history_latents"], latents], dim=2)
        real_history_latents = state["history_latents"][:, :, -int(state["total_generated_latent_frames"]) :]
        state["real_history_latents"] = real_history_latents
        state["last_latents"] = latents

        vae_dtype = self.vae.dtype
        latents_mean = state["latents_mean"].to(device=device, dtype=vae_dtype)
        latents_std = state["latents_std"].to(device=device, dtype=vae_dtype)
        current_latents = (
            real_history_latents[:, :, -WAH_NUM_LATENT_FRAMES_PER_CHUNK:].to(vae_dtype) / latents_std
            + latents_mean
        )
        current_video = self.vae.decode(current_latents, return_dict=False)[0]
        self._record_decoded_chunk_boundary(state, current_video)
        self._commit_autoregressive_conditioning(state)

        history_video = state.get("history_video")
        if history_video is None:
            history_video = current_video
        else:
            history_video = torch.cat([history_video, current_video], dim=2)
        state["history_video"] = history_video

        finalized_history_video = self._trim_decoded_video(history_video)
        returned_frame_count = int(state.get("returned_frame_count", 0))
        video_delta = finalized_history_video[:, :, returned_frame_count:]
        state["returned_frame_count"] = int(finalized_history_video.shape[2])
        state["last_video_delta"] = video_delta
        return video_delta

    @torch.no_grad()
    def generate_next_chunk(
        self,
        state: dict[str, Any],
        *,
        camera_poses: torch.Tensor | np.ndarray | None = None,
        target_intrinsics: torch.Tensor | np.ndarray | None = None,
        warp_video: Any | None = None,
        warp_visibility_mask: Any | None = None,
        output_type: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate one autoregressive WAH chunk and return the finalized video delta plus next state."""
        if not isinstance(state, dict):
            raise TypeError("state must be created by init_autoregressive_state().")
        conditioning_type = state.get("conditioning_type")
        if conditioning_type == "camera":
            if camera_poses is None:
                raise ValueError("camera_poses is required for a camera autoregressive state.")
            if warp_video is not None or warp_visibility_mask is not None:
                raise ValueError("warp_video inputs cannot be passed to a camera autoregressive state.")
            self._prepare_autoregressive_camera_chunk(state, camera_poses, target_intrinsics)
        elif conditioning_type == "warp":
            if warp_video is None:
                raise ValueError("warp_video is required for a warp autoregressive state.")
            if camera_poses is not None or target_intrinsics is not None:
                raise ValueError("camera inputs cannot be passed to a warp autoregressive state.")
            self._prepare_autoregressive_warp_chunk(state, warp_video, warp_visibility_mask)
        else:
            raise ValueError("state is missing a valid conditioning_type.")

        video_delta = self._generate_next_chunk_from_state(state)
        selected_output_type = state.get("output_type") if output_type is None else output_type
        if selected_output_type == "latent":
            video = state["last_latents"]
        else:
            video = self.video_processor.postprocess_video(
                video_delta.detach().clone(),
                output_type=selected_output_type,
            )
        return video, state

    def finalize_autoregressive_state(
        self,
        state: dict[str, Any],
        *,
        output_type: str | None = None,
        return_dict: bool = True,
        free_model_hooks: bool = True,
        return_warp_debug: bool | None = None,
        warp_debug_dir: str | Path | None = None,
        warp_debug_fps: int | None = None,
    ) -> Any:
        selected_output_type = state.get("output_type") if output_type is None else output_type
        if selected_output_type == "latent":
            video = state.get("real_history_latents")
        else:
            history_video = state.get("history_video")
            if history_video is None:
                raise RuntimeError("No autoregressive chunks have been generated yet.")
            video = self.video_processor.postprocess_video(
                self._trim_decoded_video(history_video).detach().clone(),
                output_type=selected_output_type,
            )

        selected_return_warp_debug = (
            bool(state.get("return_warp_debug")) if return_warp_debug is None else bool(return_warp_debug)
        )
        selected_warp_debug_dir = state.get("warp_debug_dir") if warp_debug_dir is None else warp_debug_dir
        selected_warp_debug_fps = int(state.get("warp_debug_fps", 16) if warp_debug_fps is None else warp_debug_fps)
        warp_debug = None
        if selected_return_warp_debug or selected_warp_debug_dir is not None:
            warp_debug = self.collect_warp_debug(
                state,
                save_dir=selected_warp_debug_dir,
                fps=selected_warp_debug_fps,
            )

        self._current_timestep = None
        self._set_wah_lora_enabled(bool(state.get("lora_active", False)))
        if bool(free_model_hooks):
            self.maybe_free_model_hooks()

        if not return_dict:
            return (video, warp_debug) if selected_return_warp_debug else (video,)
        if selected_return_warp_debug:
            return WarpAsHistoryPipelineOutput(frames=video, warp_debug=warp_debug)
        return HeliosPipelineOutput(frames=video)

    def _run_original_helios(
        self,
        *,
        prompt: str | list[str] | None,
        image: PipelineImageInput | None,
        warp_visibility_mask: Any | None,
        target_intrinsics: torch.Tensor | np.ndarray | None,
        lora_path: str | Path | None,
        height: int,
        width: int,
        num_frames: int,
        negative_prompt: str | list[str] | None,
        generator: torch.Generator | list[torch.Generator] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        output_type: str | None,
        return_dict: bool,
        num_videos_per_prompt: int,
        add_noise_to_image_latents: bool,
        is_amplify_first_chunk: bool,
        return_warp_debug: bool,
        warp_debug_dir: str | Path | None,
        helios_kwargs: dict[str, Any],
    ) -> Any:
        if return_warp_debug or warp_debug_dir is not None:
            raise ValueError("Warp debug requires camera_poses or warp_video; original Helios inference has no warp.")
        if warp_visibility_mask is not None:
            raise ValueError("warp_visibility_mask requires warp_video; omit both for original Helios inference.")
        if target_intrinsics is not None:
            raise ValueError("target_intrinsics requires camera_poses; omit both for original Helios inference.")
        if _normalize_optional_lora_path(lora_path) is not None:
            raise ValueError(
                "lora_path is only supported for Warp-as-History conditioning. "
                "Pass camera_poses or warp_video to use a WAH LoRA, or omit lora_path for original Helios inference."
            )

        self._set_wah_lora_enabled(False)
        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            num_videos_per_prompt=num_videos_per_prompt,
            image=image,
            add_noise_to_image_latents=bool(add_noise_to_image_latents),
            is_amplify_first_chunk=bool(is_amplify_first_chunk),
            **helios_kwargs,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None,
        image: PipelineImageInput | None = None,
        warp_video: Any | None = None,
        *,
        warp_visibility_mask: Any | None = None,
        camera_poses: torch.Tensor | np.ndarray | None = None,
        target_intrinsics: torch.Tensor | np.ndarray | None = None,
        lora_path: str | Path | None = None,
        visible_token_drop: bool = True,
        rope_alignment: bool = True,
        warp_invisible_fill: str = "mean_first_frame",
        height: int = 384,
        width: int = 640,
        num_frames: int = WAH_NUM_FRAMES,
        negative_prompt: str | list[str] | None = WAH_NEGATIVE_PROMPT,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        num_videos_per_prompt: int = 1,
        add_noise_to_image_latents: bool = False,
        add_noise_to_warp_latents: bool = True,
        warp_noise_sigma_min: float = 0.111,
        warp_noise_sigma_max: float = 0.135,
        is_amplify_first_chunk: bool = True,
        lora_prompt_trigger: str | None = None,
        prev_chunk_history_sizes: list[int] | tuple[int, int, int] = WAH_PREV_CHUNK_HISTORY_SIZES,
        camera_control_translation_scale: float = CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
        camera_control_translation_scale_use_first_frame_depth: bool = (
            CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH
        ),
        camera_control_warp_invisible_fill: str = CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
        camera_control_warp_render_mode: str = CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
        camera_control_warp_target_fill_radius: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
        camera_control_warp_target_fill_min_neighbors: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
        camera_control_mesh_break_mode: str = CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
        camera_control_mesh_depth_rtol: float = CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
        camera_control_mesh_normal_tol_deg: float = CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
        camera_control_pi3x_keyframe_memory: bool = CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY,
        return_warp_debug: bool = False,
        warp_debug_dir: str | Path | None = None,
        warp_debug_fps: int = 16,
        **helios_kwargs: Any,
    ) -> Any:
        if warp_video is None and camera_poses is None:
            return self._run_original_helios(
                prompt=prompt,
                image=image,
                warp_visibility_mask=warp_visibility_mask,
                target_intrinsics=target_intrinsics,
                lora_path=lora_path,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                negative_prompt=negative_prompt,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                num_videos_per_prompt=num_videos_per_prompt,
                add_noise_to_image_latents=bool(add_noise_to_image_latents),
                is_amplify_first_chunk=bool(is_amplify_first_chunk),
                return_warp_debug=bool(return_warp_debug),
                warp_debug_dir=warp_debug_dir,
                helios_kwargs=helios_kwargs,
            )
        if helios_kwargs:
            unsupported = ", ".join(sorted(helios_kwargs))
            raise ValueError(
                "Original Helios arguments are only supported when neither warp_video nor camera_poses is provided: "
                f"{unsupported}."
            )
        if image is None:
            raise ValueError("image is required when using Warp-as-History conditioning.")

        using_camera_warp = warp_video is None
        if using_camera_warp:
            if camera_poses is None:
                raise ValueError("Either warp_video or camera_poses must be provided.")
            if warp_visibility_mask is not None:
                raise ValueError("warp_visibility_mask is only supported when warp_video is provided.")
        elif camera_poses is not None:
            raise ValueError("Pass either warp_video or camera_poses, not both.")

        state = self.init_autoregressive_state(
            prompt=prompt,
            image=image,
            conditioning_type="camera" if using_camera_warp else "warp",
            lora_path=lora_path,
            visible_token_drop=bool(visible_token_drop),
            rope_alignment=bool(rope_alignment),
            warp_invisible_fill=str(warp_invisible_fill),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            negative_prompt=negative_prompt,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            num_videos_per_prompt=num_videos_per_prompt,
            add_noise_to_image_latents=bool(add_noise_to_image_latents),
            add_noise_to_warp_latents=bool(add_noise_to_warp_latents),
            warp_noise_sigma_min=float(warp_noise_sigma_min),
            warp_noise_sigma_max=float(warp_noise_sigma_max),
            is_amplify_first_chunk=bool(is_amplify_first_chunk),
            lora_prompt_trigger=lora_prompt_trigger,
            prev_chunk_history_sizes=prev_chunk_history_sizes,
            camera_control_translation_scale=float(camera_control_translation_scale),
            camera_control_translation_scale_use_first_frame_depth=bool(
                camera_control_translation_scale_use_first_frame_depth
            ),
            camera_control_warp_invisible_fill=str(camera_control_warp_invisible_fill),
            camera_control_warp_render_mode=str(camera_control_warp_render_mode),
            camera_control_warp_target_fill_radius=int(camera_control_warp_target_fill_radius),
            camera_control_warp_target_fill_min_neighbors=int(camera_control_warp_target_fill_min_neighbors),
            camera_control_mesh_break_mode=str(camera_control_mesh_break_mode),
            camera_control_mesh_depth_rtol=float(camera_control_mesh_depth_rtol),
            camera_control_mesh_normal_tol_deg=float(camera_control_mesh_normal_tol_deg),
            camera_control_pi3x_keyframe_memory=bool(camera_control_pi3x_keyframe_memory),
            return_warp_debug=bool(return_warp_debug),
            warp_debug_dir=warp_debug_dir,
            warp_debug_fps=int(warp_debug_fps),
        )

        num_chunks = int(state["num_warp_chunks"])
        window_num_frames = int(state["window_num_frames"])
        warp_video_tensor = None
        visibility_mask = None
        if not using_camera_warp:
            device = self._wah_execution_device()
            total_warp_frames = num_chunks * window_num_frames
            warp_video_tensor = self._coerce_warp_video_tensor(
                warp_video,
                height=int(height),
                width=int(width),
                device=device,
            )
            if warp_video_tensor.shape[0] != 1:
                raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
            if int(warp_video_tensor.shape[2]) != total_warp_frames:
                raise ValueError(
                    "warp_video must contain exactly one full warp rollout for the requested frame count: "
                    f"{total_warp_frames} frames for num_frames={int(num_frames)} "
                    f"({num_chunks} chunks x {window_num_frames} frames)."
                )
            visibility_mask = self._coerce_visibility_mask(warp_visibility_mask)
            if visibility_mask is None:
                visibility_mask = torch.ones(
                    1,
                    1,
                    total_warp_frames,
                    int(height),
                    int(width),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                visibility_mask = self._resize_visibility_mask(
                    visibility_mask,
                    batch_size=warp_video_tensor.shape[0],
                    num_frames=total_warp_frames,
                    height=int(height),
                    width=int(width),
                    device=device,
                )

        try:
            for chunk_index in range(num_chunks):
                frame_start = chunk_index * window_num_frames
                if using_camera_warp:
                    if chunk_index == 0:
                        chunk_camera_poses = self._slice_frame_sequence(
                            camera_poses,
                            0,
                            window_num_frames,
                            "camera_poses",
                        )
                        chunk_target_intrinsics = self._slice_frame_sequence(
                            target_intrinsics,
                            0,
                            window_num_frames,
                            "target_intrinsics",
                        )
                    else:
                        chunk_camera_poses = self._slice_frame_sequence(
                            camera_poses,
                            frame_start - 1,
                            window_num_frames + 1,
                            "camera_poses",
                        )
                        chunk_target_intrinsics = self._slice_frame_sequence(
                            target_intrinsics,
                            frame_start - 1,
                            window_num_frames + 1,
                            "target_intrinsics",
                        )
                    self.generate_next_chunk(
                        state,
                        camera_poses=chunk_camera_poses,
                        target_intrinsics=chunk_target_intrinsics,
                        output_type="latent",
                    )
                else:
                    frame_end = frame_start + window_num_frames
                    self.generate_next_chunk(
                        state,
                        warp_video=warp_video_tensor[:, :, frame_start:frame_end],
                        warp_visibility_mask=visibility_mask[:, :, frame_start:frame_end],
                        output_type="latent",
                    )

            return self.finalize_autoregressive_state(
                state,
                output_type=output_type,
                return_dict=return_dict,
                free_model_hooks=True,
                return_warp_debug=bool(return_warp_debug),
                warp_debug_dir=warp_debug_dir,
                warp_debug_fps=int(warp_debug_fps),
            )
        finally:
            self._wah_state = None
            self._set_wah_lora_enabled(bool(state.get("lora_active", False)))

    def stage2_sample(
        self,
        latents: torch.Tensor = None,
        pyramid_num_stages: int = None,
        pyramid_num_inference_steps_list: list[int] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        guidance_scale: float | None = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: dict | None = None,
        device: torch.device | None = None,
        transformer_dtype: torch.dtype = None,
        generator: torch.Generator | None = None,
        use_zero_init: bool | None = True,
        zero_steps: int | None = 1,
        is_amplify_first_chunk: bool = False,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        progress_bar=None,
    ):
        state = getattr(self, "_wah_state", None)
        if state is None:
            return super().stage2_sample(
                latents=latents,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                attention_kwargs=attention_kwargs,
                device=device,
                transformer_dtype=transformer_dtype,
                generator=generator,
                use_zero_init=use_zero_init,
                zero_steps=zero_steps,
                is_amplify_first_chunk=is_amplify_first_chunk,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                progress_bar=progress_bar,
            )

        batch_size, num_channel, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_channel, height, width)
        for _ in range(pyramid_num_stages - 1):
            height //= 2
            width //= 2
            latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
        latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(0, 2, 1, 3, 4)

        batch_size = latents.shape[0]
        start_point_list = None
        if self.config.is_distilled:
            start_point_list = [latents]

        i = 0
        try:
            for i_s in range(pyramid_num_stages):
                use_wah_lora = bool(state["lora_active"]) and i_s == 0
                if use_wah_lora:
                    if not self._fuse_wah_lora():
                        self._set_wah_lora_enabled(True)
                else:
                    self._set_wah_lora_enabled(False)

                patch_size = self.transformer.config.patch_size
                image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                    patch_size[0] * patch_size[1] * patch_size[2]
                )
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 4096),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
                self.scheduler.set_timesteps(
                    pyramid_num_inference_steps_list[i_s],
                    i_s,
                    device=device,
                    mu=mu,
                    is_amplify_first_chunk=is_amplify_first_chunk,
                )
                timesteps = self.scheduler.timesteps

                if i_s > 0:
                    height *= 2
                    width *= 2
                    num_frames = latents.shape[2]
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(
                        batch_size * num_frames, num_channel, height // 2, width // 2
                    )
                    latents = F.interpolate(latents, size=(height, width), mode="nearest")
                    latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(
                        0, 2, 1, 3, 4
                    )

                    ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                    gamma = self.scheduler.config.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                    beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                    batch_size, channel, num_frames, height, width = latents.shape
                    noise = self.sample_block_noise(
                        batch_size,
                        channel,
                        num_frames,
                        height,
                        width,
                        patch_size,
                        device,
                        generator,
                    )
                    noise = noise.to(device=device, dtype=transformer_dtype)
                    latents = alpha * latents + beta * noise

                    if self.config.is_distilled:
                        start_point_list.append(latents)

                if i_s == 0:
                    pyramid_base_histories = self._build_pyramid_base_histories(
                        state=state,
                        device=device,
                        history_dtype=latents_history_short.dtype,
                        generator=generator,
                        base_latents_history_short=latents_history_short,
                    )

                for idx, t in enumerate(timesteps):
                    if i_s == 0:
                        current_indices_hidden_states = pyramid_base_histories["indices_hidden_states"]
                        current_indices_latents_history_short = pyramid_base_histories[
                            "indices_latents_history_short"
                        ]
                        current_indices_latents_history_mid = pyramid_base_histories["indices_latents_history_mid"]
                        current_indices_latents_history_long = pyramid_base_histories["indices_latents_history_long"]
                        current_latents_history_short = pyramid_base_histories["latents_history_short"]
                        current_latents_history_mid = pyramid_base_histories["latents_history_mid"]
                        current_latents_history_long = pyramid_base_histories["latents_history_long"]
                        current_history_visible_mask_short = pyramid_base_histories["history_visible_mask_short"]
                        current_history_visible_mask_mid = pyramid_base_histories["history_visible_mask_mid"]
                        current_history_visible_mask_long = pyramid_base_histories["history_visible_mask_long"]
                    else:
                        current_indices_hidden_states = indices_hidden_states
                        current_indices_latents_history_short = indices_latents_history_short
                        current_indices_latents_history_mid = indices_latents_history_mid
                        current_indices_latents_history_long = indices_latents_history_long
                        current_latents_history_short = latents_history_short
                        current_latents_history_mid = latents_history_mid
                        current_latents_history_long = latents_history_long
                        current_history_visible_mask_short = None
                        current_history_visible_mask_mid = None
                        current_history_visible_mask_long = None

                    timestep = t.expand(latents.shape[0]).to(torch.int64)
                    model_latents = latents.to(transformer_dtype)

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=model_latents,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            indices_hidden_states=current_indices_hidden_states,
                            indices_latents_history_short=current_indices_latents_history_short,
                            indices_latents_history_mid=current_indices_latents_history_mid,
                            indices_latents_history_long=current_indices_latents_history_long,
                            latents_history_short=_optional_to_dtype(
                                current_latents_history_short, transformer_dtype
                            ),
                            latents_history_mid=_optional_to_dtype(current_latents_history_mid, transformer_dtype),
                            latents_history_long=_optional_to_dtype(current_latents_history_long, transformer_dtype),
                            history_visible_mask_short=current_history_visible_mask_short,
                            history_visible_mask_mid=current_history_visible_mask_mid,
                            history_visible_mask_long=current_history_visible_mask_long,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=model_latents,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                                indices_hidden_states=current_indices_hidden_states,
                                indices_latents_history_short=current_indices_latents_history_short,
                                indices_latents_history_mid=current_indices_latents_history_mid,
                                indices_latents_history_long=current_indices_latents_history_long,
                                latents_history_short=_optional_to_dtype(
                                    current_latents_history_short, transformer_dtype
                                ),
                                latents_history_mid=_optional_to_dtype(current_latents_history_mid, transformer_dtype),
                                latents_history_long=_optional_to_dtype(
                                    current_latents_history_long, transformer_dtype
                                ),
                                history_visible_mask_short=current_history_visible_mask_short,
                                history_visible_mask_mid=current_history_visible_mask_mid,
                                history_visible_mask_long=current_history_visible_mask_long,
                            )[0]

                        if self.config.is_cfg_zero_star:
                            noise_pred_text = noise_pred
                            positive_flat = noise_pred_text.view(batch_size, -1)
                            negative_flat = noise_uncond.view(batch_size, -1)

                            alpha = optimized_scale(positive_flat, negative_flat)
                            alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                            alpha = alpha.to(noise_pred_text.dtype)

                            if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                                noise_pred = noise_pred_text * 0.0
                            else:
                                noise_pred = (
                                    noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                                )
                        else:
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        generator=generator,
                        return_dict=False,
                        cur_sampling_step=idx,
                        dmd_noisy_tensor=start_point_list[i_s] if start_point_list is not None else None,
                        dmd_sigmas=self.scheduler.sigmas,
                        dmd_timesteps=self.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if progress_bar is not None:
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

                    i += 1
        finally:
            self._unfuse_wah_lora()
            self._set_wah_lora_enabled(bool(state["lora_active"]))

        total_prev_history = sum(int(x) for x in state.get("prev_chunk_history_sizes", (0, 0, 0)))
        if total_prev_history > 0:
            prev_history = state.get("prev_history_latent_window")
            current_history = latents.detach()
            if prev_history is not None:
                prev_history = prev_history.to(device=current_history.device, dtype=current_history.dtype)
                current_history = torch.cat([prev_history, current_history], dim=2)
            state["prev_history_latent_window"] = current_history[:, :, -total_prev_history:].detach()
        state["chunk_index"] = int(state.get("chunk_index", 0)) + 1
        return latents
