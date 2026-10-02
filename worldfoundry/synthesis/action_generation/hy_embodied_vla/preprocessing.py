"""WorldFoundry input adaptation for Hy-VLA's three-camera tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

_DATA_CONFIG = load_vla_va_wam_runtime_config("hy-embodied-vla")
_CAMERA_CONFIG = _DATA_CONFIG["camera_keys"]
CAMERA_KEYS = tuple(str(key) for key in _CAMERA_CONFIG)
_CAMERA_ALIASES = {
    str(key): tuple(str(alias) for alias in aliases)
    for key, aliases in _CAMERA_CONFIG.items()
}


def _frame_tensor(frame: Any):
    import torch

    from worldfoundry.core.media.processing.image_utils import load_pil_image

    if torch.is_tensor(frame):
        tensor = frame.detach()
        if tensor.ndim != 3:
            raise ValueError(f"Expected one image frame with rank 3, got {tuple(tensor.shape)}")
        if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor[:3].to(dtype=torch.float32)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"Hy-VLA RGB frame must have 1, 3, or 4 channels, got {tensor.shape[0]}")
        if tensor.numel():
            minimum, maximum = tensor.min(), tensor.max()
            if minimum < 0.0 and minimum >= -1.0 and maximum <= 1.0:
                tensor = (tensor + 1.0) * 0.5
            elif maximum > 1.0 or minimum < 0.0:
                tensor = tensor.clamp(0, 255) / 255.0
        return tensor.contiguous()

    image = load_pil_image(frame, first_sequence_item=False)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _camera_tensor(value: Any, *, use_video_encoder: bool, history_size: int):
    import torch

    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.ndim == 5:
            if tensor.shape[2] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(0, 1, 4, 2, 3)
            result = tensor
        elif tensor.ndim == 4:
            if tensor.shape[1] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(0, 3, 1, 2)
            if use_video_encoder and tensor.shape[1] in {1, 3, 4}:
                result = tensor.unsqueeze(0)
            else:
                result = tensor
        elif tensor.ndim == 3:
            result = _frame_tensor(tensor).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported Hy-VLA camera tensor shape: {tuple(tensor.shape)}")
        result = result.to(dtype=torch.float32)
        if result.numel():
            minimum, maximum = result.min(), result.max()
            if minimum < 0.0 and minimum >= -1.0 and maximum <= 1.0:
                result = (result + 1.0) * 0.5
            elif maximum > 1.0 or minimum < 0.0:
                result = result.clamp(0, 255) / 255.0
        if use_video_encoder and result.ndim == 4:
            result = result.unsqueeze(1)
    elif isinstance(value, np.ndarray) and value.ndim >= 4:
        return _camera_tensor(
            torch.from_numpy(value),
            use_video_encoder=use_video_encoder,
            history_size=history_size,
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        frames = [_frame_tensor(frame) for frame in value]
        if not frames:
            raise ValueError("Hy-VLA camera history cannot be empty")
        result = torch.stack(frames, dim=0).unsqueeze(0)
    else:
        result = _frame_tensor(value).unsqueeze(0)
        if use_video_encoder:
            result = result.unsqueeze(1)

    channel_axis = 2 if result.ndim == 5 else 1
    channels = result.shape[channel_axis]
    if channels == 1:
        repeats = [1] * result.ndim
        repeats[channel_axis] = 3
        result = result.repeat(*repeats)
    elif channels == 4:
        result = result.narrow(channel_axis, 0, 3)
    elif channels != 3:
        raise ValueError(f"Hy-VLA RGB input must have 1, 3, or 4 channels, got {channels}")

    if use_video_encoder:
        if result.ndim != 5:
            raise ValueError(f"Video Hy-VLA input must have shape (B,K,C,H,W), got {tuple(result.shape)}")
        if result.shape[0] != 1:
            raise ValueError("WorldFoundry Hy-VLA currently supports one observation per inference call")
        result = result[:, -history_size:]
        missing = history_size - result.shape[1]
        if missing > 0:
            padding = torch.zeros(
                result.shape[0],
                missing,
                *result.shape[2:],
                dtype=result.dtype,
                device=result.device,
            )
            result = torch.cat([padding, result], dim=1)
    elif result.ndim == 5:
        result = result[:, -1]
    if not use_video_encoder:
        if result.ndim != 4:
            raise ValueError(f"Image Hy-VLA input must have shape (B,C,H,W), got {tuple(result.shape)}")
        if result.shape[0] != 1:
            raise ValueError("WorldFoundry Hy-VLA currently supports one observation per inference call")
    return result.contiguous()


def _camera_mapping(images: Any, *, replicate_single_image: bool) -> dict[str, Any]:
    if isinstance(images, Mapping):
        resolved: dict[str, Any] = {}
        for official_key, aliases in _CAMERA_ALIASES.items():
            for alias in aliases:
                if alias in images and images[alias] is not None:
                    resolved[official_key] = images[alias]
                    break
        return resolved

    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        values = list(images)
        if len(values) == 3:
            return dict(zip(CAMERA_KEYS, values, strict=True))
    if images is None:
        return {}
    keys = CAMERA_KEYS if replicate_single_image else CAMERA_KEYS[:1]
    return {key: images for key in keys}


def build_model_batch(
    *,
    images: Any,
    state: Any,
    instruction: str,
    use_video_encoder: bool,
    history_size: int,
    replicate_single_image: bool,
) -> dict[str, Any]:
    """Create the exact batch consumed by ``HyVLA.forward_evaluate``."""

    import torch

    camera_values = _camera_mapping(images, replicate_single_image=replicate_single_image)
    if not camera_values:
        raise ValueError(
            "Hy-VLA requires at least one RGB view; provide a frame, three-view sequence, "
            "or mapping keyed by top_head/hand_left/hand_right"
        )
    batch = {
        key: _camera_tensor(value, use_video_encoder=use_video_encoder, history_size=history_size)
        for key, value in camera_values.items()
    }
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    if state_tensor.ndim == 1:
        state_tensor = state_tensor.unsqueeze(0)
    if state_tensor.ndim != 2 or state_tensor.shape[0] != 1:
        raise ValueError(f"Hy-VLA state must have shape (D,) or (1,D), got {tuple(state_tensor.shape)}")
    batch["observation.state"] = state_tensor
    batch["task"] = [str(instruction)]
    return batch


__all__ = ["CAMERA_KEYS", "build_model_batch"]
