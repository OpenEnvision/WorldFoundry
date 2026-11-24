from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _video_to_uint8_frames(input_video: Any) -> np.ndarray:
    frames = np.asarray(input_video)
    if frames.ndim != 4:
        raise ValueError(f"expected video with shape [T,H,W,C] or [T,C,H,W], got {frames.shape}")
    if frames.shape[-1] not in {1, 3, 4} and frames.shape[1] in {1, 3, 4}:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating) and frames.max(initial=0) <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return np.ascontiguousarray(frames)


def _frame_condition(value: Any, index: int) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    array = np.asarray(value)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    idx = min(index, array.shape[0] - 1)
    return np.asarray(array[idx]).reshape(-1).astype(np.float32, copy=False)


def _condition_label(keyboard: np.ndarray, mouse: np.ndarray) -> str:
    labels: list[str] = []
    key_names = ("W", "S", "A", "D", "Q", "E")
    for idx, value in enumerate(keyboard[: len(key_names)]):
        if abs(float(value)) > 1e-4:
            labels.append(key_names[idx])
    if mouse.size >= 2:
        dx, dy = float(mouse[0]), float(mouse[1])
        if abs(dx) > 1e-4:
            labels.append("LOOK R" if dx > 0 else "LOOK L")
        if abs(dy) > 1e-4:
            labels.append("LOOK D" if dy > 0 else "LOOK U")
    return " + ".join(labels[:4])


def _load_mouse_icon(
    mouse_icon_path: str | os.PathLike[str] | None, frame_height: int, mouse_scale: float
) -> Image.Image | None:
    if not mouse_icon_path:
        return None
    path = Path(mouse_icon_path)
    if not path.is_file():
        return None
    try:
        icon = Image.open(path).convert("RGBA")
    except Exception:
        return None
    target_height = max(20, int(frame_height * 0.06 * max(float(mouse_scale or 1.0), 0.1)))
    ratio = target_height / max(icon.height, 1)
    target_width = max(20, int(icon.width * ratio))
    return icon.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _draw_overlay(
    frame: np.ndarray,
    *,
    label: str,
    mouse_icon: Image.Image | None,
) -> np.ndarray:
    if not label and mouse_icon is None:
        return frame
    image = Image.fromarray(frame, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin = max(12, int(min(width, height) * 0.02))

    if label:
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        box = (
            margin,
            height - margin - text_h - 14,
            margin + text_w + 18,
            height - margin,
        )
        draw.rounded_rectangle(box, radius=6, fill=(0, 0, 0, 145))
        draw.text((box[0] + 9, box[1] + 7), label, fill=(255, 255, 255, 235), font=font)

    if mouse_icon is not None:
        x = width - margin - mouse_icon.width
        y = height - margin - mouse_icon.height
        image.alpha_composite(mouse_icon, (x, y))

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def process_video(
    input_video: Any,
    output_video: str | os.PathLike[str],
    config: Any = None,
    mouse_icon_path: str | os.PathLike[str] | None = None,
    mouse_scale: float = 1.0,
    mouse_rotation: float = 0,
    default_frame_res: tuple[int, int] = (704, 1280),
) -> None:
    """Write Matrix-Game-3 inference frames to mp4 with optional action overlay."""

    del mouse_rotation, default_frame_res
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _video_to_uint8_frames(input_video)

    keyboard_condition = None
    mouse_condition = None
    if isinstance(config, (tuple, list)) and len(config) >= 2:
        keyboard_condition, mouse_condition = config[0], config[1]

    icon = _load_mouse_icon(mouse_icon_path, frames.shape[1], mouse_scale)
    rendered: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        label = _condition_label(
            _frame_condition(keyboard_condition, idx),
            _frame_condition(mouse_condition, idx),
        )
        rendered.append(_draw_overlay(frame, label=label, mouse_icon=icon))

    fps = int(os.getenv("WORLDFOUNDRY_MATRIX_GAME3_FPS", "17") or "17")
    imageio.mimsave(str(output_path), rendered, fps=max(fps, 1), macro_block_size=None)
