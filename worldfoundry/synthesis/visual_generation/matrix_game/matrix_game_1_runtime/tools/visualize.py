from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from diffusers.utils import export_to_video
except Exception:  # pragma: no cover - fallback for minimal runtimes
    export_to_video = None


def parse_config(config: tuple[Any, Any]) -> tuple[dict[int, dict[str, bool]], dict[int, tuple[float, float]]]:
    key, mouse = config
    key_data: dict[int, dict[str, bool]] = {}
    mouse_data: dict[int, tuple[float, float]] = {}
    for index in range(len(mouse)):
        key_row = key[index]
        if len(key_row) == 7:
            w, s, a, d, space, attack, _ = key_row
        else:
            w, s, a, d, space, attack = key_row
        mouse_y, mouse_x = mouse[index]
        mouse_y = -1 * mouse_y
        key_data[index] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
            "Space": bool(space),
            "Attack": bool(attack),
        }
        if index == 0:
            mouse_data[index] = (320, 176)
        else:
            mouse_data[index] = (
                mouse_data[index - 1][0] + mouse_x * 15 * 0.2,
                mouse_data[index - 1][1] + mouse_y * 15 * 4 * 0.2,
            )
    return key_data, mouse_data


def draw_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int = 10,
    alpha: float = 0.5,
) -> None:
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_keys_on_frame(
    frame: np.ndarray,
    keys: dict[str, bool],
    *,
    key_size: tuple[int, int] = (80, 50),
    spacing: int = 20,
    bottom_margin: int = 30,
) -> None:
    height, width, _ = frame.shape
    horizontal_shift = 90
    vertical_shift = -20
    all_shift = 50
    positions = {
        "W": (
            width // 2 - key_size[0] // 2 - horizontal_shift - all_shift + spacing * 2,
            height - bottom_margin - key_size[1] * 2 + vertical_shift - 20,
        ),
        "A": (
            width // 2 - key_size[0] * 2 + 5 - horizontal_shift - all_shift + spacing * 2,
            height - bottom_margin - key_size[1] + vertical_shift,
        ),
        "S": (
            width // 2 - key_size[0] // 2 - horizontal_shift - all_shift + spacing * 2,
            height - bottom_margin - key_size[1] + vertical_shift,
        ),
        "D": (
            width // 2 + key_size[0] - 5 - horizontal_shift - all_shift + spacing * 2,
            height - bottom_margin - key_size[1] + vertical_shift,
        ),
        "Space": (
            width // 2 + key_size[0] * 2 + spacing * 4 - horizontal_shift - all_shift,
            height - bottom_margin - key_size[1] + vertical_shift,
        ),
        "Attack": (
            width // 2 + key_size[0] * 3 + spacing * 9 - horizontal_shift - all_shift,
            height - bottom_margin - key_size[1] + vertical_shift,
        ),
    }
    for key_name, (x, y) in positions.items():
        wide = key_name in {"Space", "Attack"}
        box_width = key_size[0] + 40 if wide else key_size[0]
        color = (0, 255, 0) if keys.get(key_name, False) else (200, 200, 200)
        draw_rounded_rectangle(
            frame,
            (x, y),
            (x + box_width, y + key_size[1]),
            color,
            radius=10,
            alpha=0.8 if keys.get(key_name, False) else 0.5,
        )
        text_size = cv2.getTextSize(key_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x + (box_width - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def overlay_icon(
    frame: np.ndarray,
    icon: np.ndarray | None,
    position: tuple[int, int],
    *,
    scale: float = 1.0,
    rotation: float = 0,
) -> None:
    if icon is None:
        return
    if icon.ndim == 3 and icon.shape[2] == 3:
        alpha_channel = np.full(icon.shape[:2] + (1,), 255, dtype=icon.dtype)  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        icon = np.concatenate([icon, alpha_channel], axis=2)

    x, y = position
    icon_h, icon_w, _ = icon.shape
    scaled_width = max(1, int(icon_w * scale))
    scaled_height = max(1, int(icon_h * scale))
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    center = (scaled_width // 2, scaled_height // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(
        icon_resized,
        matrix,
        (scaled_width, scaled_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    icon_h, icon_w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape
    top_left_x = max(0, int(x - icon_w // 2))
    top_left_y = max(0, int(y - icon_h // 2))
    bottom_right_x = min(frame_w, int(x + icon_w // 2))
    bottom_right_y = min(frame_h, int(y + icon_h // 2))
    if bottom_right_x <= top_left_x or bottom_right_y <= top_left_y:
        return

    icon_x_start = max(0, int(-x + icon_w // 2))
    icon_y_start = max(0, int(-y + icon_h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)
    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    alpha = icon_region[:, :, 3] / 255.0
    icon_rgb = icon_region[:, :, :3]
    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    for channel in range(3):
        frame_region[:, :, channel] = (1 - alpha) * frame_region[:, :, channel] + alpha * icon_rgb[:, :, channel]
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame_region


def _write_video(frames: list[np.ndarray], output_video: str, fps: int) -> None:
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    if export_to_video is not None:
        export_to_video([frame.astype(np.float32) / 255.0 for frame in frames], output_video, fps=fps)
        return
    import imageio.v3 as iio

    iio.imwrite(output_video, np.stack(frames).astype(np.uint8), fps=fps)


def process_video(
    input_video: np.ndarray,
    output_video: str,
    config: tuple[Any, Any],
    mouse_icon_path: str,
    mouse_scale: float = 2.0,
    mouse_rotation: float = 0,
    fps: int = 16,
) -> None:
    key_data, mouse_data = parse_config(config)
    frame_width = input_video[0].shape[1]
    frame_height = input_video[0].shape[0]
    mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)
    out_video: list[np.ndarray] = []
    for frame_idx, raw_frame in enumerate(input_video):
        frame = np.array(raw_frame, copy=True)
        keys = key_data.get(
            frame_idx, {"W": False, "A": False, "S": False, "D": False, "Space": False, "Attack": False}
        )
        raw_mouse_pos = mouse_data.get(frame_idx, (frame_width // 4, frame_height // 4))
        mouse_position = (int(raw_mouse_pos[0] * 2), int(raw_mouse_pos[1] * 2))
        draw_keys_on_frame(frame, keys, key_size=(75, 75), spacing=10, bottom_margin=20)
        overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
        out_video.append(frame)
    _write_video(out_video, output_video, fps)


def save_video(input_video: np.ndarray, output_video: str, fps: int = 16) -> None:
    _write_video([np.array(frame, copy=True) for frame in input_video], output_video, fps)
