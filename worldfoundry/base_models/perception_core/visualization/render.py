"""Shared renderers for official perception-model outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PALETTE = np.asarray(
    [
        (0, 209, 178),
        (255, 102, 85),
        (66, 135, 245),
        (255, 194, 51),
        (171, 108, 255),
        (38, 166, 154),
        (239, 83, 80),
        (92, 107, 192),
    ],
    dtype=np.uint8,
)


def as_rgb_uint8(image: Any) -> np.ndarray:
    """Normalize PIL/numpy/tensor-like image data to an RGB uint8 array."""

    if isinstance(image, (str, Path)):
        array = np.asarray(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected an image array, got shape {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array.astype(np.float32))
        if array.min(initial=0.0) < 0.0:
            array = (array + 1.0) * 0.5
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def render_detections(
    image: Any,
    boxes: Any,
    *,
    labels: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    normalized: bool = False,
) -> np.ndarray:
    """Draw official detector outputs expressed as xyxy boxes."""

    rgb = as_rgb_uint8(image)
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 240))
    for index, box in enumerate(box_array):
        x0, y0, x1, y1 = box.tolist()
        if normalized:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        x0, x1 = sorted((float(np.clip(x0, 0, width - 1)), float(np.clip(x1, 0, width - 1))))
        y0, y1 = sorted((float(np.clip(y0, 0, height - 1)), float(np.clip(y1, 0, height - 1))))
        color = tuple(int(value) for value in _PALETTE[index % len(_PALETTE)])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        label = labels[index] if labels is not None and index < len(labels) else f"object {index + 1}"
        if scores is not None and index < len(scores):
            label = f"{label} {float(scores[index]):.2f}"
        text_box = draw.textbbox((x0, y0), label, font=font)
        text_height = text_box[3] - text_box[1] + 4
        top = max(0, y0 - text_height)
        draw.rectangle((x0, top, max(x0 + 4, text_box[2] + 4), y0), fill=color)
        draw.text((x0 + 2, top + 2), label, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def render_masks(image: Any, masks: Any, *, alpha: float = 0.5) -> np.ndarray:
    """Overlay binary or integer-label segmentation masks on an image."""

    rgb = as_rgb_uint8(image).astype(np.float32)
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        labels = [mask_array == value for value in np.unique(mask_array) if value != 0]
    elif mask_array.ndim == 3:
        labels = [mask_array[index] > 0 for index in range(mask_array.shape[0])]
    else:
        raise ValueError(f"Expected masks shaped HxW or NxHxW, got {mask_array.shape}.")
    if any(mask.shape != rgb.shape[:2] for mask in labels):
        raise ValueError("Mask and image spatial dimensions must match.")
    output = rgb.copy()
    opacity = float(np.clip(alpha, 0.0, 1.0))
    for index, mask in enumerate(labels):
        color = _PALETTE[index % len(_PALETTE)].astype(np.float32)
        output[mask] = output[mask] * (1.0 - opacity) + color * opacity
    return np.clip(output, 0, 255).astype(np.uint8)


def render_depth(depth: Any, *, inverse: bool = False) -> np.ndarray:
    """Colorize an official monocular-depth output with robust percentile scaling."""

    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected depth shaped HxW, got {array.shape}.")
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    values = np.where(valid, array, np.nan)
    if inverse:
        values = np.divide(1.0, values, out=np.full_like(values, np.nan), where=values > 0)
    from worldfoundry.base_models.perception_core.visualization.colormaps import colorize_depth_affine

    return colorize_depth_affine(values, mask=valid, cmap="turbo")


def render_normals(normals: Any, *, mask: Any | None = None) -> np.ndarray:
    """Convert HxWx3 surface normals in [-1, 1] to the shared RGB convention."""

    array = np.asarray(normals, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected normals shaped HxWx3, got {array.shape}.")
    valid = np.asarray(np.isfinite(array).all(axis=-1))
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    from worldfoundry.base_models.perception_core.visualization.colormaps import colorize_normal

    return colorize_normal(np.nan_to_num(array), mask=valid)


def render_keypoints(
    image: Any,
    keypoints: Any,
    *,
    edges: Sequence[Sequence[int]] | None = None,
    score_threshold: float = 0.0,
    normalized: bool = False,
) -> np.ndarray:
    """Draw detector/pose keypoints expressed as Nx2 or Nx3 coordinates."""

    canvas = Image.fromarray(as_rgb_uint8(image))
    draw = ImageDraw.Draw(canvas)
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError(f"Expected keypoints shaped Nx2 or Nx3, got {points.shape}.")
    width, height = canvas.size
    xy = points[:, :2].copy()
    if normalized:
        xy *= np.asarray([width, height], dtype=np.float32)
    visible = np.isfinite(xy).all(axis=1)
    if points.shape[1] == 3:
        visible &= points[:, 2] >= score_threshold
    radius = max(2, round(min(width, height) / 180))
    for edge_index, edge in enumerate(() if edges is None else edges):
        start, end = int(edge[0]), int(edge[1])
        if start >= len(xy) or end >= len(xy) or not (visible[start] and visible[end]):
            continue
        color = tuple(int(value) for value in _PALETTE[edge_index % len(_PALETTE)])
        draw.line((tuple(xy[start]), tuple(xy[end])), fill=color, width=max(2, radius))
    for index, (x, y) in enumerate(xy):
        if not visible[index]:
            continue
        color = tuple(int(value) for value in _PALETTE[index % len(_PALETTE)])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.asarray(canvas)


def render_text_overlay(image: Any, text: str, *, position: str = "bottom") -> np.ndarray:
    """Overlay caption, tag, safety, or action-recognition text on media."""

    canvas = Image.fromarray(as_rgb_uint8(image))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    width, height = canvas.size
    margin = max(6, round(min(width, height) / 80))
    max_chars = max(16, int((width - 2 * margin) / 7))
    words = str(text).strip().split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current or not lines:
        lines.append(current)
    line_height = font.getbbox("Ag")[3] + 4
    box_height = line_height * len(lines) + 2 * margin
    top = margin if position == "top" else max(0, height - box_height)
    draw.rectangle((0, top, width, min(height, top + box_height)), fill=(0, 0, 0, 180))
    for index, line in enumerate(lines):
        draw.text((margin, top + margin + index * line_height), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(canvas)


def render_optical_flow(flow: Any) -> np.ndarray:
    """Render HxWx2 flow with the Middlebury color wheel used by official RAFT tools."""

    array = np.asarray(flow, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 2 and array.shape[-1] != 2:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"Expected optical flow shaped HxWx2, got {array.shape}.")
    from worldfoundry.base_models.perception_core.optical_flow.flowformerplusplus.core.utils.flow_viz import (
        flow_to_image,
    )

    return flow_to_image(np.nan_to_num(array))


def render_tracks(
    frames: Any,
    tracks: Any,
    *,
    visibility: Any | None = None,
    trace_length: int = 12,
) -> list[np.ndarray]:
    """Draw CoTracker/DoT-style point tracks and recent traces on RGB frames."""

    frame_list = [as_rgb_uint8(frame) for frame in np.asarray(frames)]
    track_array = np.asarray(tracks, dtype=np.float32)
    if track_array.ndim == 4 and track_array.shape[0] == 1:
        track_array = track_array[0]
    if track_array.ndim != 3 or track_array.shape[-1] != 2:
        raise ValueError(f"Expected tracks shaped TxNx2, got {track_array.shape}.")
    if len(frame_list) != track_array.shape[0]:
        raise ValueError("Track and frame counts must match.")
    visible = np.ones(track_array.shape[:2], dtype=bool) if visibility is None else np.asarray(visibility).astype(bool)
    if visible.ndim == 3 and visible.shape[-1] == 1:
        visible = visible[..., 0]
    if visible.shape != track_array.shape[:2]:
        raise ValueError("Visibility must be shaped TxN.")

    rendered: list[np.ndarray] = []
    radius = max(2, round(min(frame_list[0].shape[:2]) / 180))
    for frame_index, frame in enumerate(frame_list):
        canvas = Image.fromarray(frame)
        draw = ImageDraw.Draw(canvas)
        for track_index in range(track_array.shape[1]):
            color = tuple(int(value) for value in _PALETTE[track_index % len(_PALETTE)])
            start = max(0, frame_index - max(int(trace_length), 0))
            trace = [
                tuple(float(v) for v in track_array[index, track_index])
                for index in range(start, frame_index + 1)
                if visible[index, track_index]
            ]
            if len(trace) > 1:
                draw.line(trace, fill=color, width=max(1, radius // 2))
            if visible[frame_index, track_index]:
                x, y = track_array[frame_index, track_index]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        rendered.append(np.asarray(canvas))
    return rendered


def render_feature_pca(features: Any, *, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Project dense DINO-style patch features to an RGB PCA visualization."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        height, width, channels = array.shape
        flat = array.reshape(-1, channels)
    elif array.ndim == 2:
        token_count, channels = array.shape
        side = round(token_count**0.5)
        if side * side != token_count:
            raise ValueError("Flat feature tokens must form a square grid.")
        height = width = side
        flat = array
    else:
        raise ValueError(f"Expected dense features shaped HxWxC or NxC, got {array.shape}.")
    centered = np.nan_to_num(flat) - np.nan_to_num(flat).mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:3].T
    lo = np.percentile(projected, 1, axis=0)
    hi = np.percentile(projected, 99, axis=0)
    projected = np.clip((projected - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    image = (projected.reshape(height, width, 3) * 255.0).astype(np.uint8)
    if output_size is not None:
        image = np.asarray(Image.fromarray(image).resize(output_size, Image.Resampling.BILINEAR))
    return image


def load_json_detections(path: str | Path) -> tuple[Any, Sequence[str] | None, Sequence[float] | None]:
    """Load the common official JSON detection schemas used by GroundingDINO/YOLO."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        boxes = [item["box"] if "box" in item else item["bbox"] for item in payload]
        labels = [str(item.get("label") or item.get("phrase") or "object") for item in payload]
        scores = [float(item.get("score", item.get("confidence", 1.0))) for item in payload]
        return boxes, labels, scores
    boxes = payload.get("boxes") or payload.get("bboxes")
    if boxes is None:
        raise ValueError("Detection JSON must contain boxes/bboxes or a list of detection objects.")
    return boxes, payload.get("labels") or payload.get("phrases"), payload.get("scores")
