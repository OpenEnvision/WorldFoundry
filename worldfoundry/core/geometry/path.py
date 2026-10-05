"""Model-neutral camera-path contracts for explorable world generation.

Model runtimes accept this JSON-compatible ``camera_path`` schema to sample
camera poses and captions without a viewer dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

CAMERA_PATH_SCHEMA_VERSION = 1
CAMERA_PATH_INTERPOLATIONS = frozenset({"linear", "smooth"})


class CameraPathError(ValueError):
    """Raised when a camera-path payload violates the explorer contract."""


def default_camera_path(*, fps: float = 16.0) -> dict[str, Any]:
    """Return a short forward camera path suitable for a new editor session."""

    return {
        "schema_version": CAMERA_PATH_SCHEMA_VERSION,
        "interpolation": "smooth",
        "loop": False,
        "fps": float(fps),
        "duration_sec": 5.0,
        "coordinate_space": "world",
        "keyframes": [
            {
                "id": "keyframe-1",
                "t": 0.0,
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "fov": 60.0,
                "prompt": "",
            },
            {
                "id": "keyframe-2",
                "t": 1.0,
                "position": [0.0, 0.0, 0.12],
                "rotation": [0.0, 0.0, 0.0],
                "fov": 60.0,
                "prompt": "",
            },
        ],
    }


def _finite_float(value: Any, *, label: str) -> float:
    """Parse a finite float; NaN/inf would make keyframe times non-monotonic without a clear error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraPathError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise CameraPathError(f"{label} must be a finite number.")
    return result


def _vector(value: Any, *, label: str, length: int = 3) -> list[float]:
    """Require a fixed-length numeric vector; strings are rejected so ``\"1,2,3\"`` cannot sneak in."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise CameraPathError(f"{label} must contain exactly {length} numbers.")
    return [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def normalize_camera_path(
    payload: Mapping[str, Any],
    *,
    min_keyframes: int = 2,
    max_keyframes: int = 128,
) -> dict[str, Any]:
    """Validate and canonicalize a JSON-compatible camera path."""

    if not isinstance(payload, Mapping):
        raise CameraPathError("camera_path must be a JSON object.")

    raw_keyframes = payload.get("keyframes")
    if not isinstance(raw_keyframes, Sequence) or isinstance(raw_keyframes, (str, bytes)):
        raise CameraPathError("camera_path.keyframes must be an array.")
    if not min_keyframes <= len(raw_keyframes) <= max_keyframes:
        raise CameraPathError(
            f"camera_path requires between {min_keyframes} and {max_keyframes} keyframes."
        )

    interpolation = str(payload.get("interpolation") or "smooth").strip().lower()
    if interpolation not in CAMERA_PATH_INTERPOLATIONS:
        raise CameraPathError(
            "camera_path.interpolation must be one of "
            + ", ".join(sorted(CAMERA_PATH_INTERPOLATIONS))
            + "."
        )
    coordinate_space = str(payload.get("coordinate_space") or "world").strip().lower()
    if coordinate_space not in {"local", "world"}:
        raise CameraPathError("camera_path.coordinate_space must be 'local' or 'world'.")

    fps = _finite_float(payload.get("fps", 16.0), label="camera_path.fps")
    duration_sec = _finite_float(
        payload.get("duration_sec", max(len(raw_keyframes) - 1, 1)),
        label="camera_path.duration_sec",
    )
    if fps <= 0:
        raise CameraPathError("camera_path.fps must be positive.")
    if duration_sec <= 0:
        raise CameraPathError("camera_path.duration_sec must be positive.")

    keyframes: list[dict[str, Any]] = []
    previous_t = -math.inf
    for index, raw_keyframe in enumerate(raw_keyframes):
        if not isinstance(raw_keyframe, Mapping):
            raise CameraPathError(f"camera_path.keyframes[{index}] must be an object.")
        t = _finite_float(raw_keyframe.get("t", index), label=f"keyframes[{index}].t")
        if t <= previous_t:
            raise CameraPathError("camera-path keyframe times must be strictly increasing.")
        previous_t = t
        fov = _finite_float(raw_keyframe.get("fov", 60.0), label=f"keyframes[{index}].fov")
        if not 1.0 <= fov < 179.0:
            raise CameraPathError("camera-path field of view must be in [1, 179) degrees.")
        keyframes.append(
            {
                "id": str(raw_keyframe.get("id") or f"keyframe-{index + 1}"),
                "t": t,
                "position": _vector(
                    raw_keyframe.get("position", (0.0, 0.0, 0.0)),
                    label=f"keyframes[{index}].position",
                ),
                "rotation": _vector(
                    raw_keyframe.get("rotation", (0.0, 0.0, 0.0)),
                    label=f"keyframes[{index}].rotation",
                ),
                "fov": fov,
                "prompt": str(
                    raw_keyframe.get("prompt")
                    or raw_keyframe.get("region_hint")
                    or ""
                ).strip(),
            }
        )

    first_t = keyframes[0]["t"]
    span = keyframes[-1]["t"] - first_t
    if span <= 0:
        raise CameraPathError("camera-path keyframes must span a positive duration.")
    for keyframe in keyframes:
        keyframe["t"] = (keyframe["t"] - first_t) / span

    normalized: dict[str, Any] = {
        "schema_version": CAMERA_PATH_SCHEMA_VERSION,
        "interpolation": interpolation,
        "loop": bool(payload.get("loop", False)),
        "fps": fps,
        "duration_sec": duration_sec,
        "coordinate_space": coordinate_space,
        "keyframes": keyframes,
    }
    raw_frame_count = payload.get("frame_count")
    if raw_frame_count is not None:
        try:
            frame_count = int(raw_frame_count)
        except (TypeError, ValueError) as exc:
            raise CameraPathError("camera_path.frame_count must be an integer.") from exc
        if frame_count < 2:
            raise CameraPathError("camera_path.frame_count must be at least 2.")
        normalized["frame_count"] = frame_count
    return normalized


def _smoothstep(values: Any) -> Any:
    """Hermite smoothstep; eases segment alpha so linear lerp does not look piecewise-linear."""
    return values * values * (3.0 - 2.0 * values)


def sample_camera_path(
    payload: Mapping[str, Any],
    *,
    frame_stride: int | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Sample a normalized path into world-to-camera matrices and zoom factors.

    ``frame_stride`` is intended for autoregressive runtimes that require a
    fixed number of frames between authored keyframes.  When supplied, the
    result contains ``1 + stride * (keyframe_count - 1)`` frames.
    """

    import numpy as np

    path = normalize_camera_path(payload)
    keyframes = path["keyframes"]
    if frame_stride is not None:
        if int(frame_stride) < 1:
            raise CameraPathError("frame_stride must be positive.")
        sampled_frame_count = 1 + int(frame_stride) * (len(keyframes) - 1)
    elif frame_count is not None:
        sampled_frame_count = int(frame_count)
    elif "frame_count" in path:
        sampled_frame_count = int(path["frame_count"])
    else:
        sampled_frame_count = max(int(round(path["duration_sec"] * path["fps"])) + 1, 2)
    if sampled_frame_count < 2:
        raise CameraPathError("sampled camera paths require at least two frames.")

    key_times = np.asarray([item["t"] for item in keyframes], dtype=np.float64)
    sample_times = np.linspace(0.0, 1.0, sampled_frame_count, dtype=np.float64)
    positions = np.asarray([item["position"] for item in keyframes], dtype=np.float64)
    rotations_rad = np.unwrap(
        np.deg2rad(np.asarray([item["rotation"] for item in keyframes], dtype=np.float64)),
        axis=0,
    )
    fovs = np.asarray([item["fov"] for item in keyframes], dtype=np.float64)

    segment_indices = np.searchsorted(key_times, sample_times, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(key_times) - 2)
    segment_start = key_times[segment_indices]
    segment_end = key_times[segment_indices + 1]
    alpha = (sample_times - segment_start) / np.maximum(segment_end - segment_start, 1e-8)
    if path["interpolation"] == "smooth":
        alpha = _smoothstep(alpha)
    alpha_column = alpha[:, None]

    sampled_positions = (
        positions[segment_indices] * (1.0 - alpha_column)
        + positions[segment_indices + 1] * alpha_column
    )
    sampled_rotations = (
        rotations_rad[segment_indices] * (1.0 - alpha_column)
        + rotations_rad[segment_indices + 1] * alpha_column
    )
    sampled_fovs = (
        fovs[segment_indices] * (1.0 - alpha)
        + fovs[segment_indices + 1] * alpha
    )

    camera_w2c = np.repeat(np.eye(4, dtype=np.float32)[None], sampled_frame_count, axis=0)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    for index, (position, rotation) in enumerate(zip(sampled_positions, sampled_rotations)):
        pitch, yaw, roll = rotation
        forward = np.asarray(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            dtype=np.float64,
        )
        forward /= np.linalg.norm(forward) + 1e-8
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-8:
            right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(forward, right)
        up /= np.linalg.norm(up) + 1e-8
        if roll:
            rolled_right = math.cos(roll) * right + math.sin(roll) * up
            rolled_up = -math.sin(roll) * right + math.cos(roll) * up
            right, up = rolled_right, rolled_up
        rotation_w2c = np.stack([right, up, forward], axis=0)
        camera_w2c[index, :3, :3] = rotation_w2c.astype(np.float32)
        camera_w2c[index, :3, 3] = (-rotation_w2c @ position).astype(np.float32)

    base_fov = float(sampled_fovs[0])
    zoom_factors = np.tan(np.deg2rad(base_fov) / 2.0) / np.tan(
        np.deg2rad(sampled_fovs) / 2.0
    )
    keyframe_indices = np.rint(key_times * (sampled_frame_count - 1)).astype(np.int64)
    captions = {
        str(int(sample_index)): str(keyframe["prompt"])
        for sample_index, keyframe in zip(keyframe_indices, keyframes)
        if keyframe["prompt"]
    }
    return {
        "camera_path": path,
        "camera_w2c": camera_w2c,
        "zoom_factors": zoom_factors.astype(np.float32),
        "chunk_captions": captions,
        "frame_count": sampled_frame_count,
        "sample_times": sample_times.astype(np.float32),
    }


__all__ = [
    "CameraPathError",
    "CAMERA_PATH_INTERPOLATIONS",
    "CAMERA_PATH_SCHEMA_VERSION",
    "default_camera_path",
    "normalize_camera_path",
    "sample_camera_path",
]
