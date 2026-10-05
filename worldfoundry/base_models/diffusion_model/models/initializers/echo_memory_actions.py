"""Echo-Memory planar camera actions (RT12 rows at latent rate).

Converts compact WorldFoundry trajectories (``\"left*80\"``), JSON files,
integer-keyed mappings, or explicit ``[F,12]`` RT rows into the
published Echo convention: XY translation, Z-axis yaw, poses relative
to frame 0, sampled at Wan latent stride (default 4).

This is **not** a :class:`~...contracts.LatentInitializer`.  Recipes /
condition encoders call :func:`echo_camera_trajectory_actions` and place
the result in ``request.inputs[\"actions\"]`` for the Echo denoiser.
Vertical translation and pitch are rejected: upstream checkpoints were
trained planar-only.  Yaw sign is flipped vs the generic OpenCV helper
so it matches Echo's CCW-positive ``R_z``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.core.geometry.trajectory import parse_camera_trajectory


def _rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _as_rt12_rows(values: Any) -> np.ndarray:
    if isinstance(values, Mapping):
        values = values.get("actions", values)
        if isinstance(values, Mapping):
            try:
                items = sorted(values.items(), key=lambda item: int(item[0]))
            except (TypeError, ValueError) as exc:
                raise ValueError("Echo action mappings require integer frame keys") from exc
            values = [value for _, value in items]
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[-1] != 12:
        raise ValueError(f"Echo RT actions must be [F,12], got {rows.shape}")
    if len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("Echo RT actions must be non-empty and finite")
    return rows


def _relative_to_first(rows: np.ndarray) -> np.ndarray:
    reference_t = rows[0, :3]
    reference_r = rows[0, 3:].reshape(3, 3)
    if not np.allclose(reference_r.T @ reference_r, np.eye(3), atol=1e-4):
        raise ValueError("first Echo RT rotation is not orthonormal")
    reference_inverse = reference_r.T
    relative = np.empty_like(rows)
    for index, row in enumerate(rows):
        rotation = row[3:].reshape(3, 3)
        relative[index, :3] = reference_inverse @ (row[:3] - reference_t)
        relative[index, 3:] = (reference_inverse @ rotation).reshape(-1)
    return relative


def _align_to_latent_rate(
    rows: np.ndarray,
    *,
    frame_count: int,
    temporal_stride: int,
    pixel_rate: bool,
) -> np.ndarray:
    latent_count = (frame_count - 1) // temporal_stride + 1
    if not pixel_rate and len(rows) == latent_count:
        return rows
    if pixel_rate:
        if len(rows) < frame_count:
            rows = np.concatenate(
                [rows, np.repeat(rows[-1:], frame_count - len(rows), axis=0)],
                axis=0,
            )
        rows = rows[:frame_count]
        return rows[np.arange(latent_count) * temporal_stride]
    if len(rows) >= frame_count:
        return rows[:frame_count][np.arange(latent_count) * temporal_stride]
    if len(rows) > latent_count:
        step = max(1, len(rows) // latent_count)
        return rows[np.minimum(np.arange(latent_count) * step, len(rows) - 1)]
    if len(rows) < latent_count:
        return np.concatenate(
            [rows, np.repeat(rows[-1:], latent_count - len(rows), axis=0)],
            axis=0,
        )
    return rows


def _planar_rows(
    trajectory: str | Sequence[Mapping[str, float]],
    *,
    translation_step: float,
    rotation_step_degrees: float,
) -> np.ndarray:
    motions = parse_camera_trajectory(
        trajectory,
        translation_step=translation_step,
        rotation_step_degrees=rotation_step_degrees,
    )
    position = np.zeros(3, dtype=np.float64)
    echo_yaw = 0.0
    rows = [np.concatenate([position.copy(), _rotation_z(echo_yaw).reshape(-1)])]
    for motion in motions:
        if abs(float(motion.get("up", 0.0))) > 0 or abs(float(motion.get("pitch", 0.0))) > 0:
            raise ValueError(
                "Echo-Memory checkpoints were trained with planar XY translation and Z-axis yaw; "
                "vertical translation and pitch are unsupported"
            )
        # WorldFoundry's generic OpenCV trajectory convention uses the opposite
        # yaw sign. Echo's published actions use CCW-positive R_z.
        echo_yaw -= float(motion.get("yaw", 0.0))
        right = float(motion.get("right", 0.0))
        forward = float(motion.get("forward", 0.0))
        local_xy = np.asarray([right, forward], dtype=np.float64)
        cosine, sine = math.cos(echo_yaw), math.sin(echo_yaw)
        position[:2] += np.asarray(
            [cosine * local_xy[0] - sine * local_xy[1], sine * local_xy[0] + cosine * local_xy[1]]
        )
        rows.append(np.concatenate([position.copy(), _rotation_z(echo_yaw).reshape(-1)]))
    return np.stack(rows)


def echo_camera_trajectory_actions(
    trajectory: Any,
    *,
    frame_count: int,
    temporal_stride: int = 4,
    translation_step: float = 0.00125,
    rotation_step_degrees: float = 0.5625,
) -> np.ndarray:
    """Return Echo RT12 actions shaped ``[T_latent, 12]``.

    Compact WorldFoundry controls (for example ``"w*80"``), action JSON
    files, integer-keyed mappings, and explicit pixel/latent-rate RT rows are
    accepted. All poses are normalized relative to row zero. Compact controls
    are translated to Echo's published XY-plane, Z-yaw convention rather than
    reusing the generic OpenCV camera-matrix convention. The default translation
    step gives ``w*80`` a +Y displacement of 0.1 per 81-frame chunk; the
    default rotation step gives ``left*80`` a 45-degree turn. Both match the
    upstream replay examples.
    """

    frame_count = int(frame_count)
    temporal_stride = int(temporal_stride)
    if frame_count <= 0 or temporal_stride <= 0:
        raise ValueError("frame_count and temporal_stride must be positive")

    pixel_rate = False
    if isinstance(trajectory, str):
        candidate = Path(trajectory).expanduser()
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as stream:
                rows = _as_rt12_rows(json.load(stream))
        elif candidate.suffix.lower() == ".json":
            raise FileNotFoundError(f"Echo action JSON not found: {candidate}")
        else:
            rows = _planar_rows(
                trajectory,
                translation_step=translation_step,
                rotation_step_degrees=rotation_step_degrees,
            )
            pixel_rate = True
    elif isinstance(trajectory, Sequence) and trajectory and isinstance(trajectory[0], Mapping):
        rows = _planar_rows(
            trajectory,
            translation_step=translation_step,
            rotation_step_degrees=rotation_step_degrees,
        )
        pixel_rate = True
    else:
        rows = _as_rt12_rows(trajectory)

    rows = _relative_to_first(rows)
    rows = _align_to_latent_rate(
        rows,
        frame_count=frame_count,
        temporal_stride=temporal_stride,
        pixel_rate=pixel_rate,
    )
    return rows.astype(np.float32, copy=False)


__all__ = ["echo_camera_trajectory_actions"]
