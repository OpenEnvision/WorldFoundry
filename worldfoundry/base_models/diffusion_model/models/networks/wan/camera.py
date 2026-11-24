"""Wan camera-control Plücker geometry.

Wan camera-control checkpoints consume a planar camera trajectory as compact
19-D rows and then as ``[T, H, W, 6]`` Plücker features. ``SimpleAdapter`` in
:mod:`.adapter` builds its control stem from these; the geometry stays with the
Wan family so other callers do not reimplement it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def generate_planar_camera_coordinates(
    direction: str,
    length: int,
    *,
    speed: float = 1 / 54,
    origin: Sequence[float] = (
        0,
        0.532139961,
        0.946026558,
        0.5,
        0.5,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    ),
) -> list[list[float]]:
    """Create the compact camera rows used by Wan camera-control checkpoints.

    Each row is the 19-D layout ``[id, fx, fy, cx, cy, 0, 0, *w2c_3x4]``.
    Planar directions translate the camera along world X (left/right)
    and Y (up/down) by *speed* per frame, starting from *origin*.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    supported = {
        "Left",
        "Right",
        "Up",
        "Down",
        "LeftUp",
        "LeftDown",
        "RightUp",
        "RightDown",
    }
    if direction not in supported:
        raise ValueError(f"unsupported camera direction {direction!r}")
    coordinates = [list(origin)]
    while len(coordinates) < length:
        row = coordinates[-1].copy()
        if "Left" in direction:
            row[9] += speed
        if "Right" in direction:
            row[9] -= speed
        if "Up" in direction:
            row[13] += speed
        if "Down" in direction:
            row[13] -= speed
        coordinates.append(row)
    return coordinates


def wan_camera_coordinates_to_plucker(
    coordinates: Sequence[Sequence[float]],
    *,
    width: int = 672,
    height: int = 384,
    original_width: int = 1280,
    original_height: int = 720,
    device: Any = "cpu",
):
    """Convert Wan 19-D camera rows to ``[T, H, W, 6]`` Plücker features.

    Rescales normalized fx/fy for letterboxed resize from
    ``original_width x original_height`` to ``width x height``, recenters
    poses so frame 0 is the identity, then calls
    :func:`worldfoundry.core.geometry.transforms.ray_condition`.
    """

    import torch

    from worldfoundry.core.geometry.transforms import ray_condition

    rows = np.asarray(coordinates, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 19:
        raise ValueError(f"camera coordinates must be [T,19], got {rows.shape}")
    intrinsics = rows[:, 1:5].copy()
    sample_ratio = float(width) / float(height)
    source_ratio = float(original_width) / float(original_height)
    if source_ratio > sample_ratio:
        intrinsics[:, 0] *= (height * source_ratio) / width
    else:
        intrinsics[:, 1] *= (width / source_ratio) / height
    intrinsics *= np.asarray((width, height, width, height), dtype=np.float32)

    world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None], len(rows), axis=0)
    world_to_camera[:, :3, :] = rows[:, 7:].reshape(-1, 3, 4)
    camera_to_world = np.linalg.inv(world_to_camera)
    target = np.eye(4, dtype=np.float32)
    relative_transform = target @ world_to_camera[0]
    relative_camera_to_world = np.stack([target, *[relative_transform @ pose for pose in camera_to_world[1:]]])
    k = torch.as_tensor(intrinsics, device=device).unsqueeze(0)
    c2w = torch.as_tensor(relative_camera_to_world, device=device).unsqueeze(0)
    return ray_condition(k, c2w, height, width, device=device)[0]
