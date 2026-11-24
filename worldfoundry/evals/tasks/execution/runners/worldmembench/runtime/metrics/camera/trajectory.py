# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Reference path normalization and Sim(3) alignment for full-frame trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import PoseTrajectory

MOTION_TYPES = frozenset({"translation", "rotation", "mixed"})


def canonicalize_target(
    trajectory: PoseTrajectory,
    motion_type: str,
    *,
    epsilon: float = 1e-12,
) -> tuple[PoseTrajectory, float | None]:
    if motion_type not in MOTION_TYPES:
        raise ValueError(f"camera_motion_type must be one of {sorted(MOTION_TYPES)}; got {motion_type!r}")
    if motion_type == "rotation":
        return trajectory, None

    centers = trajectory.poses[:, :3, 3]
    path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    if not np.isfinite(path_length) or path_length <= epsilon:
        raise ValueError("translation/mixed target must have non-zero finite total path length")
    poses = trajectory.poses.copy()
    poses[:, :3, 3] = (centers - centers[0]) / path_length
    return (
        PoseTrajectory(
            poses,
            source=f"{trajectory.source}:canonical",
        ),
        path_length,
    )


@dataclass(frozen=True)
class UmeyamaAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    poses: np.ndarray


def umeyama_align(
    estimated: PoseTrajectory,
    target: PoseTrajectory,
    *,
    epsilon: float = 1e-12,
) -> UmeyamaAlignment:
    if estimated.num_poses != target.num_poses:
        raise ValueError("estimated and target trajectories must have equal length")
    source = estimated.poses[:, :3, 3]
    destination = target.poses[:, :3, 3]
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = float(np.square(source_centered).sum() / len(source))

    if source_variance <= epsilon:
        alignment_rotation = np.eye(3, dtype=np.float64)
        scale = 0.0
    else:
        covariance = destination_centered.T @ source_centered / len(source)
        left, singular_values, right_transpose = np.linalg.svd(covariance)
        reflection = np.ones(3, dtype=np.float64)
        if np.linalg.det(left @ right_transpose) < 0:
            reflection[-1] = -1.0
        reflection_guard = np.diag(reflection)
        alignment_rotation = left @ reflection_guard @ right_transpose
        scale = float((singular_values * reflection).sum() / (source_variance + epsilon))

    translation = destination_mean - scale * alignment_rotation @ source_mean
    aligned = estimated.poses.copy()
    aligned[:, :3, :3] = np.einsum(
        "ij,njk->nik",
        alignment_rotation,
        estimated.poses[:, :3, :3],
    )
    aligned[:, :3, 3] = scale * (alignment_rotation @ source.T).T + translation
    return UmeyamaAlignment(
        scale=scale,
        rotation=alignment_rotation,
        translation=translation,
        poses=aligned,
    )
