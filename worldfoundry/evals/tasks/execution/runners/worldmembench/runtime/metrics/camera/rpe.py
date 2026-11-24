# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Adjacent-step translation and rotation Relative Pose Error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import PoseTrajectory
from .trajectory import MOTION_TYPES


def _rotation_errors_deg(error_rotations: np.ndarray) -> np.ndarray:
    traces = np.trace(error_rotations, axis1=1, axis2=2)
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


@dataclass(frozen=True)
class RPEStatistics:
    translation_errors: np.ndarray | None
    rotation_errors_deg: np.ndarray

    @property
    def translation_mean(self) -> float | None:
        if self.translation_errors is None:
            return None
        return float(np.mean(self.translation_errors))

    @property
    def translation_median(self) -> float | None:
        if self.translation_errors is None:
            return None
        return float(np.median(self.translation_errors))

    @property
    def translation_path(self) -> float | None:
        """Sum of adjacent errors on the unit-path canonical target."""
        if self.translation_errors is None:
            return None
        return float(np.sum(self.translation_errors))

    @property
    def rotation_mean_deg(self) -> float:
        return float(np.mean(self.rotation_errors_deg))

    @property
    def rotation_median_deg(self) -> float:
        return float(np.median(self.rotation_errors_deg))

    @property
    def rotation_sum_deg(self) -> float:
        return float(np.sum(self.rotation_errors_deg))

    def rotation_path(self, target_rotation_path_deg: float) -> float:
        """Normalize the accumulated rotation error by the target path."""

        path = float(target_rotation_path_deg)
        if not np.isfinite(path) or path <= 1e-12:
            raise ValueError("target rotation path must be positive and finite")
        return self.rotation_sum_deg / path


def rotation_path_length_deg(trajectory: PoseTrajectory) -> float:
    """Return the exact sum of adjacent target rotation magnitudes."""

    rotations = trajectory.poses[:, :3, :3]
    relative = np.einsum(
        "nij,njk->nik",
        np.swapaxes(rotations[:-1], 1, 2),
        rotations[1:],
    )
    path = float(np.sum(_rotation_errors_deg(relative)))
    if not np.isfinite(path):
        raise ValueError("target rotation path must be finite")
    return path


def compute_rpe(
    target: PoseTrajectory,
    estimated: PoseTrajectory,
    motion_type: str,
) -> RPEStatistics:
    if motion_type not in MOTION_TYPES:
        raise ValueError(f"unsupported camera_motion_type: {motion_type!r}")
    if target.num_poses != estimated.num_poses:
        raise ValueError("target and estimated trajectories must have equal length")

    target_poses = target.poses
    estimated_poses = estimated.poses
    if motion_type == "rotation":
        target_relative = np.einsum(
            "nij,njk->nik",
            np.swapaxes(target_poses[:-1, :3, :3], 1, 2),
            target_poses[1:, :3, :3],
        )
        estimated_relative = np.einsum(
            "nij,njk->nik",
            np.swapaxes(estimated_poses[:-1, :3, :3], 1, 2),
            estimated_poses[1:, :3, :3],
        )
        rotation_error = np.einsum(
            "nij,njk->nik",
            np.swapaxes(target_relative, 1, 2),
            estimated_relative,
        )
        translation_errors = None
    else:
        target_relative = np.linalg.inv(target_poses[:-1]) @ target_poses[1:]
        estimated_relative = np.linalg.inv(estimated_poses[:-1]) @ estimated_poses[1:]
        error_transform = np.linalg.inv(target_relative) @ estimated_relative
        translation_errors = np.linalg.norm(error_transform[:, :3, 3], axis=1)
        rotation_error = error_transform[:, :3, :3]

    rotation_errors = _rotation_errors_deg(rotation_error)
    if translation_errors is not None and not np.isfinite(translation_errors).all():
        raise ValueError("translation RPE contains non-finite values")
    if not np.isfinite(rotation_errors).all():
        raise ValueError("rotation RPE contains non-finite values")
    return RPEStatistics(
        translation_errors=translation_errors,
        rotation_errors_deg=rotation_errors,
    )
