# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Validated trajectory value objects for Camera Following."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CAMERA_POSES = 9


@dataclass(frozen=True)
class PoseTrajectory:
    poses: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] not in {(3, 4), (4, 4)}:
            raise ValueError(f"pose trajectory must have shape (N,3,4) or (N,4,4), got {poses.shape}")
        if len(poses) < MIN_CAMERA_POSES:
            raise ValueError(f"pose trajectory requires at least {MIN_CAMERA_POSES} poses; got {len(poses)}")
        if poses.shape[1:] == (3, 4):
            padded = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
            padded[:, :3, :] = poses
            poses = padded
        if not np.isfinite(poses).all():
            raise ValueError("pose trajectory must contain only finite values")
        expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(poses[:, 3, :], expected_bottom, atol=1e-6):
            raise ValueError("pose trajectory bottom rows must be [0,0,0,1]")
        rotations = poses[:, :3, :3]
        gram = np.swapaxes(rotations, 1, 2) @ rotations
        determinants = np.linalg.det(rotations)
        if not np.allclose(gram, np.eye(3), atol=5e-3) or not np.allclose(
            determinants,
            1.0,
            atol=5e-3,
        ):
            raise ValueError("pose trajectory contains an invalid rotation matrix")

        object.__setattr__(self, "poses", np.ascontiguousarray(poses))

    @property
    def num_poses(self) -> int:
        return len(self.poses)
