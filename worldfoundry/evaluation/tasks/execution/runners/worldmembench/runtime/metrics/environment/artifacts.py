# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Read the two DA3 artifacts used by Environment scoring."""

from dataclasses import dataclass

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError


def load_poses(path, total_frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        matrices = np.asarray(data["extrinsics"], dtype=np.float64)
        indices = np.asarray(data["frame_indices"])
    if total_frames < 9 or len(matrices) != total_frames:
        raise DataInvalidError("DA3 must provide one pose per video frame, with at least nine frames")
    if not np.array_equal(indices, np.arange(total_frames)):
        raise DataInvalidError("DA3 frame_indices must equal 0..video_frame_count-1")
    if matrices.shape == (total_frames, 3, 4):
        padded = np.repeat(np.eye(4)[None], total_frames, axis=0)
        padded[:, :3] = matrices
        matrices = padded
    if matrices.shape != (total_frames, 4, 4) or not np.isfinite(matrices).all():
        raise DataInvalidError("DA3 extrinsics must be finite [T,3,4] or [T,4,4] matrices")
    poses = np.linalg.inv(matrices)
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=5e-3)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=5e-3)
    ):
        raise DataInvalidError("DA3 extrinsics contain invalid camera transformations")
    return np.ascontiguousarray(poses)


@dataclass(frozen=True)
class SceneGeometry:
    frame_indices: np.ndarray
    depths: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    c2w: np.ndarray

    @property
    def index_to_position(self) -> dict[int, int]:
        return {int(index): position for position, index in enumerate(self.frame_indices)}


def load_scene_geometry(path, total_frames: int) -> SceneGeometry:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"].item()) != 1:
            raise DataInvalidError("unsupported DA3 scene geometry schema")
        if int(data["source_frame_count"].item()) != total_frames:
            raise DataInvalidError("DA3 scene geometry source_frame_count does not match the video")
        geometry = SceneGeometry(
            np.asarray(data["frame_indices"], dtype=np.int64),
            np.asarray(data["depths"], dtype=np.float32),
            np.asarray(data["confidence"], dtype=np.float32),
            np.asarray(data["intrinsics"], dtype=np.float64),
            np.asarray(data["c2w"], dtype=np.float64),
        )
    indices, depths = geometry.frame_indices, geometry.depths
    count = len(indices)
    if (
        indices.ndim != 1
        or count < 2
        or indices[0] != 0
        or indices[-1] >= total_frames
        or not np.all(np.diff(indices) > 0)
    ):
        raise DataInvalidError("scene frame_indices must start at zero and increase within the video")
    if (
        depths.ndim != 3
        or len(depths) != count
        or geometry.confidence.shape != depths.shape
        or geometry.intrinsics.shape != (count, 3, 3)
        or geometry.c2w.shape != (count, 4, 4)
    ):
        raise DataInvalidError("inconsistent DA3 scene geometry array shapes")
    if any(not np.isfinite(array).all() for array in (depths, geometry.confidence, geometry.intrinsics, geometry.c2w)):
        raise DataInvalidError("DA3 scene geometry contains non-finite values")
    if np.any(depths < 0) or np.any(np.sum(depths > 0, axis=(1, 2)) == 0) or np.any(geometry.confidence < 0):
        raise DataInvalidError("DA3 depth/confidence must be non-negative with positive depth per frame")
    return geometry
