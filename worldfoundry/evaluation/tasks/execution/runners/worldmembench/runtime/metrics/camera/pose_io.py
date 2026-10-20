# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Read full-frame target c2w, DA3 w2c, and ViPE c2w pose arrays."""

from pathlib import Path

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)

from .schemas import PoseTrajectory


def load_trajectory(path, kind: str) -> PoseTrajectory:
    error_type = DataInvalidError if kind == "target" else EvaluationError
    try:
        path = Path(path).expanduser().resolve()
        if kind == "target" and path.suffix == ".npy":
            poses = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as data:
                poses = np.asarray(data["extrinsics" if kind == "da3" else "c2w"], dtype=np.float64)
                if "frame_indices" in data and not np.array_equal(data["frame_indices"], np.arange(len(poses))):
                    raise ValueError("frame_indices must cover every frame in exact 0..N-1 order")
        if kind == "da3":
            matrices = np.asarray(poses, dtype=np.float64)
            if matrices.ndim == 3 and matrices.shape[1:] == (3, 4):
                padded = np.repeat(np.eye(4)[None], len(matrices), axis=0)
                padded[:, :3] = matrices
                matrices = padded
            if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
                raise ValueError("DA3 extrinsics must have shape (N,3,4) or (N,4,4)")
            poses = np.linalg.inv(matrices)
        return PoseTrajectory(poses, source=kind)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        raise error_type(f"invalid {kind} pose artifact {path}: {error}") from error
