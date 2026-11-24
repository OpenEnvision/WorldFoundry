"""Camera text-file readers shared by Spatia inference stages."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def read_intrinsics_from_txt(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    values = np.asarray(
        [float(value) for value in _NUMBER_PATTERN.findall(Path(path).read_text(encoding="utf-8"))],
        dtype=np.float32,
    )
    if values.shape != (4,):
        raise ValueError(f"Expected four intrinsics values in {path}, got {values.shape}.")
    return values


def read_w2cs_from_txt(
    path: str | Path,
    *,
    homogeneous: bool = False,
) -> np.ndarray:
    with Path(path).open(encoding="utf-8") as handle:
        matrices = np.asarray([json.loads(line) for line in handle], dtype=np.float32)
    if matrices.ndim != 3 or matrices.shape[2] != 4:
        raise ValueError(f"Unexpected world-to-camera shape in {path}: {matrices.shape}.")
    if matrices.shape[1] == 3:
        row = np.zeros((matrices.shape[0], 1, 4), dtype=np.float32)
        row[:, 0, 3] = 1.0
        matrices_4x4 = np.concatenate((matrices, row), axis=1)
    elif matrices.shape[1] == 4:
        matrices_4x4 = matrices
    else:
        raise ValueError(f"Expected [T,3,4] or [T,4,4] matrices in {path}, got {matrices.shape}.")
    return matrices_4x4 if homogeneous else matrices_4x4[:, :3]


__all__ = ["read_intrinsics_from_txt", "read_w2cs_from_txt"]
