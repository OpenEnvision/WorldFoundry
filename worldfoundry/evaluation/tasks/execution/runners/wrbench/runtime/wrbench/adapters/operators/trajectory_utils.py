"""Numpy trajectory helpers for Spatia and related adapters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def goreturn_linspace(peak: float, n: int = 121) -> np.ndarray:
    half = n // 2
    fwd = np.linspace(0.0, peak, half + 1)
    rev = np.linspace(peak, 0.0, n - half)[1:]
    return np.concatenate([fwd, rev])


def make_yaw_w2c(yaw_deg_vals: list[float] | np.ndarray) -> list[np.ndarray]:
    w2cs = []
    for deg in yaw_deg_vals:
        rad = math.radians(float(deg))
        c, s = math.cos(rad), math.sin(rad)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        w2cs.append(w2c)
    return w2cs


def write_json_w2c_file(path: str | Path, w2cs: list[np.ndarray] | np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w2c in np.asarray(w2cs, dtype=np.float64):
            f.write(json.dumps(w2c.tolist()) + "\n")
