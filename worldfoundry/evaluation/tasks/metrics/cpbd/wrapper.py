"""WorldFoundry facade for CPBD blur/sharpness metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _cpbd_fn() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from compute import compute

    return compute


def compute_cpbd(image: np.ndarray) -> float:
    """Compute CPBD sharpness score for a grayscale or RGB image array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    return float(_cpbd_fn()(arr))


__all__ = ["compute_cpbd", "package_root"]
