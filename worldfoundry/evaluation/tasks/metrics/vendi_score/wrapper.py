"""WorldFoundry facade for Vendi Score diversity metric."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _vendi_module() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from vendi_score import vendi

    return vendi


def compute_vendi_score_from_features(
    features: np.ndarray,
    *,
    q: float | str = 1,
    normalize: bool = True,
) -> float:
    """Compute Vendi Score from precomputed feature vectors (higher is better)."""
    vendi = _vendi_module()
    value = vendi.score_X(np.asarray(features, dtype=np.float64), q=q, normalize=normalize)
    return float(value)


def compute_vendi_score(
    embeddings: Sequence[np.ndarray] | np.ndarray,
    *,
    q: float | str = 1,
    normalize: bool = True,
) -> float:
    """Compute Vendi Score from embedding vectors (higher is better)."""
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return compute_vendi_score_from_features(arr, q=q, normalize=normalize)


__all__ = [
    "compute_vendi_score",
    "compute_vendi_score_from_features",
    "package_root",
]
