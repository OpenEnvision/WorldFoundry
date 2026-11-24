"""WorldFoundry facade for StyleGAN Linear Separability metric."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from worldfoundry.evals.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _ls_fn() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    if TYPE_CHECKING:
        from .linear_separability_core import compute_linear_separability
    else:
        from linear_separability_core import compute_linear_separability

    return compute_linear_separability


def compute_linear_separability(confusion_matrix: np.ndarray) -> float:
    """Compute linear separability score from a 2x2 SVM confusion matrix."""
    return float(_ls_fn()(np.asarray(confusion_matrix, dtype=np.float64)))


__all__ = ["compute_linear_separability", "package_root"]
