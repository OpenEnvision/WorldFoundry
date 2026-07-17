"""WorldFoundry facade for Rényi Kernel Entropy (RKE) diversity metrics."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor"


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _rke_class() -> Any:
    prepend_import_path(VENDOR_ROOT)
    from rke_score import RKE

    return RKE


def _normalize_bandwidth(kernel_bandwidth: float | int | list[float] | tuple[float, ...] | None) -> list[float]:
    if kernel_bandwidth is None:
        return [0.3]
    if isinstance(kernel_bandwidth, (float, int)):
        return [float(kernel_bandwidth)]
    return [float(value) for value in kernel_bandwidth]


def _pick_bandwidth_result(result: float | dict[float, float]) -> float:
    if isinstance(result, dict):
        return float(next(iter(result.values())))
    return float(result)


def compute_rke(
    features: np.ndarray,
    *,
    kernel_bandwidth: float | int | list[float] | tuple[float, ...] | None = 0.3,
    n_samples: int = 1_000_000,
) -> float:
    """Compute RKE mode count (RKE-MC = exp(-RKE)) on feature vectors (higher is more diverse)."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    kernel = _rke_class()(kernel_bandwidth=_normalize_bandwidth(kernel_bandwidth))
    return _pick_bandwidth_result(kernel.compute_rke_mc(arr, n_samples=n_samples))


def compute_rrke(
    reference_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    kernel_bandwidth: float | int | list[float] | tuple[float, ...] | None = 0.3,
    x_samples: int = 500,
    y_samples: int | None = None,
) -> float:
    """Compute Relative RKE (RRKE) between reference and generated features (lower is closer)."""
    ref = np.asarray(reference_features, dtype=np.float64)
    gen = np.asarray(generated_features, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if gen.ndim == 1:
        gen = gen.reshape(1, -1)
    kernel = _rke_class()(kernel_bandwidth=_normalize_bandwidth(kernel_bandwidth))
    return _pick_bandwidth_result(kernel.compute_rrke(ref, gen, x_samples=x_samples, y_samples=y_samples))


__all__ = ["compute_rke", "compute_rrke", "package_root"]
