"""Rényi Kernel Entropy (RKE) diversity metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_rke = lazy_export(f"{__name__}.wrapper", "compute_rke", owner=__name__)
compute_rrke = lazy_export(f"{__name__}.wrapper", "compute_rrke", owner=__name__)

METRIC_ID = "rke"
ALIASES = ("renyi-kernel-entropy", "rke-mc", "rke_mc")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "diversity")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Rényi Kernel Entropy mode count (RKE-MC) on feature embeddings (higher is more diverse).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(
    features: np.ndarray,
    *,
    kernel_bandwidth: float | int | list[float] | tuple[float, ...] | None = 0.3,
    n_samples: int = 1_000_000,
) -> float:
    return compute_rke(features, kernel_bandwidth=kernel_bandwidth, n_samples=n_samples)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_rke",
    "compute_rrke",
]
