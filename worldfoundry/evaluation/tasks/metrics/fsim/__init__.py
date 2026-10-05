"""Feature Similarity Index Measure (FSIM) metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_fsim = lazy_export(f"{__name__}.wrapper", "compute_fsim", owner=__name__)

METRIC_ID = "fsim"
ALIASES = ("feature-similarity-index", "fsimc")
HIGHER_IS_BETTER = True
FAMILY = "perceptual"
TAGS = ("perceptual", "condition_consistency", "full_reference")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Feature Similarity Index Measure between reference and generated images (higher is better).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    data_range: float | None = None,
    chromatic: bool = True,
    device: str | None = None,
) -> float:
    return compute_fsim(
        reference,
        generated,
        data_range=data_range,
        chromatic=chromatic,
        device=device,
    )


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_fsim",
]
