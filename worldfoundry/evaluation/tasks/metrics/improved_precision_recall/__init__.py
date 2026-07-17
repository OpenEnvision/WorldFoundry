"""Improved Precision and Recall (α-precision / β-recall) metric."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

METRIC_ID = "improved_precision_recall"
ALIASES = (
    "ipr",
    "alpha-precision",
    "beta-recall",
    "alpha_precision",
    "beta_recall",
    "realism_score",
    "ipr-realism",
    "realism",
    "ipr_realism",
)
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Improved Precision and Recall (Kynkäänniemi et al., 2019) for generative models. "
        "Includes per-image realism score (``compute_realism_score``) from the same IPR codebase."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute_improved_precision_recall = lazy_export(
    f"{__name__}.wrapper", "compute_improved_precision_recall", owner=__name__
)
compute_realism_score = lazy_export(f"{__name__}.wrapper", "compute_realism_score", owner=__name__)


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    return compute_improved_precision_recall(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_improved_precision_recall",
    "compute_realism_score",
    "package_root",
]
package_root = package_root_export(__file__, owner=__name__)
