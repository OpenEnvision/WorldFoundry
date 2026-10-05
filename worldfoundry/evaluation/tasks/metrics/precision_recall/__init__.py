"""Precision and Recall (PRC) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_precision_recall = lazy_export(f"{__name__}.compute", "compute_precision_recall", owner=__name__)

METRIC_ID = "precision_recall"
ALIASES = ("prc", "precision-recall")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Precision and Recall for generative models (Kynkäänniemi et al., 2019).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_precision_recall

__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_precision_recall",
]
