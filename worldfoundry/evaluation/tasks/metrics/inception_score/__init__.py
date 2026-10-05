"""Inception Score (IS) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_inception_score = lazy_export(f"{__name__}.compute", "compute_inception_score", owner=__name__)

METRIC_ID = "inception_score"
ALIASES = ("is", "isc")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "fid_family")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Inception Score for generated image sets (torch-fidelity, in-tree).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_inception_score

__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_inception_score",
]
