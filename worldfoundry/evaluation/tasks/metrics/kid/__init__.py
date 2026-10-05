"""Kernel Inception Distance (KID) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_kid = lazy_export(f"{__name__}.compute", "compute_kid", owner=__name__)
polynomial_mmd = lazy_export(f"{__name__}.compute", "polynomial_mmd", owner=__name__)

METRIC_ID = "kid"
ALIASES = ("kernel-inception-distance",)
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "fid_family")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Kernel Inception Distance between reference and generated image sets.",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_kid

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "compute", "compute_kid", "polynomial_mmd"]
