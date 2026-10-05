"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

from __future__ import annotations

from typing import Literal

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

NetType = Literal["alex", "vgg", "squeeze"]
compute_lpips = lazy_export(f"{__name__}.compute", "compute_lpips", owner=__name__)

METRIC_ID = "lpips"
ALIASES: tuple[str, ...] = ()
HIGHER_IS_BETTER = False
FAMILY = "perceptual"
TAGS = ("perceptual", "condition_consistency")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Learned Perceptual Image Patch Similarity (pairwise, lower is better).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_lpips

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "NetType", "compute", "compute_lpips"]
