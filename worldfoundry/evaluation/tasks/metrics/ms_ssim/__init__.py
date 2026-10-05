"""Multi-Scale SSIM (MS-SSIM) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_ms_ssim = lazy_export(f"{__name__}.compute", "compute_ms_ssim", owner=__name__)

METRIC_ID = "ms_ssim"
ALIASES = ("ms-ssim",)
HIGHER_IS_BETTER = True
FAMILY = "perceptual"
TAGS = ("perceptual", "condition_consistency")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description="Multi-Scale SSIM (pairwise, higher is better).",
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute = compute_ms_ssim

__all__ = ["ALIASES", "FAMILY", "HIGHER_IS_BETTER", "METRIC_ID", "METRIC_MODULE", "compute", "compute_ms_ssim"]
