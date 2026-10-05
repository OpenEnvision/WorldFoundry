"""Object-wise consistency metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_object_wise_consistency = lazy_export(f"{__name__}.wrapper", "compute_object_wise_consistency", owner=__name__)
compute_object_wise_consistency_batch = lazy_export(f"{__name__}.wrapper", "compute_object_wise_consistency_batch", owner=__name__)
compute_object_wise_iou = lazy_export(f"{__name__}.wrapper", "compute_object_wise_iou", owner=__name__)

METRIC_ID = "object_wise_consistency"
ALIASES = (
    "object-wise-consistency",
    "location_aware_consistency",
    "location-aware-consistency",
    "object_wise_iou",
)
HIGHER_IS_BETTER = True
FAMILY = "perceptual"
TAGS = ("perceptual", "detection", "layout", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Object-wise consistency (arXiv:2304.13427): max IoU between YOLOR-style detections "
        "and guidance boxes, plus success rate R_suc at IoU>0.5. "
        "WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(*args: Any, **kwargs: Any) -> dict[str, float]:
    return compute_object_wise_consistency(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_object_wise_consistency",
    "compute_object_wise_consistency_batch",
    "compute_object_wise_iou",
]
