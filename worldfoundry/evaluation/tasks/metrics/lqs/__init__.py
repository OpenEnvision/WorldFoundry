"""Layout Quality Score (LQS) metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_lqs = lazy_export(f"{__name__}.wrapper", "compute_lqs", owner=__name__)

METRIC_ID = "lqs"
ALIASES = ("layout-quality-score", "layout_quality_score")
HIGHER_IS_BETTER = True
FAMILY = "perceptual"
TAGS = ("perceptual", "layout", "text_to_image")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Layout Quality Score (LayoutTransformer, arXiv:2208.06162) comparing predicted "
        "and ground-truth bounding-box layouts."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(
    groundtruth_layout: Sequence[dict[str, Any]],
    predicted_layout: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, float]:
    return compute_lqs(groundtruth_layout, predicted_layout, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_lqs",
]
