"""Image Realism Score (IRS) metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_irs = lazy_export(f"{__name__}.wrapper", "compute_irs", owner=__name__)
compute_irs_measures = lazy_export(f"{__name__}.wrapper", "compute_irs_measures", owner=__name__)
compute_irs_with_reference = lazy_export(f"{__name__}.wrapper", "compute_irs_with_reference", owner=__name__)
fit_irs_reference_means = lazy_export(f"{__name__}.wrapper", "fit_irs_reference_means", owner=__name__)

METRIC_ID = "irs"
ALIASES = ("image-realism-score", "image_realism_score")
HIGHER_IS_BETTER = True
FAMILY = "perceptual"
TAGS = ("perceptual", "no_reference", "realism", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Image Realism Score (IRS, arXiv:2309.14756): pentagon area over five calibrated "
        "image statistics. WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(*args: Any, **kwargs: Any) -> float:
    return compute_irs(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_irs",
    "compute_irs_measures",
    "compute_irs_with_reference",
    "fit_irs_reference_means",
]
