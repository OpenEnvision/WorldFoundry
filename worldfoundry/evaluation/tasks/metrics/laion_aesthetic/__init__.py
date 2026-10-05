"""LAION aesthetic quality metric (CLIP ViT-L/14 features + linear head)."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

METRIC_ID = "laion_aesthetic"
ALIASES = ("laion-aesthetic", "aesthetic-quality", "aesthetic_quality")
HIGHER_IS_BETTER = True
FAMILY = "scorer"
TAGS = ("scorer", "aesthetics", "clip")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "LAION aesthetic predictor: linear head over L2-normalized CLIP "
        "ViT-L/14 image features, averaged across frames (higher is better)."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)

compute_laion_aesthetic = lazy_export(f"{__name__}.compute", "compute_laion_aesthetic", owner=__name__)
compute = compute_laion_aesthetic

__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_laion_aesthetic",
]
