"""Manipulation Direction (MD) metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_manipulation_direction = lazy_export(f"{__name__}.wrapper", "compute_manipulation_direction", owner=__name__)
compute_manipulation_direction_batch = lazy_export(f"{__name__}.wrapper", "compute_manipulation_direction_batch", owner=__name__)
compute_manipulation_direction_from_embeddings = lazy_export(f"{__name__}.wrapper", "compute_manipulation_direction_from_embeddings", owner=__name__)
compute_manipulation_direction_from_pairs = lazy_export(f"{__name__}.wrapper", "compute_manipulation_direction_from_pairs", owner=__name__)

METRIC_ID = "manipulation_direction"
ALIASES = ("manipulation-direction", "md", "md_score")
HIGHER_IS_BETTER = True
FAMILY = "scorer"
TAGS = ("scorer", "image_manipulation", "clip", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Manipulation Direction (MD, Sensors 2023): cosine similarity between CLIP image "
        "and text change vectors. WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(*args: Any, **kwargs: Any) -> float:
    return compute_manipulation_direction_from_pairs(*args, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_manipulation_direction",
    "compute_manipulation_direction_batch",
    "compute_manipulation_direction_from_embeddings",
    "compute_manipulation_direction_from_pairs",
]
