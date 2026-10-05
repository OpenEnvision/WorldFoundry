"""Random Network Distillation (RND) diversity metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_rnd = lazy_export(f"{__name__}.wrapper", "compute_rnd", owner=__name__)
compute_rnd_from_images = lazy_export(f"{__name__}.wrapper", "compute_rnd_from_images", owner=__name__)

METRIC_ID = "rnd"
ALIASES = ("rnd-score", "random-network-distillation", "rnd_diversity")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "diversity", "text_generation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Random Network Distillation diversity score (Fowl et al., arXiv:2010.06715) "
        "on feature vectors or flattened images."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(features: np.ndarray, **kwargs: Any) -> float:
    return compute_rnd(features, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_rnd",
    "compute_rnd_from_images",
]
