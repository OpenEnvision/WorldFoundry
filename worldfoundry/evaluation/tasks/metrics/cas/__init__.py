"""Classification Accuracy Score (CAS) metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_cas_from_predictions = lazy_export(f"{__name__}.wrapper", "compute_cas_from_predictions", owner=__name__)
train_classifier_and_compute_cas = lazy_export(f"{__name__}.wrapper", "train_classifier_and_compute_cas", owner=__name__)

METRIC_ID = "cas"
ALIASES = ("classification-accuracy-score", "classification_accuracy_score")
HIGHER_IS_BETTER = True
FAMILY = "distribution"
TAGS = ("distribution", "conditional_generation", "classification", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Classification Accuracy Score (CAS, arXiv:1905.10887): train a classifier on "
        "synthetic images and measure Top-k accuracy on real labels. "
        "WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(
    real_labels: np.ndarray | list[Any],
    predicted_labels: np.ndarray | list[Any],
    **kwargs: Any,
) -> dict[str, float]:
    return compute_cas_from_predictions(real_labels, predicted_labels, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_cas_from_predictions",
    "train_classifier_and_compute_cas",
]
