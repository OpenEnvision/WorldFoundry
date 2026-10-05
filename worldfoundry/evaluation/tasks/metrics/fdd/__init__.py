"""Fréchet Denoised Distance (FDD) metric."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

compute_fdd = lazy_export(f"{__name__}.wrapper", "compute_fdd", owner=__name__)

METRIC_ID = "fdd"
ALIASES = ("frechet-denoised-distance", "frechet_denoised_distance")
HIGHER_IS_BETTER = False
FAMILY = "distribution"
TAGS = ("distribution", "image_generation", "denoised")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Fréchet Denoised Distance (FDD) using a pretrained DAE encoder "
        "(requires Google Drive checkpoint or WORLDFOUNDRY_FDD_DAE_CKPT)."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    **kwargs: Any,
) -> float:
    return compute_fdd(reference, generated, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_fdd",
]
