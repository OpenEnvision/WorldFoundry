"""SadPaD attribute metrics (SaD and PaD from one upstream repo)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path
from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


@lru_cache(maxsize=1)
def _get_sad_fn() -> Any:
    prepend_import_path(VENDOR_ROOT)
    from sad import get_SaD

    return get_SaD


@lru_cache(maxsize=1)
def _get_pad_fn() -> Any:
    prepend_import_path(VENDOR_ROOT)
    from pad import get_PaD

    return get_PaD


def compute_attribute_sad(
    hcs_real: Any,
    hcs_gen: Any,
    text_list: list[str],
    *,
    n_points: int = 1000,
) -> dict[str, np.ndarray | float]:
    """Compute Attribute SaD from precomputed HCS feature tensors."""
    sad_result = _get_sad_fn()(hcs_real, hcs_gen, text_list, n_points)
    return {
        "sad_per_attribute": sad_result,
        "kl_mean": float(np.mean(sad_result[:, 0])),
        "mean_difference_mean": float(np.mean(sad_result[:, 1])),
        "jsd_mean": float(np.mean(sad_result[:, 2])),
    }


def compute_attribute_pad(
    hcs_real: Any,
    hcs_gen: Any,
    text_list: list[str],
    *,
    n_points: int = 1000,
) -> dict[str, Any]:
    """Compute Attribute PaD from precomputed HCS feature tensors."""
    pad_result = _get_pad_fn()(hcs_real, hcs_gen, text_list, n_points)
    kl_values = pad_result[:, 0].cpu().numpy()
    jsd_values = pad_result[:, 1].cpu().numpy()
    return {
        "pad_per_pair": pad_result,
        "kl_mean": float(np.mean(kl_values)),
        "jsd_mean": float(np.mean(jsd_values)),
    }


def _compute_sad(hcs_real: Any, hcs_gen: Any, text_list: list[str], **kwargs: Any) -> dict[str, Any]:
    return compute_attribute_sad(hcs_real, hcs_gen, text_list, **kwargs)


def _compute_pad(hcs_real: Any, hcs_gen: Any, text_list: list[str], **kwargs: Any) -> dict[str, Any]:
    return compute_attribute_pad(hcs_real, hcs_gen, text_list, **kwargs)


METRIC_MODULES = (
    metric_module_from_globals(
        metric_id="attribute_sad",
        aliases=("sad", "semantic-attribute-distance", "attribute-sad"),
        description=(
            "Attribute Semantic Attribute Distance (SaD) from HCS features "
            "(requires precomputed HCS tensors; see sadpad HCS pipeline)."
        ),
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "text_to_image", "attribute", "sadpad"),
        implementation="worldfoundry.evaluation.tasks.metrics.sadpad:compute_attribute_sad",
    ),
    metric_module_from_globals(
        metric_id="attribute_pad",
        aliases=("pad", "pairwise-attribute-distance", "attribute-pad"),
        description=(
            "Attribute Pairwise Attribute Distance (PaD) from HCS features "
            "(requires precomputed HCS tensors; see sadpad HCS pipeline)."
        ),
        family="distribution",
        higher_is_better=False,
        tags=("distribution", "text_to_image", "attribute", "sadpad"),
        implementation="worldfoundry.evaluation.tasks.metrics.sadpad:compute_attribute_pad",
    ),
)

METRIC_MODULE = METRIC_MODULES[0]

compute = _compute_sad

__all__ = [
    "METRIC_MODULE",
    "METRIC_MODULES",
    "compute",
    "compute_attribute_pad",
    "compute_attribute_sad",
]
