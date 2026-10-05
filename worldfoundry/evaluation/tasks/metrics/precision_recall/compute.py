"""Precision/Recall computation via in-tree torch-fidelity (Kynkäänniemi et al., 2019)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, prepare_image_input


def _parse_numeric_result(result: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_precision_recall(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> dict[str, float]:
    """Compute precision/recall with backend input1=generated, input2=reference."""
    result = calculate_metrics()(
        input1=prepare_image_input(generated, **kwargs),
        input2=prepare_image_input(reference, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        prc=True,
        **kwargs,
    )
    return _parse_numeric_result(result)


__all__ = ["compute_precision_recall"]
