"""Inception Score computation via in-tree torch-fidelity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, prepare_image_input


def _parse_numeric_result(result: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_inception_score(
    images: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    splits: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    result = calculate_metrics()(
        input1=prepare_image_input(images, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        isc=True,
        isc_splits=splits,
        **kwargs,
    )
    return _parse_numeric_result(result)


__all__ = ["compute_inception_score"]
