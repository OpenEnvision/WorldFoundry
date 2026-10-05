"""Monge Inception Distance computation via in-tree torch-fidelity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.torch_fidelity import calculate_metrics, prepare_image_input


def _parse_numeric_result(result: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_mind(
    reference: str | Path | Sequence[str | Path],
    generated: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 64,
    cuda: bool = True,
    mind_num_projections: int = 10000,
    feature_extractor: str = "inception-v3-compat",
    **kwargs: Any,
) -> float:
    result = calculate_metrics()(
        input1=prepare_image_input(reference, **kwargs),
        input2=prepare_image_input(generated, **kwargs),
        batch_size=batch_size,
        cuda=cuda,
        feature_extractor=feature_extractor,
        mind=True,
        mind_num_projections=mind_num_projections,
        **kwargs,
    )
    parsed = _parse_numeric_result(result)
    for key, value in parsed.items():
        if "monge_inception_distance" in key.lower() or key.lower() == "mind":
            return float(value)
    if len(parsed) == 1:
        return float(next(iter(parsed.values())))
    raise KeyError("MIND key missing from torch-fidelity result")


__all__ = ["compute_mind"]
