"""Protocol aggregation kept separate from metric execution."""

from __future__ import annotations

import math
from typing import Any, Iterable



def clean_float(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {key: clean_float(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_float(item) for item in value]
    return value


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean_valid(values: Iterable[Any]) -> float | None:
    valid = [float(value) for value in values if numeric(value) is not None]
    return round(sum(valid) / len(valid), 6) if valid else None


def soft_threshold(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def skill_result(
    skill_id: str,
    status: str,
    score: float | None,
    metrics: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return clean_float(
        {
            "skill_id": skill_id,
            "status": status,
            "score": score,
            "metrics": metrics or {},
            "diagnostics": diagnostics or {},
            "notes": notes or [],
        }
    )
