"""DEVIL Dynamics official result normalization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _dimension_scores,
)

BENCHMARK_ID = "devil-dynamics"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "official DEVIL prompts with dynamics-grade filename prefixes",
        "generated videos named with the official dynamics prefixes",
        "DEVIL model_weights for dynamics scores",
        "Gemini 1.5 Pro API outputs for naturalness",
        "official DEVIL result JSON/CSV when importing scores",
    ],
}

DIMENSION_IDS = ("dynamics_range", "dynamics_controllability", "dynamics_quality")
AVERAGE_ID = "devil_dynamics_average"


def official_scores_from_records(
    records: list[Mapping[str, Any]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    if not records:
        return {}
    scores = _dimension_scores(
        records,
        DIMENSION_IDS,
        AVERAGE_ID,
        official_results_path,
    )
    if "dynamics_quality" not in scores:
        naturalness = _dimension_scores(
            records,
            ("dynamics_naturalness",),
            "dynamics_naturalness",
            official_results_path,
        ).get("dynamics_naturalness")
        if naturalness is not None:
            scores["dynamics_quality"] = naturalness
    return scores
