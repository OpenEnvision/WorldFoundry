"""PhyGenBench official result normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _dimension_scores,
)

BENCHMARK_ID = "phygenbench"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "PhyGenBench prompts.json and generated videos",
        "single_question.json, multi_question.json, and video_question.json",
        "VQAScore outputs for key physical phenomena detection",
        "GPT-4o or LLaVA-Interleave outputs for order verification",
        "GPT-4o or InternVideo2 outputs for overall naturalness",
        "PhyGenEval overall result JSON/CSV when importing scores",
    ],
}


def official_scores_from_records(
    records: list[Mapping[str, Any]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    if not records:
        return {}
    return _dimension_scores(
        records,
        ("physical_commonsense", "physical_law_adherence", "semantic_adherence"),
        "phygenbench_average",
        official_results_path,
    )
