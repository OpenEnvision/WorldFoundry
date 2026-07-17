"""VideoPhy2 official result normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.official_result_scoring import (
    OfficialMetricScore,
    _rule_followed_rate,
    _score_from_records,
)
from worldfoundry.evaluation.tasks.execution.runners.videophy.videophy_official_scoring import (
    official_scores_from_records as videophy_official_scores_from_records,
)

BENCHMARK_ID = "videophy2"

OFFICIAL_REQUIREMENTS: dict[str, Any] = {
    "reason": "judge_required",
    "required_inputs": [
        "videophysics/videophy2_test prompt and rule metadata",
        "generated videos for the official prompts",
        "human SA/PC/rule labels or VideoPhy-2-AutoEval outputs",
        "per-sample sa, pc, joint, and followed/violated physical-rule fields",
    ],
}


def official_scores_from_records(
    records: list[Mapping[str, Any]],
    official_results_path: Path | None,
) -> dict[str, OfficialMetricScore]:
    if not records:
        return {}
    scores = videophy_official_scores_from_records(
        records,
        official_results_path,
        average_id=None,
        rating_scale_max=5.0,
    )
    rule_score = _score_from_records(
        records,
        "rule_followed_rate",
        official_results_path,
        aliases=("physical_rule_followed_rate",),
    )
    if rule_score is None:
        rule_score = _rule_followed_rate(records, official_results_path)
    if rule_score is not None:
        scores["rule_followed_rate"] = rule_score
    return scores
