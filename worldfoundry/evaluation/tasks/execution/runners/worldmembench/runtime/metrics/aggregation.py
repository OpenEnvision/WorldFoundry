# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Equal-weight case aggregation shared by video metrics."""

from collections import Counter

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    ResultStatus,
    SummaryResult,
)


def aggregate_video_summary(items: list[ItemResult]) -> SummaryResult:
    # Weight videos equally within each model and retain status/coverage counts.
    groups: dict[str, list[ItemResult]] = {}
    detailed_groups: dict[tuple, list[ItemResult]] = {}
    for item in items:
        key = str(item.model_id) if item.model_id is not None else "all"
        groups.setdefault(key, []).append(item)
        detailed_groups.setdefault((item.model_id, item.category, item.condition_id), []).append(item)

    def aggregate_videos(group):
        scored = [item for item in group if item.status.is_scored and bool(item.scores)]
        score_values: dict[str, list[float]] = {}
        for item in scored:
            for score_key, value in item.scores.items():
                score_values.setdefault(score_key, []).append(float(value))
        return {
            "scores": {name: sum(values) / len(values) for name, values in score_values.items() if values},
            "metadata": {
                "n_total": len(group),
                "n_scored": len(scored),
                **{f"n_{status.value}": sum(item.status == status for item in group) for status in ResultStatus},
            },
        }

    return SummaryResult(
        num_items=len(items),
        status_counts=dict(Counter(item.status.value for item in items)),
        results={key: aggregate_videos(group) for key, group in groups.items()},
        groups=[
            {"model_id": model, "category": category, "trajectory_id": trajectory, **aggregate_videos(group)}
            for (model, category, trajectory), group in detailed_groups.items()
        ],
    )
