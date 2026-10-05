"""In-tree evaluator contract for worldarena."""

from __future__ import annotations

from collections.abc import Sequence

from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
)

BENCHMARK_ID = "worldarena"

IN_TREE_CONFIG: dict[str, object] = {
    "benchmark_id": BENCHMARK_ID,
    "evaluator_kind": "reasoning",
    "metric_ids": (
        "visual_quality",
        "motion_quality",
        "content_consistency",
        "physics_adherence",
        "three_d_accuracy",
        "controllability",
        "data_engine_success",
        "policy_evaluator_correlation",
        "action_planner_success",
        "human_quality",
        "ewm_score",
    ),
    "primary_metric": "ewm_score",
    "required_artifacts": ("generated_video",),
}


class WorldarenaInTreeEvaluator(BenchmarkZooInTreeEvaluator):
    def __init__(
        self,
        *,
        metric_ids: Sequence[str] | None = None,
        required_artifacts: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            BENCHMARK_ID,
            metric_ids=metric_ids,
            required_artifacts=required_artifacts,
        )
