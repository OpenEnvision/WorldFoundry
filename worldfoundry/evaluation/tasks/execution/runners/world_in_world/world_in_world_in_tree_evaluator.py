"""In-tree evaluator contract for world-in-world."""

from __future__ import annotations

from collections.abc import Sequence

from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
)

BENCHMARK_ID = "world-in-world"

IN_TREE_CONFIG: dict[str, object] = {
    "benchmark_id": BENCHMARK_ID,
    "evaluator_kind": "world_in_world",
    "metric_ids": (
        "active_recognition_success_rate",
        "image_goal_navigation_success_rate",
        "image_goal_navigation_spl",
        "active_embodied_qa_score",
        "active_embodied_qa_spl",
        "robotic_manipulation_success_rate",
        "interaction_trace_consistency",
        "world_in_world_average",
    ),
    "primary_metric": "world_in_world_average",
    "required_artifacts": ("generated_video", "interaction_trace"),
}


class WorldInWorldInTreeEvaluator(BenchmarkZooInTreeEvaluator):
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
