"""In-tree evaluator contract for ewmbench."""

from __future__ import annotations

from collections.abc import Sequence

from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
)

BENCHMARK_ID = "ewmbench"

IN_TREE_CONFIG: dict[str, object] = {
    "benchmark_id": BENCHMARK_ID,
    "evaluator_kind": "reasoning",
    "metric_ids": (
        "scene_consistency",
        "motion_correctness",
        "semantic_alignment",
        "diversity",
        "ewmbench_average",
    ),
    "primary_metric": "ewmbench_average",
    "required_artifacts": ("generated_video",),
}


class EwmbenchInTreeEvaluator(BenchmarkZooInTreeEvaluator):
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
