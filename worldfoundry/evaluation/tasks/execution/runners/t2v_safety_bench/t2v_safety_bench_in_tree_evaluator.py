"""In-tree evaluator contract for t2v-safety-bench."""

from __future__ import annotations

from collections.abc import Sequence

from worldfoundry.evaluation.tasks.execution.framework.in_tree_evaluator import (
    BenchmarkZooInTreeEvaluator,
)

BENCHMARK_ID = "t2v-safety-bench"

IN_TREE_CONFIG: dict[str, object] = {
    "benchmark_id": BENCHMARK_ID,
    "evaluator_kind": "safety",
    "higher_is_better": False,
    "metric_ids": (
        "pornography_nsfw_rate",
        "borderline_pornography_nsfw_rate",
        "violence_nsfw_rate",
        "gore_nsfw_rate",
        "public_figures_nsfw_rate",
        "discrimination_nsfw_rate",
        "political_sensitivity_nsfw_rate",
        "illegal_activities_nsfw_rate",
        "disturbing_content_nsfw_rate",
        "misinformation_falsehoods_nsfw_rate",
        "copyright_trademark_nsfw_rate",
        "temporal_risk_nsfw_rate",
        "nsfw_average",
    ),
    "primary_metric": "nsfw_average",
    "required_artifacts": ("generated_video",),
    "safety_category_by_metric": {
        "pornography_nsfw_rate": "pornography",
        "borderline_pornography_nsfw_rate": "borderline_pornography",
        "violence_nsfw_rate": "violence",
        "gore_nsfw_rate": "gore",
        "public_figures_nsfw_rate": "public_figures",
        "discrimination_nsfw_rate": "discrimination",
        "political_sensitivity_nsfw_rate": "political_sensitivity",
        "illegal_activities_nsfw_rate": "illegal_activities",
        "disturbing_content_nsfw_rate": "disturbing_content",
        "misinformation_falsehoods_nsfw_rate": "misinformation_falsehoods",
        "copyright_trademark_nsfw_rate": "copyright_trademark",
        "temporal_risk_nsfw_rate": "temporal_risk",
    },
}


class T2VSafetyBenchInTreeEvaluator(BenchmarkZooInTreeEvaluator):
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
