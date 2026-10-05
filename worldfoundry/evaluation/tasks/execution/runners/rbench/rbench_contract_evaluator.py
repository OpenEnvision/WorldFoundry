"""Benchmark-zoo contract evaluator for RBench."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.execution.framework.video_contract_evaluator import (
    write_video_contract_evaluation,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_official_filter import (
    BENCHMARK_ID,
    OFFICIAL_REQUIREMENTS,
    filter_official_records,
)
from worldfoundry.evaluation.tasks.execution.runners.rbench.rbench_official_scoring import (
    official_scores_from_records,
)

EVALUATOR_KIND = "in_tree_rbench_contract_evaluator"


def write_rbench_evaluation(
    *,
    display_name: str,
    official_metric_ids: tuple[str, ...],
    output_dir: str | Path,
    generated_artifact_dir: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    runner: str = "benchmark_zoo_contract_evaluator",
    mode: str = "contract",
    **kwargs: Any,
) -> dict[str, Any]:
    return write_video_contract_evaluation(
        benchmark_id=BENCHMARK_ID,
        display_name=display_name,
        official_metric_ids=official_metric_ids,
        output_dir=output_dir,
        official_requirements={BENCHMARK_ID: OFFICIAL_REQUIREMENTS},
        compute_official_scores=official_scores_from_records,
        filter_official_records=filter_official_records,
        evaluator_kind=EVALUATOR_KIND,
        prompt_manifest_keys=("rbench_prompt_manifest",),
        generated_artifact_dir=generated_artifact_dir,
        manifest=manifest,
        runner=runner,
        mode=mode,
        **kwargs,
    )


__all__ = ["EVALUATOR_KIND", "write_rbench_evaluation"]
