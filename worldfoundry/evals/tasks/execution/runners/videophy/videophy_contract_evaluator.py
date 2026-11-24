"""Benchmark-zoo contract evaluator for VideoPhy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.execution.framework.video_contract_evaluator import (
    write_video_contract_evaluation,
)
from worldfoundry.evals.tasks.execution.runners.videophy.videophy_official_scoring import (
    BENCHMARK_ID,
    OFFICIAL_REQUIREMENTS,
    official_scores_from_records,
)

EVALUATOR_KIND = "in_tree_videophy_contract_evaluator"


def write_videophy_evaluation(
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
        evaluator_kind=EVALUATOR_KIND,
        prompt_manifest_keys=("videophy_prompt_manifest",),
        generated_artifact_dir=generated_artifact_dir,
        manifest=manifest,
        runner=runner,
        mode=mode,
        **kwargs,
    )
