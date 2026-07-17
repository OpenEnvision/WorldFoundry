"""Merge embodied sharded rollout outputs into one scorecard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.tasks.embodied.metrics import metric_suite
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import (
    EVALUATE_RUN_RESULT_SCHEMA_VERSION,
    EvaluateRunResult,
)
from worldfoundry.evaluation.tasks.execution.orchestration.existing_results import run_existing_results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_shard_rows(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    candidates = sorted(output_dir.glob("shard*of*/results.jsonl"))
    if not candidates and (output_dir / "results.jsonl").exists():
        candidates = [output_dir / "results.jsonl"]
    for result_path in candidates:
        shard_dir = result_path.parent
        requests.extend(_read_jsonl(shard_dir / "requests.jsonl"))
        results.extend(_read_jsonl(result_path))
    if not results:
        raise FileNotFoundError(f"no embodied shard results found under {output_dir}")
    return requests, results


def _metric_callable(metric_ids: Sequence[str]):
    metrics = metric_suite(metric_ids, track="vla")

    def compute(request: GenerationRequest, result: GenerationResult) -> list[MetricResult]:
        return [metric.compute_sample(request, result) for metric in metrics]

    return compute


def merge_embodied_results(
    output_dir: str | Path,
    *,
    eval_id: str | None = None,
    metric_ids: Sequence[str] = ("task_success", "success_rate"),
    config: Mapping[str, Any] | None = None,
) -> EvaluateRunResult:
    """Merge sharded embodied outputs and write a root scorecard."""
    root = Path(output_dir).resolve()
    requests, results = _load_shard_rows(root)
    if not requests:
        requests = [
            {
                "sample_id": row.get("sample_id", f"sample-{index:04d}"),
                "task_name": ((row.get("metadata") or {}).get("task_spec") or {}).get("request_task_name", "embodied"),
            }
            for index, row in enumerate(results)
        ]
    model_cfg = dict((config or {}).get("model") or {})
    benchmark_ids = []
    for row in results:
        task_spec = ((row.get("metadata") or {}).get("task_spec") or {})
        suite = task_spec.get("suite")
        if suite:
            benchmark_ids.append(str(suite))
    benchmark_id = ",".join(sorted(set(benchmark_ids))) or str((config or {}).get("benchmark_id") or "embodied")
    merged = run_existing_results(
        output_dir=root,
        requests=requests,
        results=results,
        metric=_metric_callable(tuple(metric_ids)),
        benchmark={
            "suite": "vla_va_wam",
            "benchmark_name": str((config or {}).get("id") or "embodied_merged"),
            "benchmark_id": benchmark_id,
            "task_type": "embodied_closed_loop",
            "evaluation_protocol": "worldfoundry_embodied_merge",
            "official_runtime_executed": True,
            "normalizer_only": False,
            "integration_evidence": True,
        },
        model={
            "model_type": "embodied_policy",
            "model_id": str(model_cfg.get("id") or (config or {}).get("model_id") or ""),
            "model_name": str(model_cfg.get("id") or (config or {}).get("model_id") or ""),
        },
        dataset={
            "dataset_id": str((config or {}).get("id") or "embodied_merged"),
            "split": "closed_loop",
            "sample_count": len(requests),
        },
        run_id=eval_id,
        run_metadata={
            "schema_version": "worldfoundry-embodied-merge-run",
            "delegate_runner": "worldfoundry.embodied-merge",
        },
    )
    return EvaluateRunResult(
        schema_version=EVALUATE_RUN_RESULT_SCHEMA_VERSION,
        mode="existing-results",
        delegate_runner="worldfoundry.embodied-merge",
        status=merged.status,
        exit_code=merged.exit_code,
        output_dir=merged.output_dir,
        manifest_path=merged.manifest_path,
        execution_plan_path=merged.execution_plan_path,
        scorecard_path=merged.scorecard_path,
        sample_count=merged.sample_count,
        successful_sample_count=merged.successful_sample_count,
        failed_sample_count=merged.failed_sample_count,
        artifact_count=merged.artifact_count,
    )


__all__ = ["merge_embodied_results"]
