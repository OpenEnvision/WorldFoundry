"""Offline evaluation of pre-existing generation results.

Score cached ``GenerationResult`` rows without re-running GPU inference.  Aligns
requests to stored outputs, runs optional metric callables, and writes
``scorecard.json`` plus run manifests.

Sections:

* **DTOs** — :class:`ExistingResultsRunRequest` / :class:`ExistingResultsRunResult`.
* **Coercion** — request/result alignment and metric normalization helpers.
* **Orchestration** — :func:`run_existing_results` batch loop and aggregation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from worldfoundry.core.io.formats.serialization import JsonlWriter
from worldfoundry.evaluation.api import (
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
)
from worldfoundry.evaluation.reporting import (
    REPRODUCIBILITY_PACKAGES,
    write_run_manifest_artifacts,
    write_run_report_artifacts,
)
from worldfoundry.evaluation.reporting.scorecard import write_scorecard
from worldfoundry.evaluation.tasks.metrics.registry import BuiltinExistingResultsMetric
from worldfoundry.evaluation.utils import (
    build_run_fingerprint,
    build_version_context,
    read_json_or_jsonl,
    write_json,
    write_jsonl,
)

from .cache import generation_cache_hit_metadata
from .run_session import (
    artifact_report_paths as _artifact_paths,
)
from .run_session import (
    coerce_generation_result,
    prepare_run_output_dir,
    reset_session_jsonl,
    run_stage,
    session_paths,
)
from .run_session import (
    coerce_mapping as _coerce_mapping,
)
from .run_session import (
    coerce_optional_mapping as _coerce_optional_mapping,
)
from .run_session import (
    utcnow_iso as _utcnow_iso,
)
from .scoring import build_metrics_summary, metric_owners, score_samples

# ---------------------------------------------------------------------------
# Types and request DTOs
# ---------------------------------------------------------------------------

JsonRow = dict[str, Any]


ExistingResultsMetric = Callable[[GenerationRequest, GenerationResult], Any]


@dataclass(frozen=True)
class ExistingResultsRunRequest:
    """Inputs for an offline existing-results evaluation run."""
    output_dir: str | Path
    requests: Sequence[Any]
    results: Any
    metric: ExistingResultsMetric | None = None
    metrics: Sequence[Any] = ()
    resume: bool = False
    metric_batch_size: int = 32
    benchmark: Mapping[str, Any] | Any | None = None
    model: Mapping[str, Any] | Any | None = None
    dataset: Mapping[str, Any] | Any | None = None
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    run_metadata: Mapping[str, Any] | None = None
    cache_paths: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExistingResultsRunResult:
    """Summary paths and counts after ``run_existing_results`` completes."""
    status: str
    exit_code: int
    output_dir: Path
    manifest_path: Path
    execution_plan_path: Path
    scorecard_path: Path
    sample_count: int
    successful_sample_count: int
    failed_sample_count: int
    artifact_count: int


class ExistingResultsRunner:
    """Thin wrapper delegating to :func:`run_existing_results`."""

    def run(
        self,
        request: ExistingResultsRunRequest | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExistingResultsRunResult:
        """Delegate to :func:`run_existing_results`."""
        return run_existing_results(request, **kwargs)


# ---------------------------------------------------------------------------
# Coercion and alignment helpers
# ---------------------------------------------------------------------------


def _sample_id_from(value: Any) -> str | None:
    """Discovers a reasonable `sample_id` from generic dictionaries or namespace objects."""
    if isinstance(value, Mapping):
        for key in ("sample_id", "id"):
            item = value.get(key)
            if item is not None:
                return str(item)
    for attr in ("sample_id", "id"):
        if hasattr(value, attr):
            item = getattr(value, attr)
            if item is not None:
                return str(item)
    return None


def _normalize_requests(source: Sequence[Any]) -> list[GenerationRequest]:
    """Force-casts a batch of abstract objects into rigid `GenerationRequest` contracts."""
    rows: list[GenerationRequest] = []
    for index, item in enumerate(source):
        if isinstance(item, GenerationRequest):
            request = item
        elif isinstance(item, Mapping):
            row = dict(item)
            row.setdefault("sample_id", _sample_id_from(row) or f"sample-{index:04d}")
            row.setdefault("task_name", "existing_results")
            request = GenerationRequest.from_dict(row)
        else:
            row = _coerce_mapping(item)
            row.setdefault("sample_id", _sample_id_from(item) or f"sample-{index:04d}")
            row.setdefault("task_name", "existing_results")
            request = GenerationRequest.from_dict(row)
        rows.append(request)
    return rows


def _looks_like_result_row(value: Mapping[str, Any]) -> bool:
    """Heuristic logic testing if a generic map aligns structurally with the GenerationResult contract."""
    keys = {"sample_id", "request_id", "model_id", "artifacts", "status", "error", "timings", "metadata"}
    return bool(keys.intersection(value.keys()))


def _load_result_source(source: Any) -> list[Any]:
    """Dynamically parses result source descriptors (raw payloads, JSON paths, nested structures)."""
    if isinstance(source, (str, Path)):
        text_source = str(source)
        stripped = text_source.lstrip()
        if isinstance(source, Path) or not stripped.startswith(("{", "[")):
            path = Path(source)
            if path.exists():
                return _load_result_source(read_json_or_jsonl(path))
            if path.suffix.lower() in {".json", ".jsonl"}:
                raise FileNotFoundError(f"Existing results file not found: {path}")
        if stripped.startswith(("{", "[")):
            return _load_result_source(json.loads(text_source))

    if isinstance(source, GenerationResult):
        return [source]

    if isinstance(source, Mapping):
        if "results" in source and isinstance(source["results"], Iterable):
            return list(source["results"])
        if _looks_like_result_row(source):
            return [source]
        rows = []
        for sample_id, value in source.items():
            row = _coerce_mapping(value)
            row.setdefault("sample_id", str(sample_id))
            rows.append(row)
        return rows

    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        return list(source)

    raise TypeError("existing results must be GenerationResult, mapping, sequence, JSON string, or JSON/JSONL path")


def _align_results(requests: Sequence[GenerationRequest], source: Any) -> list[GenerationResult]:
    """Matches offline execution results against the structured generation requests batch.

    If sample_id keys match, results are matched directly.
    If sample_id strings are missing, attempts a fallback positional zip.
    """
    raw_rows = _load_result_source(source)
    keyed: dict[str, Any] = {}
    sequential: list[Any] = []
    for item in raw_rows:
        sample_id = _sample_id_from(item)
        if sample_id is None:
            sequential.append(item)
        else:
            keyed[sample_id] = item

    aligned = []
    sequential_index = 0
    for request in requests:
        if request.sample_id in keyed:
            item = keyed[request.sample_id]
            aligned.append(coerce_generation_result(item, request.sample_id))
        elif sequential_index < len(sequential):
            item = sequential[sequential_index]
            sequential_index += 1
            aligned.append(coerce_generation_result(item, request.sample_id))
        else:
            aligned.append(
                GenerationResult(
                    sample_id=request.sample_id,
                    request_id=request.request_id,
                    status="failed",
                    error="missing existing result",
                )
            )
    return aligned


def _is_failed(result: GenerationResult) -> bool:
    """Helper confirming if a generation task indicates failure status."""
    return not is_generation_result_successful(result)


def _artifact_rows(results: Sequence[GenerationResult]) -> list[JsonRow]:
    """Compiles a flat list of output artifact record rows from the generation result batch."""
    rows: list[JsonRow] = []
    for result in results:
        for name, artifact in result.artifacts.items():
            row = artifact.to_dict()
            row.setdefault("sample_id", result.sample_id)
            row.setdefault("name", name)
            rows.append(row)
    return rows


def _coerce_request(
    request: ExistingResultsRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> ExistingResultsRunRequest:
    """Coerces input parameters into a standard, fully populated ExistingResultsRunRequest DTO."""
    if isinstance(request, ExistingResultsRunRequest):
        if kwargs:
            return replace(request, **kwargs)
        return request

    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    if "output_dir" not in payload:
        raise TypeError("run_existing_results requires an output_dir")
    if "requests" not in payload:
        raise TypeError("run_existing_results requires requests")
    if "results" not in payload:
        raise TypeError("run_existing_results requires results")
    return ExistingResultsRunRequest(**payload)


def run_existing_results(
    request: ExistingResultsRunRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ExistingResultsRunResult:
    """Orchestrate offline metric evaluation over aligned request/result pairs."""
    run_request = _coerce_request(request, kwargs)
    if run_request.metric_batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    metrics = tuple(run_request.metrics) + ((run_request.metric,) if run_request.metric is not None else ())
    output_dir = prepare_run_output_dir(run_request.output_dir)
    runner_name = (run_request.run_metadata or {}).get("runner", "existing_results_runner")

    run_id = run_request.run_id or f"existing-results-{uuid4().hex[:12]}"
    started_at = (run_request.run_metadata or {}).get("started_at") or _utcnow_iso()
    requests = _normalize_requests(run_request.requests)
    results = _align_results(requests, run_request.results)
    metrics = tuple(
        metric.resolve_numeric_fields(results) if isinstance(metric, BuiltinExistingResultsMetric) else metric
        for metric in metrics
    )
    metric_owners(metrics)
    artifact_rows = _artifact_rows(results)
    paths = session_paths(output_dir)

    benchmark = _coerce_optional_mapping(
        run_request.benchmark,
        {
            "suite": "existing_results",
            "benchmark_name": "existing_results",
            "task_type": "existing_results",
            "evaluation_protocol": "existing_results",
        },
    )
    model = _coerce_optional_mapping(run_request.model, {"model_type": "existing", "model_name": "existing-results"})
    dataset = _coerce_optional_mapping(run_request.dataset, {"split": "existing"})
    dataset.setdefault("sample_count", len(requests))
    
    # Fingerprint from version context + aligned inputs (no GPU generation).
    version_context = build_version_context(
        runner=runner_name,
        benchmark=benchmark,
        model=model,
        dataset=dataset,
        metrics=metrics,
        extra={"mode": "existing-results"},
    )
    run_fingerprint = build_run_fingerprint(
        version_context=version_context,
        requests=requests,
        results=results,
    )

    plan = {
        "schema_version": "worldfoundry-execution-plan",
        "run_id": run_id,
        "created_at": started_at,
        "runner": runner_name,
        "version_context": version_context,
        "run_fingerprint": run_fingerprint,
        "stages": ["materialize", "generate" if (run_request.run_metadata or {}).get("generation_identity")
                   else "normalize_existing_results", "evaluate", "aggregate", "report"],
        "sample_count": len(requests),
        "samples": [
            {"index": index, "sample_id": request.sample_id, "status": "planned"}
            for index, request in enumerate(requests)
        ],
        "outputs": {
            "requests": "requests.jsonl",
            "results": "results.jsonl",
            "artifacts": "artifacts.jsonl" if artifact_rows else None,
            "sample_ledger": "sample_ledger.jsonl",
            "per_sample_metrics": "metrics/per_sample.jsonl",
            "summary": "metrics/summary.json",
            "run_summary": "summary.json",
            "report": "report.md",
            "scorecard": "scorecard.json",
        },
    }

    initial_manifest = {
        "schema_version": "worldfoundry-run-manifest",
        "run_id": run_id,
        "runner": runner_name,
        "status": "running",
        "stage": "evaluate",
        "started_at": started_at,
        "output_dir": str(output_dir),
        "benchmark": benchmark,
        "model": model,
        "dataset": dataset,
        "version_context": version_context,
        "run_fingerprint": run_fingerprint,
        "sample_count": len(requests),
        "execution_plan": str(paths["execution_plan"].resolve()),
        **dict(run_request.run_metadata or {}),
    }
    with run_stage(paths, "evaluate", manifest=initial_manifest):
        reset_session_jsonl(paths)
        write_json(paths["execution_plan"], plan)
        write_run_manifest_artifacts(
            output_dir=output_dir,
            base_manifest=initial_manifest,
            config={
                "runner": runner_name,
                "fail_on_sample_error": run_request.fail_on_sample_error,
                "write_artifacts_index": run_request.write_artifacts_index,
                "resume": run_request.resume,
                "metric_batch_size": run_request.metric_batch_size,
            },
            cache_paths={**dict(run_request.cache_paths or {}), "metric_checkpoint": str(output_dir / "metrics" / "checkpoint.jsonl")},
            package_names=REPRODUCIBILITY_PACKAGES,
            manifest_path=paths["manifest"],
            environment_path=paths["environment"],
            env_requirements_path=paths["env_requirements"],
        )

        # Materialize immutable stage outputs with one CPFS open/close cycle each.
        write_jsonl(paths["requests"], (request_row.to_dict() for request_row in requests), atomic=False)
        write_jsonl(paths["results"], (result.to_dict() for result in results), atomic=False)
        if artifact_rows and run_request.write_artifacts_index:
            write_jsonl(paths["artifacts"], artifact_rows, atomic=False)

        per_sample_rows: list[JsonRow] = []
        metric_batches = score_samples(
            requests, results, metrics, checkpoint_path=output_dir / "metrics" / "checkpoint.jsonl",
            resume=run_request.resume, batch_size=run_request.metric_batch_size,
        )
    
        # Per-sample metric loop (isolated failures unless fail_on_sample_error).
        # Periodic flushes retain resumable progress without reopening CPFS files
        # for every sample.
        with (
            JsonlWriter(paths["sample_ledger"]) as ledger_writer,
            JsonlWriter(paths["per_sample"]) as per_sample_writer,
        ):
            index = 0
            for batch in metric_batches:
                for metric_row in batch:
                    request_row = requests[index]
                    result = results[index]
                    generation_status = "failed" if _is_failed(result) else "succeeded"
                    generation_cache_metadata = generation_cache_hit_metadata(result)

                    # Metric callable (skipped when generation result failed).
                    metric_status = str(metric_row.get("status") or "succeeded").lower()
                    per_sample_rows.append(metric_row)

                    errors = []
                    if generation_status == "failed":
                        errors.append({"stage": "normalize_existing_results", "message": result.error or "generation failed"})
                    if metric_status == "failed":
                        errors.append({"stage": "evaluate", "message": str(metric_row.get("error") or "metric failed")})

                    ledger_status = "failed" if errors else ("skipped" if metric_status == "skipped" else "succeeded")
                    ledger_writer.write(
                        {
                            "run_id": run_id,
                            "sample_id": request_row.sample_id,
                            "index": index,
                            "status": ledger_status,
                            "generation_status": generation_status,
                            "metrics_status": metric_status,
                            "started_at": started_at,
                            "finished_at": _utcnow_iso(),
                            **(
                                {
                                    "cached": True,
                                    "cache_source": "generation_result_cache",
                                    "source_run_id": generation_cache_metadata.get("source_run_id"),
                                    "cache_key_hash": generation_cache_metadata.get("key_hash"),
                                }
                                if generation_cache_metadata
                                else {}
                            ),
                            **({"errors": errors} if errors else {}),
                        }
                    )
                    per_sample_writer.write(metric_row)
                    index += 1
                ledger_writer.flush()
                per_sample_writer.flush()

    with run_stage(paths, "aggregate"):
        summary = build_metrics_summary(requests, results, per_sample_rows, metrics)
        write_json(paths["summary"], summary)

    with run_stage(paths, "report"):
        failed_sample_count = int(summary["failed_samples"])
        scoring_failed = failed_sample_count > 0 or any(not row["valid"] for row in summary["per_metric"].values())
        status = "completed_with_failures" if scoring_failed else "succeeded"
        exit_code = 1 if scoring_failed and run_request.fail_on_sample_error else 0
        finished_at = _utcnow_iso()
        artifact_count = len(artifact_rows) if run_request.write_artifacts_index else 0
        artifact_paths = _artifact_paths(output_dir, artifact_count)
        generation = {
            "num_requests": len(requests),
            "successful": int(summary["generation"]["successful"]),
            "failed": int(summary["generation"]["failed"]),
            "error_sample_ids": list(summary["generation"]["failed_sample_ids"]),
            "throughput": {},
        }
        skipped = {
            "count": int(summary["metrics"]["skipped"]),
            "sample_ids": list(summary["metrics"]["skipped_sample_ids"]),
        }
        write_scorecard(
            paths["scorecard"],
            run={
                "run_id": run_id,
                "status": status,
                "stage": "report",
                "started_at": started_at,
                "finished_at": finished_at,
                "generation_cache": dict((run_request.run_metadata or {}).get("generation_cache") or {}),
                "version_context": version_context,
                "run_fingerprint": run_fingerprint,
            },
            benchmark=benchmark,
            model=model,
            dataset=dataset,
            generation=generation,
            metrics_summary=summary,
            artifacts=artifact_paths,
            skipped=skipped,
            provenance=(run_request.run_metadata or {}).get("evaluation_provenance"),
            evaluation_kind=(run_request.run_metadata or {}).get("evaluation_kind", "existing_results"),
        )
        write_run_report_artifacts(
            output_dir=output_dir,
            scorecard_path=paths["scorecard"],
            summary_path=paths["run_summary"],
            report_path=paths["report"],
        )

        final_manifest = dict(initial_manifest)
        final_manifest.update(
            {
                "status": status,
                "stage": "report",
                "finished_at": finished_at,
                "exit_code": exit_code,
                "successful_sample_count": int(summary["successful_samples"]),
                "failed_sample_count": failed_sample_count,
                "artifacts": artifact_paths,
                **dict(run_request.run_metadata or {}),
            }
        )
        write_run_manifest_artifacts(
            output_dir=output_dir,
            base_manifest=final_manifest,
            config={
                "runner": runner_name,
                "fail_on_sample_error": run_request.fail_on_sample_error,
                "write_artifacts_index": run_request.write_artifacts_index,
                "resume": run_request.resume,
                "metric_batch_size": run_request.metric_batch_size,
            },
            cache_paths={**dict(run_request.cache_paths or {}), "metric_checkpoint": str(output_dir / "metrics" / "checkpoint.jsonl")},
            package_names=REPRODUCIBILITY_PACKAGES,
            manifest_path=paths["manifest"],
            environment_path=paths["environment"],
            env_requirements_path=paths["env_requirements"],
        )

        return ExistingResultsRunResult(
            status=status,
            exit_code=exit_code,
            output_dir=output_dir,
            manifest_path=paths["manifest"].resolve(),
            execution_plan_path=paths["execution_plan"].resolve(),
            scorecard_path=paths["scorecard"].resolve(),
            sample_count=len(requests),
            successful_sample_count=int(summary["successful_samples"]),
            failed_sample_count=failed_sample_count,
            artifact_count=artifact_count,
        )


execute_existing_results = run_existing_results


__all__ = [
    "ExistingResultsMetric",
    "ExistingResultsRunRequest",
    "ExistingResultsRunResult",
    "ExistingResultsRunner",
    "execute_existing_results",
    "run_existing_results",
]
