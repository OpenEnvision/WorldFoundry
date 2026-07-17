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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from worldfoundry.evaluation.api import (
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    MetricResult,
    is_generation_result_successful,
)
from worldfoundry.evaluation.reporting import write_run_manifest_artifacts, write_run_report_artifacts
from worldfoundry.evaluation.reporting.scorecard import write_scorecard
from worldfoundry.evaluation.utils import (
    append_jsonl,
    build_run_fingerprint,
    build_version_context,
    jsonable,
    read_json_or_jsonl,
    reset_jsonl,
    write_json,
)

from .cache import generation_cache_hit_metadata

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


def _utcnow_iso() -> str:
    """Return UTC ISO8601 timestamp with ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_mapping(value: Any) -> JsonRow:
    """Transforms abstract variables into strict dictionaries suitable for JSON serialization."""
    json_value = jsonable(value)
    if isinstance(json_value, Mapping):
        return dict(json_value)
    return {"value": json_value}


def _coerce_optional_mapping(value: Mapping[str, Any] | Any | None, default: Mapping[str, Any]) -> JsonRow:
    """Safely coerces mapping inputs, applying a fallback dictionary if the input is None."""
    if value is None:
        return dict(default)
    return _coerce_mapping(value)


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


def _artifact_refs(value: Mapping[str, Any] | None) -> dict[str, ArtifactRef]:
    """Compiles generic artifact maps into structured ArtifactRef schema objects."""
    artifacts: dict[str, ArtifactRef] = {}
    for name, artifact in (value or {}).items():
        if isinstance(artifact, ArtifactRef):
            artifacts[str(name)] = artifact
        elif isinstance(artifact, Mapping):
            payload = dict(artifact)
            payload.setdefault("kind", str(name))
            artifacts[str(name)] = ArtifactRef.from_dict(payload)
        else:
            artifacts[str(name)] = ArtifactRef(uri=str(artifact), kind=str(name))
    return artifacts


def _coerce_generation_result(item: Any, sample_id: str) -> GenerationResult:
    """Coerces unstructured outcome dictionaries into strongly-typed GenerationResult instances.

    Safely packages unexpected keys into a trailing `metadata["extra"]` block so no
    data is inadvertently dropped from downstream metrics.
    """
    if isinstance(item, GenerationResult):
        if item.sample_id == sample_id:
            return item
        return GenerationResult(
            sample_id=sample_id,
            request_id=item.request_id,
            model_id=item.model_id,
            artifacts=item.artifacts,
            status=item.status,
            error=item.error,
            timings=item.timings,
            metadata={**dict(item.metadata), "source_sample_id": item.sample_id},
        )

    row = _coerce_mapping(item)
    row.setdefault("sample_id", sample_id)
    known_keys = {
        "sample_id",
        "request_id",
        "model_id",
        "artifacts",
        "status",
        "error",
        "timings",
        "metadata",
        "schema_version",
    }
    metadata = dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), Mapping) else {}
    extras = {key: value for key, value in row.items() if key not in known_keys}
    if extras:
        metadata.setdefault("extra", extras)
    return GenerationResult(
        sample_id=str(row["sample_id"]),
        request_id=row.get("request_id"),
        model_id=str(row.get("model_id") or ""),
        artifacts=_artifact_refs(row.get("artifacts")),
        status=str(row.get("status") or ("failed" if row.get("error") else "succeeded")),
        error=row.get("error"),
        timings=row.get("timings") or {},
        metadata=metadata,
    )


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
            aligned.append(_coerce_generation_result(item, request.sample_id))
        elif sequential_index < len(sequential):
            item = sequential[sequential_index]
            sequential_index += 1
            aligned.append(_coerce_generation_result(item, request.sample_id))
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


def _metric_value(result: MetricResult) -> Any:
    """Extracts either normalized value or fallback raw value from metric result DTOs."""
    if result.normalized_value is not None:
        return result.normalized_value
    return result.raw_value


def _metric_result_from_mapping(row: Mapping[str, Any], sample_id: str) -> list[MetricResult]:
    """Constructs MetricResult list from general dictionary shapes."""
    if "metric_id" in row:
        payload = dict(row)
        payload.setdefault("sample_id", sample_id)
        return [MetricResult.from_dict(payload)]

    metric_values = row.get("metrics")
    if isinstance(metric_values, Mapping):
        return [
            MetricResult(
                sample_id=sample_id,
                metric_id=str(metric_id),
                raw_value=value,
                normalized_value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            )
            for metric_id, value in metric_values.items()
        ]

    excluded = {"sample_id", "id", "status", "success", "error", "exception", "message"}
    results = []
    for metric_id, value in row.items():
        if metric_id in excluded:
            continue
        results.append(
            MetricResult(
                sample_id=sample_id,
                metric_id=str(metric_id),
                raw_value=value,
                normalized_value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            )
        )
    if results:
        return results
    raise ValueError("metric dict did not contain metric_id or metric values")


def _normalize_metric_output(output: Any, sample_id: str) -> list[MetricResult]:
    """Recursively normalizes polymorphic metric outputs into structured lists of MetricResult DTOs."""
    if output is None:
        return []
    if isinstance(output, MetricResult):
        return [output if output.sample_id == sample_id else MetricResult(**{**output.to_dict(), "sample_id": sample_id})]
    if isinstance(output, Mapping):
        return _metric_result_from_mapping(output, sample_id)
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
        results: list[MetricResult] = []
        for item in output:
            results.extend(_normalize_metric_output(item, sample_id))
        return results
    raise TypeError(f"unsupported metric output for sample {sample_id}: {type(output).__name__}")


def _run_metric(
    metric: ExistingResultsMetric | None,
    request: GenerationRequest,
    result: GenerationResult,
) -> tuple[JsonRow, list[MetricResult]]:
    """Calculates metrics for a single sample while protecting execution from unhandled exceptions."""
    if _is_failed(result):
        return (
            {
                "sample_id": request.sample_id,
                "status": "skipped",
                "metrics": {},
                "error": result.error or "generation failed",
            },
            [],
        )
    if metric is None:
        return {"sample_id": request.sample_id, "status": "not_run", "metrics": {}}, []

    try:
        metric_results = _normalize_metric_output(metric(request, result), request.sample_id)
    except Exception as exc:  # noqa: BLE001 - per-sample metric isolation is the contract.
        return (
            {
                "sample_id": request.sample_id,
                "status": "failed",
                "metrics": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
            [],
        )

    metrics = {
        metric_result.metric_id: _metric_value(metric_result)
        for metric_result in metric_results
        if metric_result.valid and _metric_value(metric_result) is not None
    }
    return (
        {
            "sample_id": request.sample_id,
            "status": "succeeded",
            "metrics": metrics,
            "metric_results": [metric_result.to_dict() for metric_result in metric_results],
        },
        metric_results,
    )


def _build_summary(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    per_sample_rows: Sequence[Mapping[str, Any]],
    metric_results: Sequence[MetricResult],
    metric_enabled: bool,
) -> dict[str, Any]:
    """Compiles statistics, failure identifiers, and aggregations across the entire offline batch."""
    sample_count = len(requests)
    generation_failed_ids = [result.sample_id for result in results if _is_failed(result)]
    metric_failed_ids = [
        str(row["sample_id"])
        for row in per_sample_rows
        if str(row.get("status") or "").lower() in {"failed", "failure", "error", "errored"}
    ]
    metric_skipped_ids = [
        str(row["sample_id"])
        for row in per_sample_rows
        if str(row.get("status") or "").lower() == "skipped"
    ]
    failed_ids = sorted(set(generation_failed_ids).union(metric_failed_ids).union(metric_skipped_ids))
    successful_samples = sample_count - len(failed_ids)

    values: dict[str, list[float]] = {}
    for metric_result in metric_results:
        value = _metric_value(metric_result)
        if metric_result.valid and isinstance(value, (int, float)) and not isinstance(value, bool):
            values.setdefault(metric_result.metric_id, []).append(float(value))

    per_metric = {
        metric_id: {
            "mean": sum(metric_values) / len(metric_values),
            "sample_count": len(metric_values),
            "higher_is_better": True,
        }
        for metric_id, metric_values in sorted(values.items())
        if metric_values
    }
    leaderboard = {metric_id: payload["mean"] for metric_id, payload in per_metric.items()}

    return {
        "schema_version": "worldfoundry-metrics-summary",
        "sample_count": sample_count,
        "successful_samples": successful_samples,
        "failed_samples": len(failed_ids),
        "failed_sample_ids": failed_ids,
        "generation": {
            "successful": sample_count - len(generation_failed_ids),
            "failed": len(generation_failed_ids),
            "failed_sample_ids": generation_failed_ids,
        },
        "metrics": {
            "enabled": metric_enabled,
            "successful": sum(1 for row in per_sample_rows if str(row.get("status") or "").lower() == "succeeded"),
            "failed": len(metric_failed_ids),
            "failed_sample_ids": metric_failed_ids,
            "skipped": len(metric_skipped_ids),
            "skipped_sample_ids": metric_skipped_ids,
        },
        "leaderboard": leaderboard,
        "per_metric": per_metric,
        "groups": {},
    }


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


def _artifact_paths(output_dir: Path, artifact_count: int) -> dict[str, str]:
    """Defines and resolves standard file paths for all scorecard and manifest outputs of this run."""
    paths = {
        "run_manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample_metrics": output_dir / "metrics" / "per_sample.jsonl",
        "summary": output_dir / "metrics" / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }
    if artifact_count:
        paths["artifacts"] = output_dir / "artifacts.jsonl"
    return {name: str(path.resolve()) for name, path in paths.items()}


def _coerce_request(
    request: ExistingResultsRunRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> ExistingResultsRunRequest:
    """Coerces input parameters into a standard, fully populated ExistingResultsRunRequest DTO."""
    if isinstance(request, ExistingResultsRunRequest):
        if kwargs:
            payload = asdict(request)
            payload.update(kwargs)
            return ExistingResultsRunRequest(**payload)
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
    output_dir = Path(run_request.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_request.run_id or f"existing-results-{uuid4().hex[:12]}"
    started_at = _utcnow_iso()
    requests = _normalize_requests(run_request.requests)
    results = _align_results(requests, run_request.results)
    artifact_rows = _artifact_rows(results)

    paths = {
        "manifest": output_dir / "run_manifest.json",
        "environment": output_dir / "environment.json",
        "env_requirements": output_dir / "env_requirements.json",
        "execution_plan": output_dir / "execution_plan.json",
        "requests": output_dir / "requests.jsonl",
        "results": output_dir / "results.jsonl",
        "artifacts": output_dir / "artifacts.jsonl",
        "sample_ledger": output_dir / "sample_ledger.jsonl",
        "per_sample": metrics_dir / "per_sample.jsonl",
        "summary": metrics_dir / "summary.json",
        "run_summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "scorecard": output_dir / "scorecard.json",
    }
    for key in ("requests", "results", "artifacts", "sample_ledger", "per_sample"):
        reset_jsonl(paths[key])

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
        runner="existing_results_runner",
        benchmark=benchmark,
        model=model,
        dataset=dataset,
        metric=run_request.metric,
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
        "runner": "existing_results_runner",
        "version_context": version_context,
        "run_fingerprint": run_fingerprint,
        "stages": ["materialize", "normalize_existing_results", "evaluate", "aggregate", "report"],
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
    write_json(paths["execution_plan"], plan)

    initial_manifest = {
        "schema_version": "worldfoundry-run-manifest",
        "run_id": run_id,
        "runner": "existing_results_runner",
        "status": "running",
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
    write_run_manifest_artifacts(
        output_dir=output_dir,
        base_manifest=initial_manifest,
        config={
            "runner": "existing_results_runner",
            "fail_on_sample_error": run_request.fail_on_sample_error,
            "write_artifacts_index": run_request.write_artifacts_index,
        },
        cache_paths=run_request.cache_paths or {},
        package_names=("worldfoundry", "numpy", "pandas"),
        manifest_path=paths["manifest"],
        environment_path=paths["environment"],
        env_requirements_path=paths["env_requirements"],
    )

    # Materialize requests/results JSONL and per-sample metric rows.
    for request_row in requests:
        append_jsonl(paths["requests"], request_row.to_dict())
    for result in results:
        append_jsonl(paths["results"], result.to_dict())
    if artifact_rows and run_request.write_artifacts_index:
        for row in artifact_rows:
            append_jsonl(paths["artifacts"], row)

    per_sample_rows: list[JsonRow] = []
    all_metric_results: list[MetricResult] = []
    
    # Per-sample metric loop (isolated failures unless fail_on_sample_error).
    for index, request_row in enumerate(requests):
        result = results[index]
        generation_status = "failed" if _is_failed(result) else "succeeded"
        generation_cache_metadata = generation_cache_hit_metadata(result)
        
        # Metric callable (skipped when generation result failed).
        metric_row, metric_results = _run_metric(run_request.metric, request_row, result)
        metric_status = str(metric_row.get("status") or "succeeded").lower()
        per_sample_rows.append(metric_row)
        all_metric_results.extend(metric_results)

        errors = []
        if generation_status == "failed":
            errors.append({"stage": "normalize_existing_results", "message": result.error or "generation failed"})
        if metric_status in {"failed", "failure", "error", "errored", "skipped"}:
            errors.append({"stage": "evaluate", "message": str(metric_row.get("error") or "metric failed")})

        ledger_status = "failed" if errors else "succeeded"
        append_jsonl(
            paths["sample_ledger"],
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
            },
        )
        append_jsonl(paths["per_sample"], metric_row)

    summary = _build_summary(
        requests=requests,
        results=results,
        per_sample_rows=per_sample_rows,
        metric_results=all_metric_results,
        metric_enabled=run_request.metric is not None,
    )
    write_json(paths["summary"], summary)

    failed_sample_count = int(summary["failed_samples"])
    status = "completed_with_failures" if failed_sample_count else "succeeded"
    exit_code = 1 if failed_sample_count and run_request.fail_on_sample_error else 0
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
            "started_at": started_at,
            "finished_at": finished_at,
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
            "runner": "existing_results_runner",
            "fail_on_sample_error": run_request.fail_on_sample_error,
            "write_artifacts_index": run_request.write_artifacts_index,
        },
        cache_paths=run_request.cache_paths or {},
        package_names=("worldfoundry", "numpy", "pandas"),
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
