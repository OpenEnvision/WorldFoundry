"""Shared sample scoring, progress records, and metric aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.formats.serialization import JsonlWriter, iter_jsonl
from worldfoundry.evaluation.api import (
    AggregateResult,
    GenerationRequest,
    GenerationResult,
    MetricResult,
    align_batch_metric_outputs,
    is_generation_result_successful,
)
from worldfoundry.evaluation.api.metrics import aggregate_mean
from worldfoundry.evaluation.reporting.run_manifest import redact_secrets
from worldfoundry.evaluation.utils import jsonable, metric_fingerprint

from .run_session import artifacts_available


def metric_name(metric: Any) -> str:
    return str(getattr(metric, "name", None) or getattr(metric, "__name__", None) or type(metric).__name__)


def metric_ids(metric: Any) -> tuple[str, ...]:
    ids = getattr(metric, "metric_ids", None)
    if ids is not None:
        return tuple(ids)
    return (str(metric.name),) if getattr(metric, "name", None) else ()


def metric_owners(metrics: Sequence[Any]) -> dict[str, Any]:
    """Require one executor per declared metric id."""
    owners = {}
    for metric in metrics:
        for key in metric_ids(metric):
            if key in owners:
                raise ValueError(f"duplicate metric id {key!r}; select a single metric implementation")
            owners[key] = metric
    return owners


def metric_identity(metric: Any, metric_id: str | None = None) -> dict[str, Any] | None:
    """Versioned metrics declare scoring parameters separately from runtime state."""
    if not getattr(metric, "version", None):
        return None
    definition = metric_fingerprint(metric)
    if metric_id is not None:
        definition.pop("metric_ids", None)
        definition["name"] = metric_id
        describe = getattr(metric, "metric_definition", None)
        if callable(describe):
            definition.update(describe(metric_id))
    return redact_secrets(jsonable(definition))


def metric_value(result: MetricResult) -> Any:
    return result.normalized_value if result.normalized_value is not None else result.raw_value


def normalize_metric_output(output: Any, sample_id: str) -> list[MetricResult]:
    if output is None:
        return []
    if isinstance(output, MetricResult):
        return [MetricResult.from_dict({**output.to_dict(), "sample_id": sample_id})]
    if isinstance(output, Mapping):
        if "metric_id" in output:
            return [MetricResult.from_dict({**output, "sample_id": sample_id})]
        values = output.get("metrics", output)
        excluded = {"sample_id", "id", "status", "success", "error", "exception", "message"}
        return [
            MetricResult(sample_id=sample_id, metric_id=str(key), raw_value=value)
            for key, value in values.items()
            if key not in excluded
        ]
    if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
        return [row for item in output for row in normalize_metric_output(item, sample_id)]
    raise TypeError(f"unsupported metric output for sample {sample_id}: {type(output).__name__}")


def score_samples(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    metrics: Sequence[Any],
    *,
    checkpoint_path: Path,
    resume: bool,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield scored batches, persisting each metric before starting the next one."""
    if batch_size < 1:
        raise ValueError("metric_batch_size must be positive")
    previous = {}
    if resume and checkpoint_path.exists():
        for row in iter_jsonl(checkpoint_path):
            previous[(row["sample_id"], row["metric"])] = row
    identities = [metric_identity(metric) for metric in metrics]
    with JsonlWriter(checkpoint_path, mode="a" if resume else "w") as writer:
        for start in range(0, len(requests), batch_size):
            batch_requests = requests[start : start + batch_size]
            batch_results = results[start : start + batch_size]
            collected: list[list[MetricResult]] = [[] for _ in batch_requests]
            errors: list[list[dict[str, str]]] = [[] for _ in batch_requests]
            for metric, identity in zip(metrics, identities):
                pending = []
                name = metric_name(metric)
                for index, (request, result) in enumerate(zip(batch_requests, batch_results)):
                    if not is_generation_result_successful(result):
                        # Generation success counts every attempt; quality metrics need an output.
                        collected[index].extend(
                            MetricResult(sample_id=request.sample_id, metric_id=key, raw_value=0.0)
                            if key == "generation_success"
                            else MetricResult(
                                sample_id=request.sample_id,
                                metric_id=key,
                                valid=False,
                                coverage=0,
                                skip_reason="generation_failed",
                            )
                            for key in metric_ids(metric)
                        )
                        continue
                    cached = previous.get((request.sample_id, name))
                    if (
                        identity is not None
                        and cached
                        and artifacts_available(result, checkpoint_path.parent.parent)
                        and (
                            cached["identity"] == identity
                            and cached["request"] == request.to_dict()
                            and cached["result"] == result.to_dict()
                            and set(metric_ids(metric)).issubset(row["metric_id"] for row in cached["outputs"])
                        )
                    ):
                        collected[index].extend(MetricResult.from_dict(row) for row in cached["outputs"])
                    else:
                        pending.append(index)

                outputs = None
                compute_batch = getattr(metric, "compute_batch", None)
                if pending and callable(compute_batch):
                    try:
                        outputs = align_batch_metric_outputs(
                            compute_batch([batch_requests[i] for i in pending], [batch_results[i] for i in pending]),
                            [batch_requests[i].sample_id for i in pending],
                        )
                    except Exception:
                        # A failed batch falls back once to sample-isolated scoring.
                        outputs = None
                compute_sample = getattr(metric, "compute_sample", metric)
                for position, index in enumerate(pending):
                    request, result = batch_requests[index], batch_results[index]
                    try:
                        output = outputs[position] if outputs is not None else compute_sample(request, result)
                        rows = normalize_metric_output(output, request.sample_id)
                        if not rows:
                            raise ValueError("metric returned no results")
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        errors[index].append({"metric_id": name, "message": message})
                        rows = [
                            MetricResult(
                                sample_id=request.sample_id,
                                metric_id=key,
                                valid=False,
                                coverage=0,
                                skip_reason="metric_failed",
                                diagnostics={"error": message},
                            )
                            for key in metric_ids(metric)
                        ]
                    else:
                        emitted = {row.metric_id for row in rows}
                        rows.extend(
                            MetricResult(
                                sample_id=request.sample_id,
                                metric_id=key,
                                valid=False,
                                coverage=0,
                                skip_reason="metric_not_returned",
                            )
                            for key in metric_ids(metric) if key not in emitted
                        )
                        if identity is not None and all(row.valid for row in rows):
                            writer.write(
                                {
                                    "sample_id": request.sample_id,
                                    "metric": name,
                                    "identity": identity,
                                    "request": request.to_dict(),
                                    "result": result.to_dict(),
                                    "outputs": [row.to_dict() for row in rows],
                                }
                            )
                    collected[index].extend(rows)
                writer.flush()

            batch = []
            for request, result, rows, failures in zip(batch_requests, batch_results, collected, errors):
                seen = set()
                for output in rows:
                    if output.metric_id in seen:
                        raise ValueError(f"duplicate metric id {output.metric_id!r} for sample {request.sample_id!r}")
                    seen.add(output.metric_id)
                status = "succeeded" if metrics else "not_run"
                if not is_generation_result_successful(result):
                    status = "skipped"
                    failures = [{"stage": "generate", "message": result.error or "generation failed"}]
                elif failures:
                    status = "failed"
                elif any(not row.valid for row in rows):
                    status = "skipped"
                row = {
                    "sample_id": request.sample_id,
                    "status": status,
                    "metrics": {r.metric_id: metric_value(r) for r in rows if r.valid and metric_value(r) is not None},
                    "metric_results": [r.to_dict() for r in rows],
                }
                if failures:
                    row.update(errors=failures, error="; ".join(error["message"] for error in failures))
                batch.append(row)
            yield batch


def _aggregate(metric_id: str, rows: Sequence[MetricResult], metric: Any) -> AggregateResult:
    aggregate = getattr(metric, "aggregate", None)
    if callable(aggregate):
        try:
            result = aggregate(rows)
            return (
                result
                if isinstance(result, AggregateResult)
                else AggregateResult.from_dict({**result, "metric_id": metric_id})
            )
        except Exception as exc:
            return AggregateResult(
                metric_id=metric_id,
                n_total=len(rows),
                n_skipped=len(rows),
                valid=False,
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )
    return aggregate_mean(metric_id, rows)


def build_metrics_summary(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    per_sample_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Any],
) -> dict[str, Any]:
    owners = metric_owners(metrics)
    by_metric: dict[str, list[MetricResult]] = {key: [] for key in owners}
    for row in per_sample_rows:
        for output in row["metric_results"]:
            result = MetricResult.from_dict(output)
            by_metric.setdefault(result.metric_id, []).append(result)
    per_metric, leaderboard = {}, {}
    for key, rows in sorted(by_metric.items()):
        owner = owners.get(key)
        aggregate = _aggregate(key, rows, owner)
        definition = metric_identity(owner, key)
        direction = definition.get("higher_is_better") if definition is not None else getattr(owner, "higher_is_better", None)
        if owner is None:
            from worldfoundry.evaluation.tasks.metrics.registry import default_metric_registry

            try:
                direction = default_metric_registry().get(key).higher_is_better
            except KeyError:
                pass
        payload = aggregate.to_dict()
        payload.pop("schema_version")
        payload.pop("metric_id")
        payload.update(higher_is_better=direction, sample_count=aggregate.n_valid, definition=definition)
        value = aggregate.normalized_stats.get("mean", aggregate.raw_stats.get("mean"))
        if aggregate.valid and isinstance(value, (int, float)) and not isinstance(value, bool):
            payload["mean"] = value
            leaderboard[key] = value
        per_metric[key] = payload
    generation_failed = [row.sample_id for row in results if not is_generation_result_successful(row)]
    metric_failed = [row["sample_id"] for row in per_sample_rows if row["status"] == "failed"]
    metric_skipped = [row["sample_id"] for row in per_sample_rows if row["status"] == "skipped"]
    failed = sorted(set(generation_failed + metric_failed))
    return {
        "schema_version": "worldfoundry-metrics-summary",
        "sample_count": len(requests),
        "successful_samples": len(requests) - len(set(failed + metric_skipped)),
        "skipped_samples": len(set(metric_skipped).difference(failed)),
        "failed_samples": len(failed),
        "failed_sample_ids": failed,
        "generation": {
            "successful": len(requests) - len(generation_failed),
            "failed": len(generation_failed),
            "failed_sample_ids": generation_failed,
        },
        "metrics": {
            "enabled": bool(metrics),
            "executed": len(requests) - len(generation_failed) if metrics else 0,
            "scored": sum(bool(row["metrics"]) for row in per_sample_rows),
            "successful": sum(row["status"] == "succeeded" for row in per_sample_rows),
            "failed": len(metric_failed),
            "failed_sample_ids": metric_failed,
            "skipped": len(metric_skipped),
            "skipped_sample_ids": metric_skipped,
        },
        "leaderboard": leaderboard,
        "per_metric": per_metric,
        "groups": {},
    }
