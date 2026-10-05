"""Shared scorecard / manifest layout for in-process evaluation runs.

``contract`` and ``existing_results`` write the same artifact names.  Generation
vs offline scoring stay in those modules; this module owns paths, coercion, and
directory setup.
"""

from __future__ import annotations

from asyncio import CancelledError
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from worldfoundry.core.io.formats.serialization import iter_jsonl
from worldfoundry.evaluation.api import (
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    is_generation_result_successful,
)
from worldfoundry.evaluation.api.artifacts import local_path_for_uri
from worldfoundry.evaluation.reporting.run_manifest import redact_secrets
from worldfoundry.evaluation.utils import jsonable, read_json_or_jsonl, reset_jsonl, write_json, write_jsonl

JsonRow = dict[str, Any]

SESSION_JSONL_KEYS = ("requests", "results", "artifacts", "sample_ledger", "per_sample")


def artifacts_available(result: GenerationResult, base_dir: Path) -> bool:
    """Local outputs must still exist; remote references are left to their readers."""
    return all(
        path is None or path.is_file()
        for artifact in result.artifacts.values()
        for path in (local_path_for_uri(artifact.uri, base_dir),)
    )


def resume_generation(
    paths: Mapping[str, Path], requests: Sequence[GenerationRequest], identity: Mapping[str, Any],
) -> dict[str, GenerationResult]:
    """Reuse completed generation when model configuration and requests match."""
    if not all(paths[key].exists() for key in ("manifest", "requests", "results")):
        return {}
    manifest = read_json_or_jsonl(paths["manifest"])
    if manifest.get("generation_identity") != redact_secrets(jsonable(identity)):
        return {}
    previous = {row["sample_id"]: row for row in iter_jsonl(paths["requests"])}
    expected = {row.sample_id: row.to_dict() for row in requests}
    results = {}
    for row in iter_jsonl(paths["results"]):
        sample_id = row["sample_id"]
        if sample_id in expected and previous.get(sample_id) == expected[sample_id]:
            result = GenerationResult.from_dict(row)
            if is_generation_result_successful(result) and artifacts_available(result, paths["manifest"].parent):
                results[sample_id] = result
    return results


def prepare_generation(
    paths: Mapping[str, Path], requests: Sequence[GenerationRequest], cached: Mapping[str, GenerationResult],
) -> None:
    """Record this attempt's inputs and discard scores for samples that need generation."""
    reset_session_jsonl(paths)
    write_jsonl(paths["requests"], (row.to_dict() for row in requests))
    write_jsonl(paths["results"], (row.to_dict() for row in cached.values()))
    checkpoint = paths["per_sample"].parent / "checkpoint.jsonl"
    pending = {row.sample_id for row in requests if row.sample_id not in cached}
    if pending and checkpoint.exists():
        retained = [row for row in iter_jsonl(checkpoint) if row["sample_id"] not in pending]
        write_jsonl(checkpoint, retained)


def _clear_reports(paths: Mapping[str, Path]) -> None:
    for key in ("run_summary", "scorecard", "report"):
        paths[key].unlink(missing_ok=True)


@contextmanager
def run_stage(
    paths: Mapping[str, Path], stage: str, *, manifest: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Keep stage failures and current report availability consistent across executors."""
    if manifest is not None:
        _clear_reports(paths)
        paths["summary"].unlink(missing_ok=True)
        payload = {
            "schema_version": "worldfoundry-run-manifest",
            "output_dir": str(paths["manifest"].parent),
            "started_at": utcnow_iso(),
            "artifacts": {}, "config": {}, "environment": {}, "env_requirements": {},
            **dict(manifest),
        }
    else:
        payload = read_json_or_jsonl(paths["manifest"])
    payload.update(status="running", stage=stage)
    write_json(paths["manifest"], redact_secrets(payload))
    try:
        yield
    except BaseException as exc:
        payload = read_json_or_jsonl(paths["manifest"])
        payload.update(
            status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit, CancelledError)) else "failed",
            stage=stage, finished_at=utcnow_iso(), error=f"{type(exc).__name__}: {exc}",
        )
        write_json(paths["manifest"], redact_secrets(payload))
        _clear_reports(paths)
        raise


def utcnow_iso() -> str:
    """UTC timestamp with ``Z`` suffix and no microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_mapping(value: Any) -> JsonRow:
    payload = jsonable(value)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def coerce_generation_result(item: Any, sample_id: str) -> GenerationResult:
    """Normalize online and offline results, retaining external fields in metadata.extra."""
    if isinstance(item, GenerationResult):
        if item.sample_id == sample_id:
            return item
        return replace(item, sample_id=sample_id, metadata={**item.metadata, "source_sample_id": item.sample_id})

    row = coerce_mapping(item)
    row.setdefault("sample_id", sample_id)
    known_keys = {"sample_id", "request_id", "model_id", "artifacts", "status", "error", "timings", "metadata", "schema_version"}
    metadata = dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), Mapping) else {}
    extras = {key: value for key, value in row.items() if key not in known_keys}
    if extras:
        # Explicit metadata wins conflicts; retain other fields from raw runner output.
        for key, value in (metadata.get("extra") or {}).items():
            if isinstance(value, Mapping) and isinstance(extras.get(key), Mapping):
                extras[key] = {**extras[key], **value}
            else:
                extras[key] = value
        metadata["extra"] = extras
    artifacts = {}
    for name, artifact in (row.get("artifacts") or {}).items():
        if isinstance(artifact, ArtifactRef):
            artifacts[str(name)] = artifact
        elif isinstance(artifact, Mapping):
            artifacts[str(name)] = ArtifactRef.from_dict({"kind": str(name), **artifact})
        else:
            artifacts[str(name)] = ArtifactRef(uri=str(artifact), kind=str(name))
    return GenerationResult(
        sample_id=str(row["sample_id"]),
        request_id=row.get("request_id"),
        model_id=str(row.get("model_id") or ""),
        artifacts=artifacts,
        status=str(row.get("status") or ("failed" if row.get("error") else "succeeded")),
        error=row.get("error"),
        timings=row.get("timings") or {},
        metadata=metadata,
    )


def coerce_optional_mapping(value: Mapping[str, Any] | Any | None, default: Mapping[str, Any]) -> JsonRow:
    if value is None:
        return dict(default)
    return coerce_mapping(value)


def prepare_run_output_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    return root


def session_paths(output_dir: Path) -> dict[str, Path]:
    """Working paths used while a run is in progress."""
    metrics_dir = output_dir / "metrics"
    return {
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


def reset_session_jsonl(paths: Mapping[str, Path]) -> None:
    for key in SESSION_JSONL_KEYS:
        reset_jsonl(paths[key])


def artifact_report_paths(output_dir: Path, artifact_count: int) -> dict[str, str]:
    """Absolute path map written into the finished scorecard."""
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
