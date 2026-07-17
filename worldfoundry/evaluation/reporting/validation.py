"""Schema-aware validation for WorldFoundry contract artifacts.

Each public validator inspects a JSON/JSONL payload, checks required keys and
structure, optionally verifies that referenced artifact paths exist on disk,
and accumulates *errors* and *warnings* rather than raising.  The top-level
:func:`validate_contract_file` and :func:`validate_contract_paths` dispatch
to the appropriate schema validator based on ``schema_version`` or an explicit
``kind`` hint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from worldfoundry.evaluation.utils import read_json_object, read_jsonl_objects

from .comparison_identity import COMPARISON_IDENTITY_SCHEMA_VERSION
from .run_comparison import RUN_COMPARISON_SCHEMA_VERSION
from .run_index import RUN_INDEX_SCHEMA_VERSION
from .run_manifest import (
    ENV_REQUIREMENTS_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
)
from .run_report import RUN_SUMMARY_SCHEMA_VERSION
from .scorecard import SCORECARD_SCHEMA_VERSION

# ── Schema version identifiers ──────────────────────────────
CONTRACT_VALIDATION_SCHEMA_VERSION = "worldfoundry-contract-validation"
MODEL_BENCHMARK_RUN_SCHEMA_VERSION = "worldfoundry-model-benchmark-run"
MODEL_BENCHMARK_SUITE_SCHEMA_VERSION = "worldfoundry-model-benchmark-suite"
MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION = "worldfoundry-model-benchmark-suite-result"

# ── Internal helpers ─────────────────────────────────────────
_read_json = read_json_object
_read_jsonl_rows = read_jsonl_objects


def _mapping(value: Any) -> dict[str, Any]:
    """Coerce *value* to ``dict`` if it is a ``Mapping``; otherwise return empty."""
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    """Coerce *value* to ``list`` if it is a non-string ``Sequence``; otherwise return empty."""
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _required_mapping(payload: Mapping[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    """Extract a required mapping from *payload*; append an error if missing."""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"missing or invalid mapping: {key}")
        return {}
    return dict(value)


def _required_sequence(payload: Mapping[str, Any], key: str, errors: list[str]) -> list[Any]:
    """Extract a required sequence from *payload*; append an error if missing."""
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"missing or invalid list: {key}")
        return []
    return list(value)


def _warn_empty_mapping(payload: Mapping[str, Any], key: str, warnings: list[str]) -> None:
    """Append a warning if *payload[key]* is missing or an empty mapping."""
    value = payload.get(key)
    if not isinstance(value, Mapping) or not value:
        warnings.append(f"missing or empty mapping: {key}")


def _validate_finite_numbers(value: Any, *, path: str, errors: list[str]) -> None:
    """Reject JSON's non-standard NaN/Infinity values inside metric payloads."""
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite metric value: {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_numbers(item, path=f"{path}.{key}", errors=errors)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, path=f"{path}[{index}]", errors=errors)


# ── Artifact path helpers ───────────────────────────────────
def _artifact_values(value: Any) -> list[str]:
    """Flatten an artifact descriptor into a list of filesystem path strings."""
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Mapping):
        values = []
        for key in ("path", "uri", "destination", "scorecard_path", "summary_path"):
            if value.get(key) not in (None, ""):
                values.append(str(value[key]))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(_artifact_values(item))
        return values
    return []


def _looks_like_local_path(value: str) -> bool:
    """Return whether *value* appears to reference a local filesystem path."""
    if "://" in value:
        return value.startswith("file://")
    return bool(value) and not value.startswith("memory:")


def _check_artifact_paths(
    artifacts: Mapping[str, Any],
    *,
    source_path: Path,
    errors: list[str],
) -> None:
    """Verify that each local artifact path referenced in *artifacts* exists."""
    base_dir = source_path.parent
    for name, value in artifacts.items():
        for artifact_value in _artifact_values(value):
            if not _looks_like_local_path(artifact_value):
                continue
            artifact_path = Path(artifact_value.removeprefix("file://"))
            if not artifact_path.is_absolute():
                artifact_path = base_dir / artifact_path
            if not artifact_path.exists():
                errors.append(f"artifact path does not exist: {name}={artifact_value}")


# ── Per-schema validators ────────────────────────────────────
def _validate_artifacts(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate the ``artifacts`` mapping and optionally check local paths."""
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        warnings.append("missing or invalid mapping: artifacts")
        return
    if check_artifacts:
        _check_artifact_paths(dict(artifacts), source_path=source_path, errors=errors)


def _validate_comparison_identity(
    payload: Mapping[str, Any],
    *,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate an identity when present while keeping legacy artifacts readable."""

    identity = payload.get("comparison_identity")
    if identity is None:
        return
    if not isinstance(identity, Mapping):
        errors.append("invalid mapping: comparison_identity")
        return
    if identity.get("schema_version") != COMPARISON_IDENTITY_SCHEMA_VERSION:
        errors.append(
            "comparison_identity schema_version must be "
            f"{COMPARISON_IDENTITY_SCHEMA_VERSION!r}"
        )
    if not identity.get("comparison_key"):
        errors.append("comparison_identity missing comparison_key")
    if identity.get("status") not in {"complete", "incomplete"}:
        errors.append("comparison_identity status must be 'complete' or 'incomplete'")
    if identity.get("status") == "incomplete":
        missing = [
            *(_sequence(identity.get("missing_required"))),
            *(_sequence(identity.get("missing_recommended"))),
        ]
        if missing:
            warnings.append("incomplete comparison identity: " + ", ".join(map(str, missing)))


def _validate_run_summary(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-run-summary`` payload."""
    for key in ("run", "benchmark", "model", "dataset", "counts", "metrics", "leaderboard", "eligibility"):
        _required_mapping(payload, key, errors)
    counts = _mapping(payload.get("counts"))
    for key in ("sample_count", "successful_samples", "failed_samples"):
        if key not in counts:
            errors.append(f"counts missing required key: {key}")
    _validate_finite_numbers(payload.get("metrics"), path="metrics", errors=errors)
    _validate_finite_numbers(payload.get("leaderboard"), path="leaderboard", errors=errors)
    _validate_comparison_identity(payload, warnings=warnings, errors=errors)
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_scorecard(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-scorecard`` payload."""
    for key in ("run", "benchmark", "dataset", "metrics"):
        _required_mapping(payload, key, errors)
    for key in ("model", "generation", "eligibility"):
        if not isinstance(payload.get(key), Mapping):
            warnings.append(f"missing optional normalized scorecard mapping: {key}")
    metrics = _mapping(payload.get("metrics"))
    for key in ("per_metric", "summary"):
        if not isinstance(metrics.get(key), Mapping):
            errors.append(f"metrics missing or invalid mapping: {key}")
    if not isinstance(metrics.get("leaderboard"), Mapping):
        warnings.append("metrics missing optional leaderboard mapping")
    _validate_finite_numbers(metrics, path="metrics", errors=errors)
    _validate_comparison_identity(payload, warnings=warnings, errors=errors)
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_run_index(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-run-index`` payload."""
    rows = _required_sequence(payload, "rows", errors)
    run_count = payload.get("run_count")
    if isinstance(run_count, int) and run_count != len(rows):
        errors.append(f"run_count mismatch: expected {run_count}, rows={len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"rows[{index}] is not an object")
            continue
        if not row.get("source_path"):
            errors.append(f"rows[{index}] missing source_path")
        if not isinstance(row.get("metrics"), Mapping):
            errors.append(f"rows[{index}] missing or invalid metrics")
        _validate_finite_numbers(row.get("metrics"), path=f"rows[{index}].metrics", errors=errors)
        _validate_comparison_identity(row, warnings=warnings, errors=errors)
        if check_artifacts and isinstance(row.get("artifacts"), Mapping):
            _check_artifact_paths(dict(row["artifacts"]), source_path=source_path, errors=errors)
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_run_comparison(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-run-comparison`` payload."""
    rows = _required_sequence(payload, "rows", errors)
    _required_mapping(payload, "metrics", errors)
    _validate_finite_numbers(payload.get("metrics"), path="metrics", errors=errors)
    if not isinstance(payload.get("metric_ids"), Sequence) or isinstance(payload.get("metric_ids"), (str, bytes)):
        errors.append("missing or invalid list: metric_ids")
    run_count = payload.get("run_count")
    if isinstance(run_count, int) and run_count != len(rows):
        errors.append(f"run_count mismatch: expected {run_count}, rows={len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"rows[{index}] is not an object")
            continue
        if not isinstance(row.get("metrics"), Mapping):
            errors.append(f"rows[{index}] missing or invalid metrics")
        _validate_finite_numbers(row.get("metrics"), path=f"rows[{index}].metrics", errors=errors)
        _validate_comparison_identity(row, warnings=warnings, errors=errors)
    compatibility = payload.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, Mapping):
            errors.append("invalid mapping: compatibility")
        elif compatibility.get("status") not in {
            "compatible",
            "compatible_with_incomplete_identity",
        }:
            errors.append("invalid compatibility status")
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_run_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-run-manifest`` payload."""
    for key in ("run_id", "status", "output_dir", "artifacts", "config", "environment", "env_requirements"):
        if key not in payload:
            errors.append(f"missing required key: {key}")
    environment = _required_mapping(payload, "environment", errors)
    env_requirements = _required_mapping(payload, "env_requirements", errors)
    if environment and environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        errors.append(f"environment schema_version must be {ENVIRONMENT_SCHEMA_VERSION!r}")
    if env_requirements and env_requirements.get("schema_version") != ENV_REQUIREMENTS_SCHEMA_VERSION:
        errors.append(f"env_requirements schema_version must be {ENV_REQUIREMENTS_SCHEMA_VERSION!r}")
    for key in ("python", "git", "packages", "cache_paths"):
        if environment and key not in environment:
            errors.append(f"environment missing required key: {key}")
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_model_benchmark_run(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-model-benchmark-run`` payload."""
    for key in ("status", "benchmark_id", "model_id", "output_dir", "generated_artifact_dir", "artifacts"):
        if key not in payload:
            errors.append(f"missing required key: {key}")
    _required_mapping(payload, "benchmark", errors)
    _validate_artifacts(payload, source_path=source_path, check_artifacts=check_artifacts, warnings=warnings, errors=errors)


def _validate_suite_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-model-benchmark-suite`` (suite manifest) payload."""
    _required_mapping(payload, "summary", errors)
    cells = _required_sequence(payload, "cells", errors)
    summary = _mapping(payload.get("summary"))
    total = summary.get("total")
    if isinstance(total, int) and total != len(cells):
        errors.append(f"summary.total mismatch: expected {total}, cells={len(cells)}")
    for key in ("status", "exit_code"):
        if key not in payload:
            errors.append(f"missing required key: {key}")
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            errors.append(f"cells[{index}] is not an object")
            continue
        for key in ("model_id", "benchmark_id", "status"):
            if key not in cell:
                errors.append(f"cells[{index}] missing required key: {key}")
        if cell.get("status") == "succeeded":
            if not cell.get("run_manifest_path"):
                warnings.append(f"cells[{index}] succeeded without run_manifest_path")
            if not cell.get("run_summary_path"):
                warnings.append(f"cells[{index}] succeeded without run_summary_path")
        if check_artifacts and isinstance(cell.get("artifacts"), Mapping):
            _check_artifact_paths(dict(cell["artifacts"]), source_path=source_path, errors=errors)


def _validate_suite_result(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    check_artifacts: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    """Validate a ``worldfoundry-model-benchmark-suite-result`` payload."""
    for key in ("status", "exit_code", "output_dir", "suite_manifest_path", "suite_report_path"):
        if key not in payload:
            errors.append(f"missing required key: {key}")
    _required_mapping(payload, "summary", errors)
    cells = _required_sequence(payload, "cells", errors)
    summary = _mapping(payload.get("summary"))
    total = summary.get("total")
    if isinstance(total, int) and total != len(cells):
        errors.append(f"summary.total mismatch: expected {total}, cells={len(cells)}")
    if check_artifacts:
        artifacts = {
            "suite_manifest": payload.get("suite_manifest_path"),
            "suite_report": payload.get("suite_report_path"),
        }
        _check_artifact_paths(artifacts, source_path=source_path, errors=errors)


Validator = Callable[..., None]

# ── Validator registry ──────────────────────────────────────
_SCHEMA_VALIDATORS: dict[str, tuple[str, Validator]] = {
    RUN_SUMMARY_SCHEMA_VERSION: ("run_summary", _validate_run_summary),
    SCORECARD_SCHEMA_VERSION: ("scorecard", _validate_scorecard),
    RUN_INDEX_SCHEMA_VERSION: ("run_index", _validate_run_index),
    RUN_COMPARISON_SCHEMA_VERSION: ("run_comparison", _validate_run_comparison),
    RUN_MANIFEST_SCHEMA_VERSION: ("run_manifest", _validate_run_manifest),
    MODEL_BENCHMARK_RUN_SCHEMA_VERSION: ("model_benchmark_run", _validate_model_benchmark_run),
    MODEL_BENCHMARK_SUITE_SCHEMA_VERSION: ("model_benchmark_suite", _validate_suite_manifest),
    MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION: ("model_benchmark_suite_result", _validate_suite_result),
}

# ── Kind normalisation ──────────────────────────────────────
_KIND_TO_SCHEMA = {
    kind: schema_version
    for schema_version, (kind, _validator) in _SCHEMA_VALIDATORS.items()
}
CONTRACT_ARTIFACT_KIND_ALIASES = {
    # NOTE: Maps hyphenated, underscore, and short aliases to canonical kind names.
    "auto": "auto",
    "scorecard": "scorecard",
    "run-summary": "run_summary",
    "run_summary": "run_summary",
    "summary": "run_summary",
    "run-index": "run_index",
    "run_index": "run_index",
    "index": "run_index",
    "run-comparison": "run_comparison",
    "run_comparison": "run_comparison",
    "comparison": "run_comparison",
    "run-manifest": "run_manifest",
    "run_manifest": "run_manifest",
    "model-benchmark-run": "model_benchmark_run",
    "model_benchmark_run": "model_benchmark_run",
    "suite-manifest": "model_benchmark_suite",
    "suite_manifest": "model_benchmark_suite",
    "model-benchmark-suite": "model_benchmark_suite",
    "model_benchmark_suite": "model_benchmark_suite",
    "suite-result": "model_benchmark_suite_result",
    "suite_result": "model_benchmark_suite_result",
    "model-benchmark-suite-result": "model_benchmark_suite_result",
    "model_benchmark_suite_result": "model_benchmark_suite_result",
}
CONTRACT_ARTIFACT_KIND_CHOICES = (
    "auto",
    "scorecard",
    "run-summary",
    "run-index",
    "run-comparison",
    "run-manifest",
    "model-benchmark-run",
    "suite-manifest",
    "suite-result",
)


def normalize_contract_artifact_kind(kind: str) -> str:
    """Resolve an artifact kind alias to its canonical internal name."""
    return CONTRACT_ARTIFACT_KIND_ALIASES.get(kind, kind)


def _normalize_kind(kind: str) -> str:
    return normalize_contract_artifact_kind(kind)


def _payload_from_path(path: Path, kind: str) -> dict[str, Any]:
    """Load a contract payload from *path*, handling JSONL for run-index kinds."""
    if path.suffix.lower() == ".jsonl":
        if kind not in {"auto", "run_index"}:
            raise ValueError(f"JSONL contract validation only supports run_index, got {kind}")
        rows = _read_jsonl_rows(path)
        return {
            "schema_version": RUN_INDEX_SCHEMA_VERSION,
            "run_count": len(rows),
            "rows": rows,
            "runs": rows,
            "artifacts": {},
        }
    return _read_json(path)


def validate_contract_file(
    path: str | Path,
    *,
    kind: str = "auto",
    check_artifacts: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a single contract artifact file against its expected schema.

    Args:
        path: Filesystem path to the JSON or JSONL contract artifact.
        kind: Artifact kind hint; ``"auto"`` dispatches via ``schema_version``.
        check_artifacts: When ``True``, verify that referenced local paths exist.
        strict: When ``True``, treat warnings as failures too.

    Returns:
        A validation report dict with ``ok``, ``errors``, ``warnings``, and
        metadata about the detected schema kind and version.
    """
    kind = _normalize_kind(kind)
    source_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    schema_version: str | None = None
    detected_kind = "unknown"
    try:
        payload = _payload_from_path(source_path, kind)
        schema_version = str(payload.get("schema_version") or "")
        expected_schema = _KIND_TO_SCHEMA.get(kind)
        if expected_schema is not None and schema_version != expected_schema:
            errors.append(f"expected schema_version {expected_schema!r}, got {schema_version!r}")
        schema_for_validator = expected_schema or schema_version
        validator_entry = _SCHEMA_VALIDATORS.get(schema_for_validator)
        if validator_entry is None:
            errors.append(f"unsupported schema_version: {schema_version!r}")
        else:
            detected_kind, validator = validator_entry
            validator(
                payload,
                source_path=source_path,
                check_artifacts=check_artifacts,
                warnings=warnings,
                errors=errors,
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    ok = not errors and (not strict or not warnings)
    return {
        "path": str(source_path),
        "kind": detected_kind,
        "schema_version": schema_version,
        "ok": ok,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def validate_contract_paths(
    paths: Sequence[str | Path],
    *,
    kind: str = "auto",
    check_artifacts: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate multiple contract artifact files and aggregate the results.

    Args:
        paths: Sequence of filesystem paths to contract artifact files.
        kind: Artifact kind hint applied to every file.
        check_artifacts: When ``True``, verify that referenced local paths exist.
        strict: When ``True``, treat warnings as failures.

    Returns:
        An aggregate validation report with ``ok``, ``valid_count``,
        ``invalid_count``, and per-file ``results``.
    """
    results = [
        validate_contract_file(path, kind=kind, check_artifacts=check_artifacts, strict=strict)
        for path in paths
    ]
    ok = all(result["ok"] for result in results)
    return {
        "schema_version": CONTRACT_VALIDATION_SCHEMA_VERSION,
        "ok": ok,
        "path_count": len(results),
        "valid_count": sum(1 for result in results if result["ok"]),
        "invalid_count": sum(1 for result in results if not result["ok"]),
        "strict": strict,
        "check_artifacts": check_artifacts,
        "results": results,
    }


def build_markdown_contract_validation(report: Mapping[str, Any]) -> str:
    """Render an aggregate validation report as a Markdown table with per-file details."""
    lines = [
        "# WorldFoundry Contract Validation",
        "",
        f"- OK: {'true' if report.get('ok') else 'false'}",
        f"- Paths: {report.get('path_count', 0)}",
        f"- Invalid: {report.get('invalid_count', 0)}",
        "",
        "| Path | Kind | Schema | OK | Errors | Warnings |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for result in _sequence(report.get("results")):
        if not isinstance(result, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in (
                    result.get("path", ""),
                    result.get("kind", ""),
                    result.get("schema_version", ""),
                    "true" if result.get("ok") else "false",
                    result.get("error_count", 0),
                    result.get("warning_count", 0),
                )
            )
            + " |"
        )
        for error in result.get("errors") or ():
            lines.append(f"- ERROR {result.get('path')}: {error}")
        for warning in result.get("warnings") or ():
            lines.append(f"- WARNING {result.get('path')}: {warning}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CONTRACT_VALIDATION_SCHEMA_VERSION",
    "build_markdown_contract_validation",
    "validate_contract_file",
    "validate_contract_paths",
]
