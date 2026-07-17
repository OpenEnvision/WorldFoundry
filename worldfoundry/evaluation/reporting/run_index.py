"""Build, load, filter, and render run index artifacts.

A *run index* aggregates multiple run summaries into a single payload with
cross-run metric metadata, enabling filtering and comparison across runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import (
    escape_markdown_cell as _escape_markdown_cell,
    format_value as _format_value,
    read_json_object,
    write_json,
    write_jsonl,
)

from .run_report import (
    _dedupe_labels,
    _int_or_zero,
    _mapping,
    _normalise_roots,
    _row_from_summary,
    _run_summary_candidate,
    load_run_summary,
)

RUN_INDEX_SCHEMA_VERSION = "worldfoundry-run-index"

# ── Internal helpers ─────────────────────────────────────────

def _rows_from_index_payload(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    """Extract and normalise the ``rows`` list from a loaded index payload."""
    rows = payload.get("rows") or payload.get("runs")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"run index rows must be a JSON array in {path}")
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"run index row {index} must be a JSON object in {path}")
        parsed_rows.append(dict(row))
    return parsed_rows

def _index_summary_from_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an index payload from a flat list of row dicts."""
    benchmarks = sorted({str(row["benchmark"]) for row in rows if row.get("benchmark")})
    datasets = sorted({str(row["dataset_id"]) for row in rows if row.get("dataset_id")})
    metric_ids = sorted(
        {
            str(metric_id)
            for row in rows
            for metric_id in (row.get("metric_ids") or _mapping(row.get("metrics")).keys())
        }
    )
    comparison_keys = sorted({str(row["comparison_key"]) for row in rows if row.get("comparison_key")})
    comparison_identity_statuses = _comparison_identity_statuses(rows)
    return {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "roots": [],
        "root": None,
        "run_count": len(rows),
        "benchmarks": benchmarks,
        "datasets": datasets,
        "metric_ids": metric_ids,
        "comparison_keys": comparison_keys,
        "comparison_identity_statuses": comparison_identity_statuses,
        "runs": list(rows),
        "rows": list(rows),
        "issues": [],
        "artifacts": {"index_source": str(path.resolve())},
    }

def load_run_index(path: str | Path) -> dict[str, Any]:
    """Load a run index from index.json or index.jsonl."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"run index path does not exist: {source_path}")
    if source_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError(f"run index JSONL line {line_number} must be a JSON object in {source_path}")
            rows.append(dict(row))
        return _index_summary_from_rows(source_path, rows)

    payload = read_json_object(source_path)
    schema_version = payload.get("schema_version")
    if schema_version != RUN_INDEX_SCHEMA_VERSION:
        raise ValueError(f"unsupported run index schema_version in {source_path}: {schema_version!r}")
    rows = _rows_from_index_payload(payload, source_path)
    index = dict(payload)
    index["rows"] = rows
    index["runs"] = rows
    index["run_count"] = len(rows)
    index.setdefault("artifacts", {})
    return index

def _filter_values(values: Sequence[str] | None) -> set[str] | None:
    """Normalise a filter sequence to a ``set`` of non-empty strings; ``None`` means no filter."""
    if values is None:
        return None
    normalized = {str(value) for value in values if value not in (None, "")}
    return normalized or None

def _matches_filter(value: Any, accepted: set[str] | None) -> bool:
    """Return whether *value* matches the *accepted* filter set (or all when ``None``)."""
    return accepted is None or str(value) in accepted

def _matches_any_filter(values: Sequence[Any], accepted: set[str] | None) -> bool:
    """Return whether any item in *values* matches the *accepted* filter set."""
    if accepted is None:
        return True
    return any(value not in (None, "") and str(value) in accepted for value in values)

def _row_metric_ids(row: Mapping[str, Any]) -> set[str]:
    """Extract the set of metric IDs available in a single index row."""
    metric_ids = row.get("metric_ids")
    if isinstance(metric_ids, Sequence) and not isinstance(metric_ids, (str, bytes)):
        return {str(metric_id) for metric_id in metric_ids}
    return {str(metric_id) for metric_id in _mapping(row.get("metrics"))}

def select_run_index_rows(
    index: Mapping[str, Any],
    *,
    benchmarks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    run_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    require_score_valid: bool = False,
    require_leaderboard_valid: bool = False,
    required_metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select comparable rows from a loaded run index.

    Field filters are exact matches. Repeated values are OR'd within a field and
    different fields are AND'd together.
    """

    benchmark_filter = _filter_values(benchmarks)
    model_filter = _filter_values(models)
    dataset_filter = _filter_values(datasets)
    run_id_filter = _filter_values(run_ids)
    status_filter = _filter_values(statuses)
    required_metric_set = _filter_values(required_metrics) or set()
    rows = _rows_from_index_payload(index, Path("<run-index>"))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not _matches_filter(row.get("benchmark"), benchmark_filter):
            continue
        if not _matches_any_filter([row.get("model_id"), row.get("model_name")], model_filter):
            continue
        if not _matches_filter(row.get("dataset_id"), dataset_filter):
            continue
        if not _matches_filter(row.get("run_id"), run_id_filter):
            continue
        if not _matches_filter(row.get("status"), status_filter):
            continue
        if require_score_valid and row.get("score_valid") is not True:
            continue
        if require_leaderboard_valid and row.get("leaderboard_valid") is not True:
            continue
        if required_metric_set and not required_metric_set.issubset(_row_metric_ids(row)):
            continue
        selected.append(dict(row))
    return selected

def run_paths_from_index(
    path: str | Path,
    *,
    benchmarks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    run_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    require_score_valid: bool = False,
    require_leaderboard_valid: bool = False,
    required_metrics: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Load a run index, apply filters, and return comparable run path entries.

    Args:
        path: Path to the run index JSON or JSONL file.
        benchmarks: Exact-match benchmark filter (OR within, AND across fields).
        models: Exact-match model ID or name filter.
        datasets: Exact-match dataset filter.
        run_ids: Exact-match run ID filter.
        statuses: Exact-match status filter.
        require_score_valid: Only include rows where ``score_valid`` is ``True``.
        require_leaderboard_valid: Only include rows where ``leaderboard_valid`` is ``True``.
        required_metrics: Only include rows containing all listed metric IDs.

    Returns:
        List of dicts with ``path`` and ``label`` for each selected run.

    Raises:
        ValueError: If the selection matches no runs.
    """
    index = load_run_index(path)
    rows = select_run_index_rows(
        index,
        benchmarks=benchmarks,
        models=models,
        datasets=datasets,
        run_ids=run_ids,
        statuses=statuses,
        require_score_valid=require_score_valid,
        require_leaderboard_valid=require_leaderboard_valid,
        required_metrics=required_metrics,
    )
    selected: list[dict[str, str]] = []
    for row in rows:
        source_path = row.get("source_path")
        if source_path in (None, ""):
            continue
        selected.append(
            {
                "path": str(source_path),
                "label": str(row.get("label") or row.get("run_id") or Path(str(source_path)).parent.name),
            }
        )
    if not selected:
        raise ValueError(f"run index selection matched no comparable runs: {path}")
    return selected

def discover_run_summaries(roots: str | Path | Sequence[str | Path]) -> list[Path]:
    """Find root-level run summary files below one or more roots.

    Discovery is recursive for directories and stops descending once a run
    directory is found, so nested artifact files named summary.json are not
    indexed as separate runs.
    """

    summary_paths: list[Path] = []
    for root in _normalise_roots(roots):
        if not root.exists():
            raise FileNotFoundError(f"run index root does not exist: {root}")
        if root.is_file():
            summary_paths.append(root)
            continue
        direct = _run_summary_candidate(root)
        if direct is not None:
            summary_paths.append(direct)
            continue

        for dirpath, dirnames, _filenames in os.walk(root):
            dirnames[:] = sorted(dirnames)
            current = Path(dirpath)
            candidate = _run_summary_candidate(current)
            if candidate is not None:
                summary_paths.append(candidate)
                dirnames[:] = []

    return sorted(summary_paths, key=lambda path: str(path.resolve()))

def _index_row_from_summary(
    *,
    index: int,
    summary: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    """Build a single index row dict from a loaded run summary."""
    row = _row_from_summary(index=index, summary=summary, source_path=source_path, label=None)
    run = _mapping(summary.get("run"))
    metrics = _mapping(row.get("metrics"))
    row.update(
        {
            "run_dir": str(source_path.parent.resolve()),
            "source_file": source_path.name,
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "worldfoundry_version": run.get("worldfoundry_version"),
            "run_fingerprint": run.get("run_fingerprint"),
            "metric_ids": sorted(str(metric_id) for metric_id in metrics),
        }
    )
    return row

def _invalid_index_row(*, index: int, source_path: Path, error: str) -> dict[str, Any]:
    """Build a placeholder row for a run summary that failed to load."""
    return {
        "index": index,
        "label": source_path.parent.name if source_path.name in {"summary.json", "scorecard.json"} else source_path.stem,
        "source_path": str(source_path.resolve()),
        "run_dir": str(source_path.parent.resolve()),
        "source_file": source_path.name,
        "status": "invalid",
        "metrics": {},
        "metric_ids": [],
        "issue": error,
    }

def _duplicate_run_id_issues(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Detect duplicate ``run_id`` values across rows and return issue strings."""
    by_run_id: dict[str, list[str]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if run_id in (None, ""):
            continue
        by_run_id.setdefault(str(run_id), []).append(str(row.get("label") or row.get("source_path")))
    return [
        f"duplicate run_id: {run_id} ({', '.join(labels)})"
        for run_id, labels in sorted(by_run_id.items())
        if len(labels) > 1
    ]


def _comparison_identity_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count complete, incomplete, and unavailable comparison identities."""

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("comparison_identity_status") or "unavailable")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))

def build_run_index(
    roots: str | Path | Sequence[str | Path],
    *,
    include_invalid: bool = False,
) -> dict[str, Any]:
    """Discover run summaries under *roots* and build a structured index payload.

    Args:
        roots: One or more root directories / files to scan for run summaries.
        include_invalid: When ``True``, include rows for summaries that failed
            to load (marked ``status="invalid"``).

    Returns:
        An index payload dict conforming to ``RUN_INDEX_SCHEMA_VERSION``.
    """
    source_paths = discover_run_summaries(roots)
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    seen_paths: set[str] = set()
    for source_path in source_paths:
        resolved = str(source_path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            summary = load_run_summary(source_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issue = f"skipped {source_path}: {exc}"
            issues.append(issue)
            if include_invalid:
                rows.append(_invalid_index_row(index=len(rows), source_path=source_path, error=str(exc)))
            continue
        rows.append(_index_row_from_summary(index=len(rows), summary=summary, source_path=source_path))

    rows = _dedupe_labels(rows)
    issues.extend(_duplicate_run_id_issues(rows))
    if not rows:
        issues.append("no run summaries found")

    benchmarks = sorted({str(row["benchmark"]) for row in rows if row.get("benchmark")})
    datasets = sorted({str(row["dataset_id"]) for row in rows if row.get("dataset_id")})
    metric_ids = sorted(
        {
            str(metric_id)
            for row in rows
            for metric_id in (row.get("metric_ids") or _mapping(row.get("metrics")).keys())
        }
    )
    comparison_keys = sorted({str(row["comparison_key"]) for row in rows if row.get("comparison_key")})
    root_paths = [str(root.resolve()) for root in _normalise_roots(roots)]
    return {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "roots": root_paths,
        "root": root_paths[0] if len(root_paths) == 1 else None,
        "run_count": len(rows),
        "benchmarks": benchmarks,
        "datasets": datasets,
        "metric_ids": metric_ids,
        "comparison_keys": comparison_keys,
        "comparison_identity_statuses": _comparison_identity_statuses(rows),
        "runs": rows,
        "rows": rows,
        "issues": issues,
        "artifacts": {},
    }


def build_markdown_run_index(index: Mapping[str, Any]) -> str:
    """Render a run index payload as a Markdown table with an issues section."""
    rows = [dict(row) for row in index.get("rows") or () if isinstance(row, Mapping)]
    lines = [
        "# WorldFoundry Run Index",
        "",
        f"- Runs: {_format_value(index.get('run_count'))}",
        f"- Roots: {_format_value(index.get('roots') or [])}",
        f"- Benchmarks: {_format_value(index.get('benchmarks') or [])}",
        f"- Metrics: {_format_value(index.get('metric_ids') or [])}",
        f"- Comparison identities: {_format_value(index.get('comparison_identity_statuses') or {})}",
        "",
    ]

    headers = [
        "Run",
        "Status",
        "Benchmark",
        "Model",
        "Dataset",
        "Evaluation",
        "Protocol",
        "Samples",
        "Failed",
        "Score Valid",
        "Metrics",
        "Source",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [
            row.get("label"),
            row.get("status"),
            row.get("benchmark"),
            row.get("model_id") or row.get("model_name"),
            row.get("dataset_id"),
            row.get("evaluation_mode"),
            row.get("protocol_id"),
            row.get("sample_count"),
            row.get("failed_samples"),
            row.get("score_valid"),
            row.get("metric_ids") or sorted(_mapping(row.get("metrics"))),
            row.get("source_path"),
        ]
        lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in values) + " |")

    issues = [str(issue) for issue in index.get("issues") or ()]
    if issues:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {_format_value(issue)}" for issue in issues)

    return "\n".join(lines).rstrip() + "\n"


def write_run_index(
    roots: str | Path | Sequence[str | Path],
    *,
    include_invalid: bool = False,
    output_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    output_html: str | Path | None = None,
) -> dict[str, Any]:
    """Build a run index and write JSON, JSONL, and/or HTML browser artifacts.

    Args:
        roots: Root directories / files to scan for run summaries.
        include_invalid: Include rows for unloadable summaries.
        output_dir: Directory for default artifact filenames.
        output_json: Explicit path for the index JSON file.
        output_jsonl: Explicit path for the index JSONL file.
        output_html: Explicit path for the browser HTML file.

    Returns:
        The index payload with ``artifacts`` updated to include output paths.
    """
    index = build_run_index(roots, include_invalid=include_invalid)
    resolved_output_json = Path(output_json) if output_json is not None else None
    resolved_output_jsonl = Path(output_jsonl) if output_jsonl is not None else None
    resolved_output_html = Path(output_html) if output_html is not None else None
    if output_dir is not None:
        output_root = Path(output_dir)
        resolved_output_json = resolved_output_json or output_root / "index.json"
        resolved_output_jsonl = resolved_output_jsonl or output_root / "index.jsonl"
        resolved_output_html = resolved_output_html or output_root / "index.html"

    artifacts: dict[str, str] = {}
    if resolved_output_json is not None:
        artifacts["index_json"] = str(resolved_output_json.resolve())
    if resolved_output_jsonl is not None:
        artifacts["index_jsonl"] = str(resolved_output_jsonl.resolve())
    if resolved_output_html is not None:
        artifacts["index_html"] = str(resolved_output_html.resolve())
    index["artifacts"] = artifacts

    if resolved_output_json is not None:
        write_json(resolved_output_json, index)
    if resolved_output_jsonl is not None:
        write_jsonl(resolved_output_jsonl, index["rows"])
    if resolved_output_html is not None:
        from .run_browser import write_run_browser

        write_run_browser(index, resolved_output_html)
    return index
