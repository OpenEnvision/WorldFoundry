"""Build and render run comparisons across multiple evaluation runs.

A *run comparison* aligns several run summaries side-by-side, computes metric
deltas from a designated baseline, and identifies the best-performing run per
metric.  Results are emitted as a JSON payload and/or a Markdown table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import (
    escape_markdown_cell as _escape_markdown_cell,
    format_value as _format_value,
    write_json,
    write_text,
)

from .comparison_identity import compare_identities, comparison_identity_from_summary
from .run_report import _dedupe_labels, _mapping, _number_or_none, _row_from_summary, _run_summary_path, load_run_summary

RUN_COMPARISON_SCHEMA_VERSION = "worldfoundry-run-comparison"


def build_markdown_comparison(comparison: Mapping[str, Any]) -> str:
    """Render a run comparison payload as a Markdown table with optional baseline deltas."""
    metric_ids = [str(metric_id) for metric_id in comparison.get("metric_ids") or ()]
    rows = [dict(row) for row in comparison.get("rows") or () if isinstance(row, Mapping)]
    lines = [
        "# WorldFoundry Run Comparison",
        "",
        f"- Runs: {_format_value(comparison.get('run_count'))}",
        f"- Benchmarks: {_format_value(comparison.get('benchmarks') or [])}",
        f"- Metrics: {_format_value(metric_ids)}",
        f"- Compatibility: {_format_value(_mapping(comparison.get('compatibility')).get('status'))}",
        "",
    ]

    headers = [
        "Run",
        "Status",
        "Benchmark",
        "Model",
        "Samples",
        "Failed",
        "Score Valid",
        "Leaderboard Valid",
        *[
            column
            for metric_id in metric_ids
            for column in (
                [metric_id, f"{metric_id} delta"]
                if comparison.get("baseline") is not None
                else [metric_id]
            )
        ],
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        metric_values = []
        for metric_id in metric_ids:
            metric_values.append(metrics.get(metric_id))
            if comparison.get("baseline") is not None:
                metric_values.append(_mapping(row.get("delta_from_baseline")).get(metric_id))
        values = [
            row.get("label"),
            row.get("status"),
            row.get("benchmark"),
            row.get("model_id") or row.get("model_name"),
            row.get("sample_count"),
            row.get("failed_samples"),
            row.get("score_valid"),
            row.get("leaderboard_valid"),
            *metric_values,
        ]
        lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in values) + " |")

    best = _mapping(comparison.get("best_by_metric"))
    if best:
        lines.extend(["", "## Best By Metric", "", "| Metric | Run | Value |", "| --- | --- | ---: |"])
        for metric_id, payload in sorted(best.items()):
            if isinstance(payload, Mapping):
                direction = "higher" if payload.get("higher_is_better") is not False else "lower"
                value = f"{_format_value(payload.get('value'))} ({direction} is better)"
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _escape_markdown_cell(metric_id),
                            _escape_markdown_cell(payload.get("label") or payload.get("run_id")),
                            _escape_markdown_cell(value),
                        )
                    )
                    + " |"
                )

    compatibility_warnings = [
        str(warning) for warning in _mapping(comparison.get("compatibility")).get("warnings") or ()
    ]
    if compatibility_warnings:
        lines.extend(["", "## Compatibility Warnings", ""])
        lines.extend(f"- {_format_value(warning)}" for warning in compatibility_warnings)

    return "\n".join(lines).rstrip() + "\n"

# ── Internal helpers ─────────────────────────────────────────
def _higher_is_better(summaries: Sequence[Mapping[str, Any]], metric_id: str) -> bool:
    """Determine whether higher values are better for *metric_id* across *summmaries*."""
    values: list[bool] = []
    for summary in summaries:
        per_metric = _mapping(_mapping(summary.get("metrics")).get("per_metric"))
        metric_payload = _mapping(per_metric.get(metric_id))
        value = metric_payload.get("higher_is_better")
        if isinstance(value, bool):
            values.append(value)
    if values and all(value is False for value in values):
        return False
    return True

def _best_by_metric(
    *,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    metric_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Identify the best run for each metric, respecting higher/lower-is-better semantics."""
    best: dict[str, dict[str, Any]] = {}
    for metric_id in metric_ids:
        higher_is_better = _higher_is_better(summaries, metric_id)
        candidates = []
        for row in rows:
            value = _number_or_none(_mapping(row.get("metrics")).get(metric_id))
            if value is not None:
                candidates.append((float(value), row))
        if not candidates:
            continue
        best_value, best_row = (
            max(candidates, key=lambda item: item[0])
            if higher_is_better
            else min(candidates, key=lambda item: item[0])
        )
        best[metric_id] = {
            "value": best_value,
            "label": best_row.get("label"),
            "run_id": best_row.get("run_id"),
            "model_id": best_row.get("model_id"),
            "higher_is_better": higher_is_better,
        }
    return best

def _metric_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    metric_ids: Sequence[str],
    baseline_label: str | None,
) -> dict[str, dict[str, Any]]:
    """Compute per-metric value tables and baseline deltas across *rows*."""
    baseline_row = next((row for row in rows if row.get("label") == baseline_label), None)
    baseline_metrics = _mapping(baseline_row.get("metrics")) if baseline_row is not None else {}
    metrics: dict[str, dict[str, Any]] = {}
    for metric_id in metric_ids:
        values = {
            str(row.get("label")): _mapping(row.get("metrics")).get(metric_id)
            for row in rows
            if metric_id in _mapping(row.get("metrics"))
        }
        baseline_value = _number_or_none(baseline_metrics.get(metric_id))
        deltas = {}
        if baseline_label is not None and baseline_value is not None:
            for row in rows:
                label = str(row.get("label"))
                if label == baseline_label:
                    continue
                value = _number_or_none(_mapping(row.get("metrics")).get(metric_id))
                if value is not None:
                    deltas[label] = float(value) - float(baseline_value)
        metrics[metric_id] = {
            "values": values,
            "deltas": deltas,
        }
    return metrics

def _resolve_baseline_label(rows: Sequence[Mapping[str, Any]], baseline: int | str | None) -> str | None:
    """Resolve a baseline identifier (index, label, run ID, or model ID) to a row label."""
    if baseline is None:
        return None
    if isinstance(baseline, int):
        if baseline < 0 or baseline >= len(rows):
            raise ValueError(f"baseline index out of range: {baseline}")
        return str(rows[baseline].get("label"))
    baseline_text = str(baseline)
    for row in rows:
        if baseline_text in {str(row.get("label")), str(row.get("run_id")), str(row.get("model_id"))}:
            return str(row.get("label"))
    raise ValueError(f"baseline did not match a run label, run_id, or model_id: {baseline_text}")

def build_run_comparison(
    runs: Sequence[str | Path],
    *,
    labels: Sequence[str | None] | None = None,
    baseline: int | str | None = None,
    metric_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a structured run comparison payload from multiple run summaries.

    Args:
        runs: Paths to run summary or scorecard JSON files.
        labels: Optional per-run display labels; must match length of *runs*.
        baseline: Baseline identifier (0-based index, run label, run ID, or
            model ID) for delta computation; ``None`` disables deltas.
        metric_ids: Explicit metric list; ``None`` uses metrics present in every run.

    Returns:
        A comparison payload dict conforming to
        ``RUN_COMPARISON_SCHEMA_VERSION``.
    """
    if not runs:
        raise ValueError("at least one run summary path is required")
    if labels is not None and len(labels) != len(runs):
        raise ValueError("--label count must match run path count")

    loaded: list[tuple[dict[str, Any], Path]] = []
    for run_path in runs:
        source_path = _run_summary_path(run_path)
        loaded.append((load_run_summary(source_path), source_path))

    summaries = [summary for summary, _source_path in loaded]
    rows = _dedupe_labels(
        [
            _row_from_summary(
                index=index,
                summary=summary,
                source_path=source_path,
                label=labels[index] if labels is not None else None,
            )
            for index, (summary, source_path) in enumerate(loaded)
        ]
    )
    identities = [comparison_identity_from_summary(summary) for summary in summaries]
    incompatibilities, compatibility_warnings = compare_identities(identities)
    if incompatibilities:
        raise ValueError("runs are not comparable: " + "; ".join(incompatibilities))

    metric_ids_by_run = [set(map(str, _mapping(row.get("metrics")))) for row in rows]
    available_metric_ids = set().union(*metric_ids_by_run)
    common_metric_ids = set.intersection(*metric_ids_by_run) if metric_ids_by_run else set()
    selected_metric_ids = (
        [str(metric_id) for metric_id in metric_ids]
        if metric_ids is not None
        else sorted(common_metric_ids)
    )
    unknown_metric_ids = sorted(set(selected_metric_ids).difference(available_metric_ids))
    partially_available_metric_ids = sorted(
        set(selected_metric_ids).intersection(available_metric_ids).difference(common_metric_ids)
    )
    baseline_label = _resolve_baseline_label(rows, baseline)
    metric_matrix = _metric_matrix(rows=rows, metric_ids=selected_metric_ids, baseline_label=baseline_label)
    if baseline_label is not None:
        for row in rows:
            label = str(row.get("label"))
            row["delta_from_baseline"] = {
                metric_id: payload["deltas"][label]
                for metric_id, payload in metric_matrix.items()
                if label in payload["deltas"]
            }
    benchmarks = sorted({str(row["benchmark"]) for row in rows if row.get("benchmark")})
    datasets = sorted({str(row["dataset_id"]) for row in rows if row.get("dataset_id")})
    comparison_keys = sorted(
        {str(identity["comparison_key"]) for identity in identities if identity.get("comparison_key")}
    )

    return {
        "schema_version": RUN_COMPARISON_SCHEMA_VERSION,
        "run_count": len(rows),
        "baseline": baseline_label,
        "benchmarks": benchmarks,
        "datasets": datasets,
        "metric_ids": selected_metric_ids,
        "available_metric_ids": sorted(available_metric_ids),
        "common_metric_ids": sorted(common_metric_ids),
        "compatibility": {
            "status": "compatible_with_incomplete_identity" if compatibility_warnings else "compatible",
            "comparison_key": comparison_keys[0] if len(comparison_keys) == 1 else None,
            "comparison_keys": comparison_keys,
            "warnings": compatibility_warnings,
        },
        "runs": rows,
        "rows": rows,
        "metrics": metric_matrix,
        "best_by_metric": _best_by_metric(rows=rows, summaries=summaries, metric_ids=selected_metric_ids),
        "issues": [
            *[f"metric not found in any run: {metric_id}" for metric_id in unknown_metric_ids],
            *[
                f"metric not found in every run: {metric_id}"
                for metric_id in partially_available_metric_ids
            ],
        ],
        "artifacts": {},
    }

def write_run_comparison(
    runs: Sequence[str | Path],
    *,
    labels: Sequence[str | None] | None = None,
    baseline: int | str | None = None,
    metric_ids: Sequence[str] | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    """Build a run comparison and optionally write JSON + Markdown artifacts.

    Args:
        runs: Paths to run summary or scorecard JSON files.
        labels: Optional per-run display labels.
        baseline: Baseline identifier for delta computation.
        metric_ids: Explicit metric list.
        output_json: Destination for the comparison JSON file.
        output_md: Destination for the comparison Markdown file.

    Returns:
        The comparison payload with ``artifacts`` updated to include output
        paths for any written files.
    """
    comparison = build_run_comparison(runs, labels=labels, baseline=baseline, metric_ids=metric_ids)
    artifacts: dict[str, str] = {}
    if output_json is not None:
        artifacts["comparison_json"] = str(Path(output_json).resolve())
    if output_md is not None:
        artifacts["comparison_markdown"] = str(Path(output_md).resolve())
    comparison["artifacts"] = artifacts
    if output_json is not None:
        write_json(Path(output_json), comparison)
    if output_md is not None:
        write_text(Path(output_md), build_markdown_comparison(comparison))
    return comparison
