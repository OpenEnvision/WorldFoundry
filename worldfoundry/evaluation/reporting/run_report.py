"""Build, load, and render compact run summaries and Markdown reports.

A *run summary* is a normalized subset of a full scorecard, focused on the
fields needed for cross-run comparison and index aggregation.  This module
also produces a human-readable Markdown report from a summary payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.utils import (
    escape_markdown_cell,
    format_value,
    mapping_or_empty,
    read_json_object,
    write_json,
    write_text,
)

from .comparison_identity import build_comparison_identity, comparison_identity_from_summary


RUN_SUMMARY_SCHEMA_VERSION = "worldfoundry-run-summary"

# ── Aliases for shared utility functions ──────────────────────
_mapping = mapping_or_empty
_format_value = format_value
_escape_markdown_cell = escape_markdown_cell


# ── Internal helpers ─────────────────────────────────────────
def _first_present(payload: Mapping[str, Any], *keys: str, default: str = "") -> str:
    """Return the first non-empty value found under *keys* in *payload*."""
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _int_or_zero(*values: Any) -> int:
    """Return the first integer-convertible value from *values*, or ``0``."""
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value))
        except ValueError:
            continue
    return 0


def _run_summary_path(path: str | Path) -> Path:
    """Resolve *path* to an actual summary/scorecard file, checking directories for ``summary.json``."""
    source = Path(path)
    if source.is_dir():
        for candidate in (source / "summary.json", source / "scorecard.json"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"run directory does not contain summary.json or scorecard.json: {source}")
    if not source.exists():
        raise FileNotFoundError(f"run summary path does not exist: {source}")
    return source


def _run_summary_candidate(run_dir: Path) -> Path | None:
    """Return the first existing summary/scorecard file in *run_dir*, or ``None``."""
    for candidate in (run_dir / "summary.json", run_dir / "scorecard.json"):
        if candidate.is_file():
            return candidate
    return None


def _normalise_roots(roots: str | Path | Sequence[str | Path]) -> list[Path]:
    """Convert a single root or a sequence into a list of ``Path`` objects."""
    if isinstance(roots, (str, Path)):
        return [Path(roots)]
    return [Path(root) for root in roots]


def _number_or_none(value: Any) -> float | int | None:
    """Return *value* as a number if it is numeric and not a bool; otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _label_for_summary(summary: Mapping[str, Any], source_path: Path, explicit_label: str | None) -> str:
    """Choose a display label from explicit hint, run ID, model ID, or path name."""
    if explicit_label:
        return explicit_label
    run = _mapping(summary.get("run"))
    model = _mapping(summary.get("model"))
    for value in (run.get("run_id"), model.get("model_id"), model.get("model_name")):
        if value not in (None, ""):
            return str(value)
    if source_path.name == "summary.json" or source_path.name == "scorecard.json":
        return source_path.parent.name
    return source_path.stem


def _dedupe_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every row has a unique label, appending ``#N`` suffixes for duplicates."""
    seen: dict[str, int] = {}
    deduped = []
    for row in rows:
        item = dict(row)
        base_label = str(item.get("label") or f"run-{item.get('index', len(deduped))}")
        count = seen.get(base_label, 0) + 1
        seen[base_label] = count
        item["label"] = base_label if count == 1 else f"{base_label}#{count}"
        deduped.append(item)
    return deduped


def _row_from_summary(
    *,
    index: int,
    summary: Mapping[str, Any],
    source_path: Path,
    label: str | None,
) -> dict[str, Any]:
    """Flatten a run summary into a compact comparison-friendly row dict."""
    run = _mapping(summary.get("run"))
    benchmark = _mapping(summary.get("benchmark"))
    model = _mapping(summary.get("model"))
    dataset = _mapping(summary.get("dataset"))
    counts = _mapping(summary.get("counts"))
    eligibility = _mapping(summary.get("eligibility"))
    leaderboard = _mapping(summary.get("leaderboard"))
    artifacts = _mapping(summary.get("artifacts"))
    generation_cache = _mapping(run.get("generation_cache") or summary.get("generation_cache"))
    comparison_identity = comparison_identity_from_summary(summary)
    evaluation = _mapping(summary.get("evaluation"))

    return {
        "index": index,
        "label": _label_for_summary(summary, source_path, label),
        "source_path": str(source_path.resolve()),
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "benchmark": _first_present(benchmark, "benchmark_name", "benchmark_id", "name", "id"),
        "task_type": benchmark.get("task_type"),
        "model_id": model.get("model_id"),
        "model_name": model.get("model_name"),
        "dataset_id": _first_present(dataset, "dataset_id", "name", "id"),
        "evaluation_mode": evaluation.get("mode") or comparison_identity.get("evaluation_mode"),
        "protocol_id": comparison_identity.get("protocol_id"),
        "protocol_fidelity": comparison_identity.get("protocol_fidelity"),
        "data_fidelity": comparison_identity.get("data_fidelity"),
        "comparison_key": comparison_identity.get("comparison_key"),
        "comparison_identity_status": comparison_identity.get("status"),
        "comparison_identity": comparison_identity,
        "sample_count": _int_or_zero(counts.get("sample_count")),
        "successful_samples": _int_or_zero(counts.get("successful_samples")),
        "failed_samples": _int_or_zero(counts.get("failed_samples")),
        "score_valid": eligibility.get("score_valid"),
        "leaderboard_valid": eligibility.get("leaderboard_valid"),
        "leaderboard_eligible": eligibility.get("leaderboard_eligible"),
        "metrics": dict(leaderboard),
        "artifacts": dict(artifacts),
        "generation_cache": dict(generation_cache),
    }


def build_run_summary(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact root-level run summary from a normalized scorecard."""

    run = _mapping(scorecard.get("run"))
    benchmark = _mapping(scorecard.get("benchmark"))
    model = _mapping(scorecard.get("model"))
    dataset = _mapping(scorecard.get("dataset"))
    generation = _mapping(scorecard.get("generation"))
    metrics = _mapping(scorecard.get("metrics"))
    metrics_summary = _mapping(metrics.get("summary"))
    eligibility = _mapping(scorecard.get("eligibility"))
    artifacts = _mapping(scorecard.get("artifacts"))
    provenance = _mapping(scorecard.get("provenance"))
    evaluation = _mapping(scorecard.get("evaluation"))
    comparison_identity = build_comparison_identity(
        benchmark=benchmark,
        dataset=dataset,
        metrics=metrics,
        provenance=provenance,
        evaluation_kind=str(evaluation.get("mode") or evaluation.get("kind") or "unknown"),
        explicit=_mapping(scorecard.get("comparison_identity")),
    )

    sample_count = _int_or_zero(
        metrics_summary.get("sample_count")
        or generation.get("num_requests")
        or dataset.get("sample_count")
    )
    successful_samples = _int_or_zero(
        metrics_summary.get("successful_samples")
        or generation.get("successful")
    )
    failed_samples = _int_or_zero(
        metrics_summary.get("failed_samples")
        or generation.get("failed")
    )

    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "source_schema_version": scorecard.get("schema_version"),
        "run": {
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "worldfoundry_version": run.get("worldfoundry_version"),
            "run_fingerprint": run.get("run_fingerprint"),
        },
        "benchmark": {
            "benchmark_id": _first_present(benchmark, "benchmark_id", "id", "benchmark_name", "name"),
            "benchmark_name": _first_present(benchmark, "benchmark_name", "name", "id"),
            "benchmark_revision": _first_present(
                benchmark,
                "benchmark_revision",
                "revision",
                "repo_revision",
                "upstream_revision",
            ),
            "task_type": benchmark.get("task_type"),
            "suite": benchmark.get("suite"),
            "evaluation_protocol": benchmark.get("evaluation_protocol"),
            "protocol_id": comparison_identity.get("protocol_id"),
            "protocol_revision": comparison_identity.get("protocol_revision"),
            "protocol_config_hash": comparison_identity.get("protocol_config_hash"),
        },
        "model": {
            "model_id": _first_present(model, "model_id", "model_name", "name"),
            "model_name": _first_present(model, "model_name", "model_id", "name"),
            "model_type": model.get("model_type"),
        },
        "dataset": {
            "dataset_id": _first_present(dataset, "dataset_id", "name", "id"),
            "name": _first_present(dataset, "name", "dataset_id", "id"),
            "split": dataset.get("split"),
            "sample_count": dataset.get("sample_count"),
            "dataset_revision": comparison_identity.get("dataset_revision"),
            "dataset_hash": comparison_identity.get("dataset_hash"),
        },
        "evaluation": {
            "kind": evaluation.get("kind"),
            "mode": comparison_identity.get("evaluation_mode"),
        },
        "provenance": provenance,
        "comparison_identity": comparison_identity,
        "counts": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": list(metrics_summary.get("failed_sample_ids") or ()),
        },
        "generation": generation,
        "metrics": {
            "leaderboard": dict(metrics.get("leaderboard") or {}),
            "per_metric": dict(metrics.get("per_metric") or {}),
            "summary": metrics_summary,
        },
        "leaderboard": dict(metrics.get("leaderboard") or {}),
        "eligibility": {
            "score_valid": eligibility.get("score_valid"),
            "leaderboard_valid": eligibility.get("leaderboard_valid"),
            "leaderboard_eligible": eligibility.get("leaderboard_eligible"),
            "reasons": list(eligibility.get("reasons") or ()),
            "blocking_reasons": list(eligibility.get("blocking_reasons") or ()),
        },
        "artifacts": artifacts,
    }


def load_run_summary(path: str | Path) -> dict[str, Any]:
    """Load a compact run summary from a run directory, summary.json, or scorecard.json."""

    source_path = _run_summary_path(path)
    payload = read_json_object(source_path)
    schema_version = payload.get("schema_version")
    if schema_version == RUN_SUMMARY_SCHEMA_VERSION:
        return payload
    if schema_version == "worldfoundry-scorecard":
        return build_run_summary(payload)
    raise ValueError(f"unsupported run summary schema_version in {source_path}: {schema_version!r}")


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    """Render a run summary as a human-readable Markdown report."""
    run = _mapping(summary.get("run"))
    benchmark = _mapping(summary.get("benchmark"))
    model = _mapping(summary.get("model"))
    dataset = _mapping(summary.get("dataset"))
    counts = _mapping(summary.get("counts"))
    eligibility = _mapping(summary.get("eligibility"))
    leaderboard = _mapping(summary.get("leaderboard"))
    artifacts = _mapping(summary.get("artifacts"))
    evaluation = _mapping(summary.get("evaluation"))
    comparison_identity = comparison_identity_from_summary(summary)

    lines = [
        "# WorldFoundry Run Report",
        "",
        f"- Run ID: {_format_value(run.get('run_id'))}",
        f"- Status: {_format_value(run.get('status'))}",
        f"- Benchmark: {_format_value(benchmark.get('benchmark_name'))}",
        f"- Model: {_format_value(model.get('model_id') or model.get('model_name'))}",
        f"- Dataset: {_format_value(dataset.get('dataset_id') or dataset.get('name'))}",
        f"- Evaluation mode: {_format_value(evaluation.get('mode') or comparison_identity.get('evaluation_mode'))}",
        f"- Protocol: {_format_value(comparison_identity.get('protocol_id'))}",
        f"- Protocol fidelity: {_format_value(comparison_identity.get('protocol_fidelity'))}",
        f"- Data fidelity: {_format_value(comparison_identity.get('data_fidelity'))}",
        f"- Comparison identity: {_format_value(comparison_identity.get('status'))}",
        (
            "- Samples: "
            f"{_format_value(counts.get('sample_count'))} total, "
            f"{_format_value(counts.get('successful_samples'))} succeeded, "
            f"{_format_value(counts.get('failed_samples'))} failed"
        ),
        f"- Score valid: {_format_value(eligibility.get('score_valid'))}",
        f"- Leaderboard valid: {_format_value(eligibility.get('leaderboard_valid'))}",
        "",
        "## Leaderboard",
        "",
    ]

    if leaderboard:
        lines.extend(["| Metric | Value |", "| --- | ---: |"])
        for metric_id, value in sorted(leaderboard.items()):
            lines.append(f"| {_escape_markdown_cell(metric_id)} | {_escape_markdown_cell(value)} |")
    else:
        lines.append("No leaderboard metrics were produced.")

    failed_sample_ids = list(counts.get("failed_sample_ids") or ())
    if failed_sample_ids:
        lines.extend(["", "## Failed Samples", ""])
        for sample_id in failed_sample_ids[:50]:
            lines.append(f"- {_format_value(sample_id)}")
        if len(failed_sample_ids) > 50:
            lines.append(f"- ... {len(failed_sample_ids) - 50} more")

    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        lines.extend(["| Name | Path |", "| --- | --- |"])
        for name, path in sorted(artifacts.items()):
            lines.append(f"| {_escape_markdown_cell(name)} | `{_escape_markdown_cell(path)}` |")
    else:
        lines.append("No artifacts were indexed.")

    return "\n".join(lines).rstrip() + "\n"


def write_run_report_artifacts(
    *,
    output_dir: str | Path,
    scorecard_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    scorecard: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Build and write ``summary.json`` and ``report.md`` from a scorecard.

    Args:
        output_dir: Run output directory used for default file paths.
        scorecard_path: Explicit path to the scorecard JSON file.
        summary_path: Explicit destination for the summary JSON file.
        report_path: Explicit destination for the Markdown report file.
        scorecard: In-memory scorecard payload; overrides *scorecard_path*.

    Returns:
        A dict mapping ``"summary"`` and ``"report"`` to resolved ``Path`` objects.
    """
    root = Path(output_dir)
    scorecard_payload = dict(scorecard) if scorecard is not None else read_json_object(Path(scorecard_path or root / "scorecard.json"))
    summary = build_run_summary(scorecard_payload)
    resolved_summary_path = Path(summary_path or root / "summary.json")
    resolved_report_path = Path(report_path or root / "report.md")

    write_json(resolved_summary_path, summary)
    write_text(resolved_report_path, build_markdown_report(summary))
    return {
        "summary": resolved_summary_path.resolve(),
        "report": resolved_report_path.resolve(),
    }


__all__ = [
    "RUN_SUMMARY_SCHEMA_VERSION",
    "build_markdown_report",
    "build_run_summary",
    "write_run_report_artifacts",
]
