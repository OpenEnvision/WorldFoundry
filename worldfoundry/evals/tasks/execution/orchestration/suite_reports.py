"""Suite summary, scorecard, comparison, and Markdown report writers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evals.utils import (
    write_json,
    write_jsonl,
    write_text,
)

from .suite_requests import MODEL_BENCHMARK_SUITE_SCORECARDS_SCHEMA_VERSION


def _suite_summary(cells: Sequence[Mapping[str, Any]], *, execute: bool) -> dict[str, Any]:
    """Aggregates execution states (succeeded/failed/skipped) across all matrix cells."""
    counts: dict[str, int] = {}
    for cell in cells:
        status = str(cell.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "total": len(cells),
        "execute": execute,
        "planned": counts.get("planned", 0),
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "resident_model_loads": sum(
            1
            for cell in cells
            if cell.get("status") not in {"planned", "skipped"} and cell.get("runner_reused") is False
        ),
        "runner_reuses": sum(1 for cell in cells if cell.get("runner_reused") is True),
        "runner_state_resets": sum(1 for cell in cells if cell.get("runner_state_reset") is True),
        "generation_batches_reused": sum(int(cell.get("generation_batches_reused") or 0) for cell in cells),
        "generation_requests_reused": sum(int(cell.get("generation_requests_reused") or 0) for cell in cells),
        "generation_requests_executed": sum(int(cell.get("generation_requests_executed") or 0) for cell in cells),
        "status_counts": counts,
        "models": sorted({str(cell.get("model_id")) for cell in cells if cell.get("model_id")}),
        "benchmarks": sorted({str(cell.get("benchmark_id")) for cell in cells if cell.get("benchmark_id")}),
    }


def build_markdown_suite_report(payload: Mapping[str, Any]) -> str:
    """Generates a human-readable Markdown summary representing the entire matrix sweep."""
    summary = dict(payload.get("summary") or {})
    lines = [
        "# WorldFoundry Model x Benchmark Suite",
        "",
        f"- Status: {payload.get('status')}",
        f"- Total cells: {summary.get('total', 0)}",
        f"- Succeeded: {summary.get('succeeded', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Skipped: {summary.get('skipped', 0)}",
        f"- Resident model loads: {summary.get('resident_model_loads', 0)}",
        f"- Runner reuses: {summary.get('runner_reuses', 0)}",
        f"- Runner state resets: {summary.get('runner_state_resets', 0)}",
        f"- Generation batches reused: {summary.get('generation_batches_reused', 0)}",
        f"- Generation requests reused: {summary.get('generation_requests_reused', 0)}",
        f"- Generation requests executed: {summary.get('generation_requests_executed', 0)}",
        f"- Model workers: {summary.get('model_workers_used', 0)} / {summary.get('model_workers_requested', 1)}",
        f"- Parallel model execution: {summary.get('parallel_model_execution', False)}",
        f"- Worker CUDA groups: {', '.join(summary.get('worker_cuda_device_groups') or ()) or 'inherited'}",
        "",
        "| Model | Benchmark | Artifact | Compatibility | Status | Run |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in payload.get("cells") or ():
        if not isinstance(cell, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in (
                    cell.get("model_id", ""),
                    cell.get("benchmark_id", ""),
                    cell.get("output_artifact", ""),
                    cell.get("compatibility", ""),
                    cell.get("status", ""),
                    cell.get("run_dir", ""),
                )
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _suite_scorecard_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collect per-cell scorecard paths for the suite index."""
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        artifacts = dict(cell.get("artifacts") or {})
        benchmark_scorecard = cell.get("benchmark_scorecard_path") or artifacts.get("benchmark_scorecard")
        generation_scorecard = cell.get("generation_scorecard_path") or artifacts.get("generation_scorecard")
        if benchmark_scorecard in (None, "") and generation_scorecard in (None, ""):
            continue
        rows.append(
            {
                "index": index,
                "model_id": cell.get("model_id"),
                "requested_model_id": cell.get("requested_model_id"),
                "benchmark_id": cell.get("benchmark_id"),
                "status": cell.get("status"),
                "run_dir": cell.get("run_dir"),
                "run_summary_path": cell.get("run_summary_path"),
                "benchmark_scorecard_path": benchmark_scorecard,
                "generation_scorecard_path": generation_scorecard,
            }
        )
    return rows


def _run_summary_paths_and_labels(cells: Sequence[Mapping[str, Any]]) -> tuple[list[Path], list[str]]:
    """Extract ``run_summary`` paths and ``model:benchmark`` labels."""
    paths: list[Path] = []
    labels: list[str] = []
    for cell in cells:
        summary_path = cell.get("run_summary_path")
        if summary_path in (None, ""):
            continue
        path = Path(str(summary_path))
        if not path.is_file():
            continue
        paths.append(path)
        labels.append(f"{cell.get('model_id')}:{cell.get('benchmark_id')}")
    return paths, labels


def _write_empty_comparison(
    output_json: Path,
    output_md: Path,
    *,
    issue: str = "no completed run summaries found",
) -> dict[str, Any]:
    """Write a valid empty comparison artifact with an actionable reason."""

    from worldfoundry.evals.reporting import RUN_COMPARISON_SCHEMA_VERSION, build_markdown_comparison

    payload = {
        "schema_version": RUN_COMPARISON_SCHEMA_VERSION,
        "run_count": 0,
        "baseline": None,
        "benchmarks": [],
        "datasets": [],
        "metric_ids": [],
        "available_metric_ids": [],
        "common_metric_ids": [],
        "runs": [],
        "rows": [],
        "metrics": {},
        "best_by_metric": {},
        "issues": [issue],
        "artifacts": {
            "comparison_json": str(output_json.resolve()),
            "comparison_markdown": str(output_md.resolve()),
        },
    }
    write_json(output_json, payload)
    write_text(output_md, build_markdown_comparison(payload))
    return payload


def _write_suite_artifacts(root: Path, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write suite scorecard index, run browser, and comparison artifacts."""
    from worldfoundry.evals.reporting import (
        build_markdown_run_index,
        write_run_browser,
        write_run_comparison,
        write_run_index,
    )

    scorecard_rows = _suite_scorecard_rows(cells)
    scorecards_json = root / "scorecards" / "scorecards.json"
    scorecards_jsonl = root / "scorecards" / "scorecards.jsonl"
    scorecards_payload = {
        "schema_version": MODEL_BENCHMARK_SUITE_SCORECARDS_SCHEMA_VERSION,
        "scorecard_count": len(scorecard_rows),
        "rows": scorecard_rows,
    }
    write_json(scorecards_json, scorecards_payload)
    write_jsonl(scorecards_jsonl, scorecard_rows, atomic=False)

    summary_paths, labels = _run_summary_paths_and_labels(cells)
    index_json = root / "index" / "index.json"
    index_jsonl = root / "index" / "index.jsonl"
    index_md = root / "index" / "index.md"
    index_html = root / "index" / "index.html"
    index_roots: Sequence[str | Path] = summary_paths if summary_paths else (root,)
    index = write_run_index(index_roots, output_json=index_json, output_jsonl=index_jsonl)
    write_text(index_md, build_markdown_run_index(index))
    write_run_browser(index, index_html)

    comparison_json = root / "comparison" / "comparison.json"
    comparison_md = root / "comparison" / "comparison.md"
    completed_benchmark_ids = {
        str(cell.get("benchmark_id"))
        for cell in cells
        if cell.get("run_summary_path") and Path(str(cell["run_summary_path"])).is_file()
    }
    if len(completed_benchmark_ids) > 1:
        comparison = _write_empty_comparison(
            comparison_json,
            comparison_md,
            issue="suite spans multiple benchmarks; select one benchmark from index.json before comparing runs",
        )
    elif summary_paths:
        try:
            comparison = write_run_comparison(
                summary_paths,
                labels=labels,
                output_json=comparison_json,
                output_md=comparison_md,
            )
        except ValueError as exc:
            comparison = _write_empty_comparison(comparison_json, comparison_md, issue=str(exc))
    else:
        comparison = _write_empty_comparison(comparison_json, comparison_md)

    return {
        "scorecards_json": str(scorecards_json),
        "scorecards_jsonl": str(scorecards_jsonl),
        "index_json": str(index_json),
        "index_jsonl": str(index_jsonl),
        "index_markdown": str(index_md),
        "index_html": str(index_html),
        "comparison_json": str(comparison_json),
        "comparison_markdown": str(comparison_md),
        "scorecard_count": len(scorecard_rows),
        "indexed_run_count": index.get("run_count", 0),
        "comparison_run_count": comparison.get("run_count", 0),
    }
