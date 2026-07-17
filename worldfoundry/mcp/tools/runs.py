"""Run lifecycle payloads for the WorldFoundry MCP server.

Provides payload-building functions for previewing, submitting, inspecting,
and cancelling evaluation runs.  Each function returns a structured dictionary
suitable for MCP tool responses and delegates command construction and job
management to the underlying WorldFoundry CLI and runtime helpers.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.cli.tui_discovery import build_model_benchmark_command, build_suite_command
from worldfoundry.runtime.jobs import python_module_command

from .context import DEFAULT_CONTEXT, MCPToolContext

# ── Preview and submission ──────────────────────────────────────────────


def preview_run_payload(
    *,
    model: str,
    benchmark: str | None = None,
    benchmarks: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    suite_ids: Sequence[str] | None = None,
    plan_only: bool = False,
    resume: bool = False,
    prepare: bool = False,
    execute_download: bool = False,
    data_root: str | Path | None = None,
    generation_cache_dir: str | Path | None = None,
    generation_cache_mode: str = "off",
    model_variant: str | None = None,
    requests_path: str | Path | None = None,
    task_name: str | None = None,
    generated_artifact_dir: str | Path | None = None,
    output_artifact: str | None = None,
    metrics: Sequence[str] | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Preview the command and output directory for a planned evaluation run.

    Builds the CLI command and resolves the output directory without actually
    submitting the job.  The returned payload contains both the display command
    (for human inspection) and the wrapped ``run_command`` (for actual
    execution via :func:`python_module_command`).

    Args:
        model: Model identifier.
        benchmark: Single benchmark identifier (mutually exclusive with ``benchmarks``).
        benchmarks: Multiple benchmark identifiers.
        output_dir: Custom output directory path.
        suite_ids: Suite identifiers for multi-benchmark runs.
        plan_only: Only produce the plan, do not execute.
        resume: Resume a previously started run.
        prepare: Prepare run environment before execution.
        execute_download: Allow download steps to run.
        data_root: Local dataset cache root.
        generation_cache_dir: Cache directory for generated outputs.
        generation_cache_mode: Cache mode — ``"off"``, ``"read"``, or ``"write"``.
        model_variant: Model variant selector.
        requests_path: Path to a requests manifest file.
        task_name: Single task name to evaluate.
        generated_artifact_dir: Directory of pre-generated artifacts.
        output_artifact: Output artifact filename.
        metrics: Explicit list of metrics to compute.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Returns:
        Dictionary with ``command``, ``command_text``, ``run_command`,
        ``run_command_text``, ``output_dir``, and ``plan_only`` keys.
    """
    ctx = context or DEFAULT_CONTEXT
    selected_benchmarks = _benchmark_ids(benchmark=benchmark, benchmarks=benchmarks)
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else _default_output_dir(ctx.output_root, model, selected_benchmarks)
    )
    display_command = _build_display_command(
        model=model,
        benchmarks=selected_benchmarks,
        output_dir=resolved_output_dir,
        suite_ids=tuple(suite_ids or ()),
        plan_only=plan_only,
        resume=resume,
        prepare=prepare,
        execute_download=execute_download,
        data_root=data_root,
        generation_cache_dir=generation_cache_dir,
        generation_cache_mode=generation_cache_mode,
        model_variant=model_variant,
        requests_path=requests_path,
        task_name=task_name,
        generated_artifact_dir=generated_artifact_dir,
        output_artifact=output_artifact,
        metrics=tuple(metrics or ()),
        context=ctx,
    )
    run_command = python_module_command(display_command)
    return {
        "command": list(display_command),
        "command_text": shlex.join(display_command),
        "run_command": list(run_command),
        "run_command_text": shlex.join(run_command),
        "output_dir": str(resolved_output_dir),
        "plan_only": plan_only,
    }


async def run_evaluation_payload(
    *,
    model: str,
    benchmark: str | None = None,
    benchmarks: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    suite_ids: Sequence[str] | None = None,
    plan_only: bool = False,
    resume: bool = False,
    prepare: bool = False,
    execute_download: bool = False,
    data_root: str | Path | None = None,
    generation_cache_dir: str | Path | None = None,
    generation_cache_mode: str = "off",
    model_variant: str | None = None,
    requests_path: str | Path | None = None,
    task_name: str | None = None,
    generated_artifact_dir: str | Path | None = None,
    output_artifact: str | None = None,
    metrics: Sequence[str] | None = None,
    wait: str = "auto",
    wait_timeout_s: int = 90,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Submit an evaluation run and optionally wait for completion.

    Delegates to :func:`preview_run_payload` for command construction, then
    submits the resulting ``run_command`` to the job store.  The ``wait``
    parameter controls whether the call blocks until the job finishes:

    * ``"async"`` — return immediately with a submitted summary.
    * ``"sync"`` — block until the job reaches a terminal state.
    * ``"auto"`` — block up to ``wait_timeout_s`` seconds, then return the
      current state if still running.

    Args:
        model: Model identifier.
        benchmark: Single benchmark identifier (mutually exclusive with ``benchmarks``).
        benchmarks: Multiple benchmark identifiers.
        tasks: Task identifiers (alternative to ``benchmarks``).
        output_dir: Custom output directory path.
        suite_ids: Suite identifiers for multi-benchmark runs.
        plan_only: Only produce the plan, do not execute.
        resume: Resume a previously started run.
        prepare: Prepare run environment before execution.
        execute_download: Allow download steps to run.
        data_root: Local dataset cache root.
        generation_cache_dir: Cache directory for generated outputs.
        generation_cache_mode: Cache mode — ``"off"``, ``"read"``, or ``"write"``.
        model_variant: Model variant selector.
        requests_path: Path to a requests manifest file.
        task_name: Single task name to evaluate.
        generated_artifact_dir: Directory of pre-generated artifacts.
        output_artifact: Output artifact filename.
        metrics: Explicit list of metrics to compute.
        wait: Wait strategy — ``"auto"``, ``"async"``, or ``"sync"``.
        wait_timeout_s: Seconds to wait before returning asynchronously.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``wait`` is not one of ``"auto"``, ``"async"``, or ``"sync"``.

    Returns:
        Run result dictionary (if completed) or submitted summary (if still
        running or ``wait`` is ``"async"``).
    """
    if wait not in {"auto", "async", "sync"}:
        raise ValueError("wait must be one of: auto, async, sync")
    selected_benchmarks = _benchmark_ids(benchmark=benchmark, benchmarks=benchmarks or tasks)
    ctx = context or DEFAULT_CONTEXT
    preview = preview_run_payload(
        model=model,
        benchmarks=selected_benchmarks,
        output_dir=output_dir,
        suite_ids=suite_ids,
        plan_only=plan_only,
        resume=resume,
        prepare=prepare,
        execute_download=execute_download,
        data_root=data_root,
        generation_cache_dir=generation_cache_dir,
        generation_cache_mode=generation_cache_mode,
        model_variant=model_variant,
        requests_path=requests_path,
        task_name=task_name,
        generated_artifact_dir=generated_artifact_dir,
        output_artifact=output_artifact,
        metrics=metrics,
        context=ctx,
    )
    output_path = Path(str(preview["output_dir"]))
    job = ctx.job_store.submit(
        preview["run_command"],
        display_command=preview["command"],
        output_dir=output_path,
        metadata={
            "model": model,
            "benchmarks": list(selected_benchmarks),
            "plan_only": plan_only,
            "surface": "mcp",
        },
    )
    if wait == "async":
        return _submitted(job.to_summary())

    timeout = None if wait == "sync" else max(0, wait_timeout_s)
    completed = await _wait_for_job(job.job_id, ctx, timeout)
    if completed:
        return ctx.job_store.get(job.job_id).to_result(include_logs=False)  # type: ignore[union-attr]
    return _submitted(job.to_summary())


# ── Run listing ─────────────────────────────────────────────────────────


def list_runs_payload(
    *,
    limit: int = 50,
    status: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List MCP-managed evaluation runs, newest first."""

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    ctx = context or DEFAULT_CONTEXT
    matched = [
        job.to_summary(log_tail=0)
        for job in ctx.job_store.list()
        if status is None or job.status == status
    ]
    return {
        "runs": matched[:limit],
        "total": len(matched),
        "limit": limit,
        "status": status,
    }


# ── Run inspection ──────────────────────────────────────────────────────


def get_run_status_payload(run_id: str, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Return the current status of a previously submitted run.

    Args:
        run_id: Unique run identifier returned by :func:`run_evaluation_payload`.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``run_id`` is not found in the job store.

    Returns:
        Run summary dictionary with a ``log_tail`` of the last 20 log lines.
    """
    ctx = context or DEFAULT_CONTEXT
    job = ctx.job_store.get(run_id)
    if job is None:
        raise ValueError(f"run not found: {run_id}")
    return job.to_summary(log_tail=20)


def get_run_result_payload(
    run_id: str,
    *,
    include_logs: bool = False,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return the result payload of a completed run.

    Args:
        run_id: Unique run identifier.
        include_logs: Include full log output in the result.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``run_id`` is not found in the job store.

    Returns:
        Run result dictionary, optionally including log lines.
    """
    ctx = context or DEFAULT_CONTEXT
    job = ctx.job_store.get(run_id)
    if job is None:
        raise ValueError(f"run not found: {run_id}")
    return job.to_result(include_logs=include_logs)


def get_run_samples_payload(
    run_id: str,
    *,
    task_name: str | None = None,
    offset: int = 0,
    limit: int = 50,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return sample-level results from a completed run.

    Reads a ``.jsonl`` samples file from the run's output directory and
    returns paginated sample records.

    Args:
        run_id: Unique run identifier.
        task_name: Filter samples by task name (optional).
        offset: Skip the first *offset* samples (must be non-negative).
        limit: Maximum number of samples to return (1–500).
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Raises:
        ValueError: If ``offset`` or ``limit`` are out of range, ``task_name``
            contains path separators, the run is not found, or no samples
            file exists for the run.

    Returns:
        Dictionary with ``samples``, ``total``, ``offset``, ``limit``, and
        ``source_path`` keys.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if task_name is not None and any(separator in task_name for separator in ("/", "\\")):
        raise ValueError("task_name cannot contain path separators")

    ctx = context or DEFAULT_CONTEXT
    job = ctx.job_store.get(run_id)
    if job is None:
        raise ValueError(f"run not found: {run_id}")
    if not job.output_dir:
        raise ValueError(f"run has no output directory: {run_id}")

    run_root = Path(job.output_dir).resolve()
    for path in _sample_file_candidates(run_root, task_name):
        if path.is_file():
            return _read_sample_jsonl(path, run_root=run_root, offset=offset, limit=limit)
    raise ValueError(f"samples file not found for run {run_id!r}")


# ── Run cancellation ────────────────────────────────────────────────────


async def cancel_run_payload(run_id: str, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Cancel a running or pending evaluation run.

    Args:
        run_id: Unique run identifier.
        context: Execution context (defaults to :data:`DEFAULT_CONTEXT`).

    Returns:
        Dictionary with ``success``, ``run_id``, and ``message`` keys.
    """
    ctx = context or DEFAULT_CONTEXT
    success, message = await ctx.job_store.cancel(run_id)
    return {"success": success, "run_id": run_id, "message": message}


# ── Internal helpers ───────────────────────────────────────────────────


def _build_display_command(
    *,
    model: str,
    benchmarks: Sequence[str],
    output_dir: Path,
    suite_ids: Sequence[str],
    plan_only: bool,
    resume: bool,
    prepare: bool,
    execute_download: bool,
    data_root: str | Path | None,
    generation_cache_dir: str | Path | None,
    generation_cache_mode: str,
    model_variant: str | None,
    requests_path: str | Path | None,
    task_name: str | None,
    generated_artifact_dir: str | Path | None,
    output_artifact: str | None,
    metrics: Sequence[str],
    context: MCPToolContext,
) -> tuple[str, ...]:
    """Build the CLI command tuple for an evaluation run.

    Selects :func:`build_model_benchmark_command` for single-benchmark runs
    or :func:`build_suite_command` for multi-benchmark runs, then appends
    optional flags such as ``--resume``, ``--prepare``, etc.

    Returns:
        Tuple of command-line argument strings.
    """
    if len(benchmarks) == 1 and not suite_ids:
        command = list(
            build_model_benchmark_command(
                model_id=model,
                benchmark_id=benchmarks[0],
                output_dir=output_dir,
                model_manifest_dir=context.model_manifest_dir,
                benchmark_manifest_dir=context.benchmark_manifest_dir,
                model_variant=model_variant,
                requests_path=requests_path,
                task_name=task_name,
                generated_artifact_dir=generated_artifact_dir,
                output_artifact=output_artifact,
                metrics=metrics,
                json_output=True,
            )
        )
    else:
        command = list(
            build_suite_command(
                output_dir=output_dir,
                model_ids=(model,),
                benchmark_ids=benchmarks,
                suite_ids=suite_ids,
                model_manifest_dir=context.model_manifest_dir,
                benchmark_manifest_dir=context.benchmark_manifest_dir,
                plan_only=plan_only,
            )
        )
        command.append("--json")
    if plan_only and "--plan-only" not in command:
        command.append("--plan-only")
    if resume:
        command.append("--resume")
    if prepare:
        command.append("--prepare")
    if execute_download:
        command.append("--execute-download")
    if data_root is not None:
        command.extend(["--data-root", str(data_root)])
    if generation_cache_dir is not None:
        command.extend(["--generation-cache-dir", str(generation_cache_dir)])
    if generation_cache_mode != "off":
        command.extend(["--generation-cache-mode", generation_cache_mode])
    return tuple(command)


def _benchmark_ids(*, benchmark: str | None, benchmarks: Sequence[str] | None) -> tuple[str, ...]:
    """Merge and deduplicate ``benchmark`` and ``benchmarks`` into a tuple.

    Raises:
        ValueError: If no benchmark/task identifiers are provided.

    Returns:
        Deduplicated tuple of benchmark identifier strings.
    """
    values = [str(item) for item in (benchmarks or ()) if str(item).strip()]
    if benchmark:
        values.insert(0, benchmark)
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    if not deduped:
        raise ValueError("at least one benchmark/task id is required")
    return tuple(deduped)


def _default_output_dir(root: Path, model: str, benchmarks: Sequence[str]) -> Path:
    """Resolve a default output directory from model and benchmark identifiers.

    Joins up to two benchmark labels with the model label; appends a
    ``plus_N`` suffix if more than two benchmarks are present.
    """
    label = "__".join((_safe_path_label(model), *(_safe_path_label(item) for item in benchmarks[:2])))
    if len(benchmarks) > 2:
        label = f"{label}__plus_{len(benchmarks) - 2}"
    return root / label


def _safe_path_label(value: str) -> str:
    """Sanitise a string for safe use as a path component.

    Replaces non-alphanumeric characters (except ``-``, ``_``, ``.``) with
    ``_`` and strips leading/trailing dots and underscores.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._") or "run"


def _sample_file_candidates(run_root: Path, task_name: str | None) -> tuple[Path, ...]:
    """Enumerate candidate ``.jsonl`` sample file paths inside a run directory.

    Checks generic filenames first, then task-specific ones if ``task_name``
    is provided.  Paths that escape ``run_root`` (path traversal) are
    silently skipped.
    """
    names = ["results.jsonl", "raw_results.jsonl", "samples.jsonl"]
    if task_name:
        names.extend([f"{task_name}.jsonl", f"samples_{task_name}.jsonl"])
    candidates: list[Path] = []
    for name in names:
        path = (run_root / name).resolve()
        try:
            path.relative_to(run_root)
        except ValueError:
            continue
        candidates.append(path)
    return tuple(dict.fromkeys(candidates))


def _read_sample_jsonl(path: Path, *, run_root: Path, offset: int, limit: int) -> dict[str, Any]:
    """Read a paginated slice of JSON-Lines samples from a file.

    Args:
        path: Path to the ``.jsonl`` file.
        run_root: Root directory of the run (used to compute ``source_path``).
        offset: Skip the first *offset* valid samples.
        limit: Return at most *limit* samples.

    Raises:
        ValueError: If ``path`` escapes ``run_root`` (path traversal guard).

    Returns:
        Dictionary with ``samples``, ``total``, ``offset``, ``limit``, and
        ``source_path`` keys.
    """
    path = path.resolve()
    try:
        source_path = str(path.relative_to(run_root))
    except ValueError as exc:
        raise ValueError("samples path escapes run output directory") from exc

    samples: list[dict[str, Any]] = []
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(sample, dict):
                continue
            if total >= offset and len(samples) < limit:
                samples.append(sample)
            total += 1
    return {
        "samples": samples,
        "total": total,
        "offset": offset,
        "limit": limit,
        "source_path": source_path,
    }


def _submitted(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Format a job-store summary into a user-facing "submitted" payload."""
    return {
        "run_id": summary["job_id"],
        "status": summary["status"],
        "message": f"Run {summary['job_id']} is {summary['status']}. Use get_run_status/get_run_result to inspect it.",
        "output_dir": summary.get("output_dir"),
        "command": summary.get("command"),
    }


async def _wait_for_job(job_id: str, context: MCPToolContext, timeout_s: int | None) -> bool:
    """Poll the job store until the job is terminal or the timeout expires.

    Args:
        job_id: Unique job identifier.
        context: Execution context with a :attr:`job_store`.
        timeout_s: Maximum wait in seconds, or **None** for unlimited wait.

    Returns:
        **True** if the job reached a terminal state, **False** if the
        timeout expired first.
    """
    elapsed = 0.0
    interval = 0.5
    while True:
        job = context.job_store.get(job_id)
        if job is None or job.terminal:
            return True
        if timeout_s is not None and elapsed >= timeout_s:
            return False
        await asyncio.sleep(interval)
        elapsed += interval


__all__ = [
    "cancel_run_payload",
    "get_run_result_payload",
    "get_run_samples_payload",
    "get_run_status_payload",
    "list_runs_payload",
    "preview_run_payload",
    "run_evaluation_payload",
]
