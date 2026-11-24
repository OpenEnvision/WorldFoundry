"""Run lifecycle payloads for the WorldFoundry MCP server.

Previews, submits, inspects, and cancels evaluation runs. Commands are built
with the same builders the TUI uses (:mod:`worldfoundry.cli.command_builders`)
and executed by the shared :class:`~worldfoundry.runtime.jobs.AsyncCommandJobStore`.
Every run payload identifies the run by ``run_id``.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field

from worldfoundry.cli.command_builders import build_model_benchmark_command, build_suite_command
from worldfoundry.runtime.jobs import TERMINAL_JOB_STATUSES, python_module_command

from .context import MCPToolContext, get_default_context
from .responses import ToolNotFoundError
from .spec import mcp_tool

BenchmarkMode = Literal["official-run", "official-validation", "normalizer", "contract"]
GenerationCacheMode = Literal["off", "read", "write", "read-write", "refresh"]
WaitStrategy = Literal["auto", "async", "sync"]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

# Shared parameter declarations: preview_run and evaluate accept the same selection surface.
Model = Annotated[str, Field(description="Model id or alias from list_models.")]
Benchmarks = Annotated[
    list[str] | None,
    Field(description="Benchmark ids from list_benchmarks. One id runs a single cell; several run a suite."),
]
Suites = Annotated[list[str] | None, Field(description="Named model x benchmark suite presets.")]
Mode = Annotated[BenchmarkMode, Field(description="Benchmark runner mode.")]
OutputDir = Annotated[
    str | None, Field(description="Run output directory; defaults to a per-run directory under the MCP run root.")
]
PlanOnly = Annotated[bool, Field(description="Write the run plan without executing model or benchmark work.")]
Resume = Annotated[bool, Field(description="Reuse completed generation and metric results when inputs match.")]
CacheDir = Annotated[str | None, Field(description="Generation-result cache directory.")]
CacheMode = Annotated[GenerationCacheMode, Field(description="Generation-result cache mode.")]
ModelVariant = Annotated[str | None, Field(description="Model variant id from get_model_info.")]
RequestsPath = Annotated[
    str | None, Field(description="JSON/JSONL GenerationRequest file to run instead of the benchmark samples.")
]
TaskName = Annotated[str | None, Field(description="Task name from list_tasks to materialize requests from.")]
ArtifactDir = Annotated[
    str | None, Field(description="Score artifacts already generated in this directory instead of running the model.")
]
OutputArtifact = Annotated[str | None, Field(description="Artifact name the benchmark should score.")]
Metrics = Annotated[
    list[str] | None, Field(description="Metric ids from list_metrics; defaults to the benchmark's metrics.")
]
RunId = Annotated[str, Field(description="Run id returned by evaluate or list_runs.")]

# ── Preview and submission ──────────────────────────────────────────────


@mcp_tool("preview_run", title="Preview an evaluation run", read_only=True)
def preview_run_payload(
    *,
    model: Model,
    benchmarks: Benchmarks = None,
    suite_ids: Suites = None,
    mode: Mode = "official-run",
    output_dir: OutputDir = None,
    plan_only: PlanOnly = False,
    resume: Resume = False,
    generation_cache_dir: CacheDir = None,
    generation_cache_mode: CacheMode = "off",
    model_variant: ModelVariant = None,
    requests_path: RequestsPath = None,
    task_name: TaskName = None,
    generated_artifact_dir: ArtifactDir = None,
    output_artifact: OutputArtifact = None,
    metrics: Metrics = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Show the exact ``wf`` command and output directory ``evaluate`` would use, without running it.

    Returns ``command`` (the ``wf`` invocation), ``run_command`` (how the server launches it),
    ``output_dir``, and ``plan_only``.
    """
    ctx = context or get_default_context()
    selected = _benchmark_ids(benchmarks, suite_ids)
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else _default_output_dir(ctx.output_root, model, selected)
    )
    display_command = _build_display_command(
        model=model,
        benchmarks=selected,
        suite_ids=tuple(suite_ids or ()),
        mode=mode,
        output_dir=resolved_output_dir,
        plan_only=plan_only,
        resume=resume,
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


@mcp_tool("evaluate", title="Run an evaluation", read_only=False, idempotent=False)
async def run_evaluation_payload(
    *,
    model: Model,
    benchmarks: Benchmarks = None,
    suite_ids: Suites = None,
    mode: Mode = "official-run",
    output_dir: OutputDir = None,
    plan_only: PlanOnly = False,
    resume: Resume = False,
    generation_cache_dir: CacheDir = None,
    generation_cache_mode: CacheMode = "off",
    model_variant: ModelVariant = None,
    requests_path: RequestsPath = None,
    task_name: TaskName = None,
    generated_artifact_dir: ArtifactDir = None,
    output_artifact: OutputArtifact = None,
    metrics: Metrics = None,
    wait: Annotated[
        WaitStrategy,
        Field(description="async: return immediately; sync: wait until done; auto: wait up to wait_timeout_s."),
    ] = "auto",
    wait_timeout_s: Annotated[int, Field(ge=0, description="Seconds to wait when wait='auto'.")] = 90,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Submit an evaluation run for one model on one or more benchmarks.

    Returns the run summary with ``run_id`` and ``status``; when the run finishes within the wait
    budget, the summary also carries its result fields. Track unfinished runs with ``get_run_status``.
    """
    if wait not in {"auto", "async", "sync"}:
        raise ValueError("wait must be one of: auto, async, sync")
    ctx = context or get_default_context()
    selected = _benchmark_ids(benchmarks, suite_ids)
    run_id = uuid4().hex[:12]
    resolved_output_dir = (
        output_dir if output_dir is not None else _default_output_dir(ctx.output_root, model, selected, run_id=run_id)
    )
    preview = preview_run_payload(
        model=model,
        benchmarks=selected,
        suite_ids=suite_ids,
        mode=mode,
        output_dir=resolved_output_dir,
        plan_only=plan_only,
        resume=resume,
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
    job = ctx.job_store.submit(
        preview["run_command"],
        display_command=preview["command"],
        output_dir=Path(str(preview["output_dir"])),
        metadata={"model": model, "benchmarks": list(selected), "plan_only": plan_only, "surface": "mcp"},
        job_id=run_id,
    )
    if wait != "async":
        timeout = None if wait == "sync" else max(0, wait_timeout_s)
        if await _wait_for_job(job.job_id, ctx, timeout):
            return _run_view(job.to_result(include_logs=False))
    return _run_view(
        job.to_summary(log_tail=0),
        message=f"Run {job.job_id} is {job.status}. Track it with get_run_status and get_run_result.",
    )


# ── Run listing and inspection ──────────────────────────────────────────


@mcp_tool("list_runs", title="List evaluation runs", read_only=True)
def list_runs_payload(
    *,
    limit: Annotated[int, Field(ge=1, le=500, description="Maximum runs to return.")] = 50,
    status: Annotated[RunStatus | None, Field(description="Only runs in this lifecycle status.")] = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """List runs started through this MCP server, newest first."""

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    ctx = context or get_default_context()
    matched = [
        _run_view(job.to_summary(log_tail=0)) for job in ctx.job_store.list() if status is None or job.status == status
    ]
    return {"runs": matched[:limit], "total": len(matched), "limit": limit, "status": status}


@mcp_tool("get_run_status", title="Get run status", read_only=True)
def get_run_status_payload(run_id: RunId, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Return a run's status, timing, and the last 20 log lines."""

    return _run_view(_get_job(run_id, context).to_summary(log_tail=20))


@mcp_tool("get_run_result", title="Get run result", read_only=True)
def get_run_result_payload(
    run_id: RunId,
    *,
    include_logs: Annotated[bool, Field(description="Include the full run log.")] = False,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return a run's result: exit code, output paths, and the parsed CLI JSON result when finished."""

    return _run_view(_get_job(run_id, context).to_result(include_logs=include_logs))


@mcp_tool("get_run_samples", title="Get per-sample results", read_only=True)
def get_run_samples_payload(
    run_id: RunId,
    *,
    task_name: Annotated[str | None, Field(description="Read the samples file for this task.")] = None,
    offset: Annotated[int, Field(ge=0, description="Skip this many samples.")] = 0,
    limit: Annotated[int, Field(ge=1, le=500, description="Maximum samples to return.")] = 50,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Return a page of per-sample results from a run's ``results.jsonl`` (or task samples file)."""

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if task_name is not None and any(separator in task_name for separator in ("/", "\\")):
        raise ValueError("task_name cannot contain path separators")

    job = _get_job(run_id, context)
    if not job.output_dir:
        raise ToolNotFoundError(f"run has no output directory: {run_id}")
    run_root = Path(job.output_dir).resolve()
    for path in _sample_file_candidates(run_root, task_name):
        if path.is_file():
            return {"run_id": run_id, **_read_sample_jsonl(path, run_root=run_root, offset=offset, limit=limit)}
    raise ToolNotFoundError(f"samples file not found for run {run_id!r}; the run may still be in progress")


# ── Run cancellation ────────────────────────────────────────────────────


@mcp_tool("cancel_run", title="Cancel a run", read_only=False, destructive=True, idempotent=True)
async def cancel_run_payload(run_id: RunId, *, context: MCPToolContext | None = None) -> dict[str, Any]:
    """Cancel a queued or running evaluation run; ``cancelled`` is false when it had already finished."""

    ctx = context or get_default_context()
    job = _get_job(run_id, ctx)
    cancelled, message = await ctx.job_store.cancel(run_id)
    return {**_run_view(job.to_summary(), message=message), "cancelled": cancelled}


# ── Internal helpers ───────────────────────────────────────────────────


def _get_job(run_id: str, context: MCPToolContext | None) -> Any:
    ctx = context or get_default_context()
    job = ctx.job_store.get(run_id)
    if job is None:
        known = [item.job_id for item in ctx.job_store.list()]
        raise ToolNotFoundError.for_id("run", run_id, known)
    return job


def _run_view(record: Mapping[str, Any], *, message: str | None = None) -> dict[str, Any]:
    """Present a job-store record with ``run_id`` as the run identifier."""

    view = {"run_id": record.get("job_id"), **{key: value for key, value in record.items() if key != "job_id"}}
    if message:
        view["message"] = message
    return view


def _build_display_command(
    *,
    model: str,
    benchmarks: Sequence[str],
    suite_ids: Sequence[str],
    mode: str,
    output_dir: Path,
    plan_only: bool,
    resume: bool,
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
    """Build the ``wf`` command for one model x benchmark cell or a suite."""

    if len(benchmarks) == 1 and not suite_ids:
        command = list(
            build_model_benchmark_command(
                model_id=model,
                benchmark_id=benchmarks[0],
                output_dir=output_dir,
                mode=mode,
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
        if generated_artifact_dir is not None:
            # ``score`` scores existing artifacts; run-only options do not apply.
            return tuple(command)
    else:
        command = list(
            build_suite_command(
                output_dir=output_dir,
                mode=mode,
                model_ids=(model,),
                benchmark_ids=benchmarks,
                suite_ids=suite_ids,
                model_manifest_dir=context.model_manifest_dir,
                benchmark_manifest_dir=context.benchmark_manifest_dir,
                plan_only=plan_only,
            )
        )
        for flag, value in (
            ("--model-variant", model_variant),
            ("--requests-path", requests_path),
            ("--task-name", task_name),
            ("--generated-artifact-dir", generated_artifact_dir),
            ("--output-artifact", output_artifact),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        for metric in metrics:
            command.extend(["--metric", str(metric)])
        command.append("--json")
    if plan_only and "--plan-only" not in command:
        command.append("--plan-only")
    if resume:
        command.append("--resume")
    if generation_cache_dir is not None:
        command.extend(["--generation-cache-dir", str(generation_cache_dir)])
    if generation_cache_mode != "off":
        command.extend(["--generation-cache-mode", generation_cache_mode])
    return tuple(command)


def _benchmark_ids(benchmarks: Sequence[str] | None, suite_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Deduplicate benchmark ids; a suite alone is a valid selection."""

    selected = tuple(dict.fromkeys(str(item) for item in (benchmarks or ()) if str(item).strip()))
    if not selected and not suite_ids:
        raise ValueError("pass at least one benchmark id (benchmarks) or suite preset (suite_ids)")
    return selected


def _default_output_dir(root: Path, model: str, benchmarks: Sequence[str], *, run_id: str | None = None) -> Path:
    """Name the default output directory after the model and up to two benchmarks."""

    label = "__".join((_safe_path_label(model), *(_safe_path_label(item) for item in benchmarks[:2])))
    if len(benchmarks) > 2:
        label = f"{label}__plus_{len(benchmarks) - 2}"
    output_dir = root / label
    return output_dir if run_id is None else output_dir / _safe_path_label(run_id)


def _safe_path_label(value: str) -> str:
    """Sanitise a string for use as one path component."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._") or "run"


def _sample_file_candidates(run_root: Path, task_name: str | None) -> tuple[Path, ...]:
    """Candidate sample files inside a run directory; paths escaping ``run_root`` are skipped."""

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
    """Read one page of JSON-object lines from a samples file."""

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
    return {"samples": samples, "total": total, "offset": offset, "limit": limit, "source_path": source_path}


async def _wait_for_job(job_id: str, context: MCPToolContext, timeout_s: int | None) -> bool:
    """Poll until the job is terminal (True) or the timeout expires (False)."""

    elapsed = 0.0
    interval = 0.5
    while True:
        job = context.job_store.get(job_id)
        if job is None or job.status in TERMINAL_JOB_STATUSES:
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
