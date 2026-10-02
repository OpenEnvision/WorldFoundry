"""In-process job queue and lifecycle tracking for the WorldFoundry Studio UI.

Studio submits heavyweight inference/evaluation work as background jobs. This
module defines the job record (``StudioJob``), a thread-pool-backed store
(``StudioJobStore``), and Gradio-facing formatters for job tables and detail
panels.

The workspace app configures the worker count so users can run independent
inference jobs in parallel when enough GPUs are available.
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from html import escape
from threading import RLock
from time import monotonic
from typing import Any

from worldfoundry.core.observability.time import utc_now_iso
from worldfoundry.evaluation.reporting.run_manifest import redact_secret_text
from worldfoundry.runtime.jobs import TERMINAL_JOB_STATUSES

STUDIO_JOB_TABLE_HEADERS = ["Job ID", "Title", "Model", "Action", "Status", "Created", "Elapsed"]
DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0
MAX_SHUTDOWN_GRACE_SECONDS = 15.0


def _bounded_shutdown_grace_seconds(grace_seconds: float) -> float:
    """Validate and cap the time spent waiting for cooperative job shutdown."""

    try:
        normalized = float(grace_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("shutdown grace_seconds must be a finite non-negative number") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("shutdown grace_seconds must be a finite non-negative number")
    return min(normalized, MAX_SHUTDOWN_GRACE_SECONDS)


@dataclass
class StudioJob:
    """One Studio background job with status, logs, and optional result payload."""

    job_id: str
    title: str
    model_id: str
    display_name: str
    action: str
    job_type: str = "inference"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: Any | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    _future: Future[Any] | None = field(default=None, repr=False)
    _started_monotonic: float | None = field(default=None, repr=False)
    _completed_monotonic: float | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        """Return True when the job has reached a final status (completed/failed/cancelled)."""
        return self.status in TERMINAL_JOB_STATUSES

    def append_log(self, stream: str, text: str) -> None:
        """Append a timestamped log line from stdout/stderr/system."""
        if text:
            self.logs.append(
                {"time": utc_now_iso(), "stream": stream, "text": redact_secret_text(text)}
            )

    def log_text(self, *, limit: int | None = None) -> str:
        """Render recent log lines as plain text for the UI log panel."""
        rows = self.logs[-limit:] if limit is not None else self.logs
        return "".join(f"[{row.get('stream', 'log')}] {row.get('text', '')}" for row in rows)

    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since the job started (or 0 if not started)."""
        if self._started_monotonic is None:
            return 0.0
        end = self._completed_monotonic if self._completed_monotonic is not None else monotonic()
        return max(0.0, end - self._started_monotonic)


RunJobCallable = Callable[[StudioJob], Any]


def _result_indicates_failure(result: Any) -> tuple[bool, str | None]:
    """Inspect a run result and decide whether it should mark the job as failed.

    Runtime requirement plans are useful artifacts, but a ``blocked`` result
    must not be shown as a completed inference job in Studio.
    """
    if isinstance(result, Mapping):
        status = str(result.get("status") or "").strip().lower()
        detail = result.get("error") or result.get("blocked_reason")
    else:
        status = str(getattr(result, "status", "") or "").strip().lower()
        detail = None
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping):
            model_result = metadata.get("result")
            if isinstance(model_result, Mapping):
                detail = model_result.get("error") or model_result.get("blocked_reason")
    # Treat any non-success terminal status as a job failure for the UI.
    if status and status not in {
        "ok",
        "succeeded",
        "completed",
        "planned",
        "normalized",
        "official_verified",
        "official_results_imported",
        "official_results_normalized",
        "official_result_normalization",
        "official_results_normalization",
    }:
        suffix = f": {detail}" if detail else ""
        return True, f"run status: {status}{suffix}"
    return False, None


class StudioJobStore:
    """In-process job runner for the unified Studio UI.

    The store intentionally runs one job at a time by default so the UI cannot
    accidentally launch several heavyweight GPU pipelines in parallel.
    """

    def __init__(self, *, max_workers: int = 1, max_log_lines: int = 2000, initial_counter: int = 0) -> None:
        """Create a job store backed by a small thread pool (default: one worker)."""
        self.max_log_lines = max_log_lines
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="worldfoundry-studio-job")
        self._jobs: dict[str, StudioJob] = {}
        self._lock = RLock()
        self._counter = max(0, initial_counter)
        self._shutdown_started = False
        self._shutdown_complete = False

    def submit_run(
        self,
        *,
        title: str,
        model_id: str,
        display_name: str,
        action: str,
        job_type: str = "inference",
        metadata: Mapping[str, Any] | None,
        run_callable: RunJobCallable,
    ) -> StudioJob:
        """Enqueue a new job and execute ``run_callable(job)`` on the thread pool."""
        with self._lock:
            if self._shutdown_started:
                raise RuntimeError("StudioJobStore is shutting down and cannot accept new jobs")
            self._counter += 1
            job_id = f"studio-{self._counter:05d}"
            job = StudioJob(
                job_id=job_id,
                title=title or f"{display_name} {action}",
                model_id=model_id,
                display_name=display_name,
                action=action,
                job_type=job_type,
                metadata=dict(metadata or {}),
            )
            self._jobs[job.job_id] = job
            job._future = self._executor.submit(self._execute, job.job_id, run_callable)
            return job

    def shutdown(
        self,
        grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
        *,
        grace: float | None = None,
    ) -> None:
        """Stop accepting work and wait briefly for active jobs to cooperate.

        Queued futures are cancelled immediately. Running callables cannot be
        terminated safely from another thread, so they receive only the
        ``cancel_requested`` signal and are allowed at most ``grace_seconds``
        to return before this method releases the executor without waiting.
        """

        if grace is not None:
            if grace_seconds != DEFAULT_SHUTDOWN_GRACE_SECONDS:
                raise TypeError("pass either grace_seconds or the legacy grace alias, not both")
            grace_seconds = grace
        bounded_grace = _bounded_shutdown_grace_seconds(grace_seconds)
        pending_futures: list[Future[Any]] = []
        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            for job in self._jobs.values():
                if job.terminal:
                    continue
                job.cancel_requested = True
                future = job._future
                if future is not None and future.cancel():
                    job.status = "cancelled"
                    job.error = "cancelled before start"
                    job.completed_at = utc_now_iso()
                    job._completed_monotonic = monotonic()
                    job.append_log("system", "cancelled before start during Studio shutdown\n")
                    continue
                if future is not None and not future.done():
                    pending_futures.append(future)
                job.append_log(
                    "system",
                    "Studio shutdown requested; active inference must stop cooperatively\n",
                )

        try:
            if pending_futures and bounded_grace > 0:
                wait(pending_futures, timeout=bounded_grace)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            with self._lock:
                self._shutdown_complete = True

    def get(self, job_id: str | None) -> StudioJob | None:
        """Return a job by id, or None when id is empty or unknown."""
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[StudioJob]:
        """Return all jobs sorted by creation time (newest first)."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def delete(self, job_id: str | None) -> bool:
        """Remove a finished job from the in-memory store.

        Running or queued jobs cannot be deleted; cancel them first.
        """

        if not job_id:
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if not job.terminal:
                raise ValueError(f"cannot delete a {job.status} job")
            del self._jobs[job_id]
            return True

    def cancel(self, job_id: str | None) -> tuple[bool, str]:
        """Request cancellation; queued jobs cancel immediately, running jobs stop cooperatively."""
        job = self.get(job_id)
        if job is None:
            return False, "select a job first"
        with self._lock:
            if job.terminal:
                return False, f"job is already {job.status}"
            job.cancel_requested = True
            future = job._future
            if future is not None and future.cancel():
                job.status = "cancelled"
                job.error = "cancelled before start"
                job.completed_at = utc_now_iso()
                job._completed_monotonic = monotonic()
                job.append_log("system", "cancelled before start\n")
                return True, "cancelled"
            job.append_log("system", "cancellation requested; active inference will stop after the current call returns\n")
            return True, "cancellation requested"

    def _execute(self, job_id: str, run_callable: RunJobCallable) -> None:
        """Worker entrypoint: transition job state, invoke callable, capture errors/logs."""
        job = self.get(job_id)
        if job is None:
            return
        with self._lock:
            if job.cancel_requested:
                job.status = "cancelled"
                job.error = "cancelled before start"
                job.completed_at = utc_now_iso()
                job._completed_monotonic = monotonic()
                return
            job.status = "running"
            job.started_at = utc_now_iso()
            job._started_monotonic = monotonic()
            job.append_log("system", f"started {job.title}\n")

        try:
            result = run_callable(job)
            with self._lock:
                job.result = result
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.error = "cancelled after current inference completed"
                else:
                    failed_result, error = _result_indicates_failure(result)
                    if failed_result:
                        job.status = "failed"
                        job.error = error
                        job.append_log("system", f"{error}\n")
                    else:
                        job.status = "completed"
                        job.append_log("system", "completed\n")
        except Exception as exc:  # noqa: BLE001 - surfaced through local UI.
            with self._lock:
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.error = f"cancelled: {type(exc).__name__}: {exc}"
                    job.append_log("system", "cancelled active job\n")
                else:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.append_log("stderr", traceback.format_exc())
        finally:
            with self._lock:
                if len(job.logs) > self.max_log_lines:
                    del job.logs[: len(job.logs) - self.max_log_lines]
                job.completed_at = job.completed_at or utc_now_iso()
                job._completed_monotonic = monotonic()


def format_elapsed(job: StudioJob) -> str:
    """Format job elapsed time as a compact human-readable string."""
    seconds = int(job.elapsed_seconds())
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def format_job_choices(jobs: Sequence[StudioJob]) -> list[tuple[str, str]]:
    """Build Gradio dropdown choices mapping label -> job_id."""
    return [
        (f"{job.title} · {job.status} · {job.job_id}", job.job_id)
        for job in jobs
    ]


def format_jobs_table(jobs: Sequence[StudioJob]) -> list[list[str]]:
    """Build row data for the Studio jobs table widget."""
    rows: list[list[str]] = []
    for job in jobs:
        rows.append(
            [
                job.job_id,
                job.title,
                job.display_name,
                job.action,
                job.status,
                job.created_at,
                format_elapsed(job),
            ]
        )
    return rows


def job_detail_html(job: StudioJob | None) -> str:
    """Render HTML summary panel for one job (model, prompt, output, error)."""
    if job is None:
        return '<div class="wa-job-detail wa-job-empty">No Studio jobs yet.</div>'
    metadata = dict(job.metadata)
    prompt = str(metadata.get("prompt") or "").strip()
    prompt_html = f"<p><strong>Prompt</strong><br>{escape(prompt)}</p>" if prompt else ""
    output_dir = ""
    result = job.result
    if result is not None:
        output_dir = str(getattr(result, "output_dir", "") or "")
    if not output_dir:
        output_dir = str(metadata.get("output_root") or "")
    output_html = f"<p><strong>Output</strong><br><code>{escape(output_dir)}</code></p>" if output_dir else ""
    error_html = f"<p class=\"wa-job-error\"><strong>Error</strong><br>{escape(job.error or '')}</p>" if job.error else ""
    return f"""
<div class="wa-job-detail wa-job-status-{escape(job.status)}">
  <div class="wa-job-detail-head">
    <strong>{escape(job.title)}</strong>
    <span>{escape(job.status)}</span>
  </div>
  <p><strong>Model</strong><br>{escape(job.display_name)} <code>{escape(job.model_id)}</code></p>
  <p><strong>Action</strong><br>{escape(job.action)} · {escape(job.job_type)}</p>
  {prompt_html}
  {output_html}
  {error_html}
</div>
"""


def job_log_text(job: StudioJob | None, *, limit: int = 120) -> str:
    """Return truncated plain-text logs for the selected job."""
    if job is None:
        return ""
    return job.log_text(limit=limit)


__all__ = [
    "STUDIO_JOB_TABLE_HEADERS",
    "StudioJob",
    "StudioJobStore",
    "format_job_choices",
    "format_jobs_table",
    "job_detail_html",
    "job_log_text",
]
