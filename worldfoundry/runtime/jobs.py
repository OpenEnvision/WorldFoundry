"""Local process/job helpers for CLI surfaces and MCP/UI execution."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.core.execution.process import (
    process_group_alive,
    process_identity,
    terminate_owned_process_group,
)
from worldfoundry.core.observability.time import utc_now_iso as _utc_now_iso

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})

# asyncio streams default to a 64 KiB readline buffer; model CLIs routinely
# print JSON results larger than that, which used to raise ValueError, mark the
# job failed, and orphan the child process group.
_STREAM_BUFFER_LIMIT = 2**20

# Version tag for the persisted job-index file written by
# ``AsyncCommandJobStore(state_path=...)``.
JOB_STORE_STATE_SCHEMA_VERSION = 2

# Metadata fields persisted per job in the on-disk index. In-memory log tails
# are intentionally excluded: raw stdout/stderr/events already live in the
# per-job artifact files referenced by the ``*_log_path`` fields.
_PERSISTED_JOB_FIELDS = (
    "job_id",
    "run_id",
    "pid",
    "process_identity",
    "completion_path",
    "completion_token",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "returncode",
    "error",
    "cwd",
    "output_dir",
    "log_dir",
    "event_log_path",
    "stdout_log_path",
    "stderr_log_path",
)


def pid_alive(pid: int | None) -> bool:
    """Best-effort liveness probe for a process id (POSIX signal 0)."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another user.
        return True
    except OSError:
        return False
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return fields[0] not in {"Z", "X"}
        except (OSError, IndexError):
            return False
    return True


def _iso_to_epoch(value: str | None) -> float | None:
    """Convert an ISO-8601 timestamp (with optional ``Z`` suffix) to epoch seconds."""
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def python_module_command(command: Sequence[str], *, python_executable: str | None = None) -> tuple[str, ...]:
    """Run a ``worldfoundry-eval`` command through the current Python interpreter."""

    items = tuple(str(item) for item in command)
    if not items:
        raise ValueError("command cannot be empty")
    if items[0] in {"worldfoundry", "worldfoundry-eval"}:
        return (python_executable or sys.executable, "-m", "worldfoundry.evaluation", *items[1:])
    return items


def _decode_process_text(value: str | bytes | None) -> str:
    """Decode subprocess output to a string, replacing invalid bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Force-kill the entire process group for a subprocess on POSIX systems."""
    if sys.platform == "win32":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int,
    kill_timeout: int = 5,
) -> dict[str, Any]:
    """Run a command with a hard timeout and always return captured output.

    This helper is intended for official benchmark subprocesses. Some simulator
    or CUDA-backed scripts can ignore ordinary timeout handling while stuck in
    native code, so timeout failures are converted into structured results that
    callers can write into scorecards instead of surfacing a traceback.
    """

    command_tuple = tuple(str(item) for item in command)
    if not command_tuple:
        raise ValueError("command cannot be empty")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    from worldfoundry.core.observability.logging_setup import log_context_environment

    process_env.update(log_context_environment())

    start = time.monotonic()
    process = subprocess.Popen(
        command_tuple,
        cwd=None if cwd is None else str(cwd),
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=sys.platform != "win32",
    )
    timed_out = False
    kill_stuck = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Native CUDA simulators can ignore regular interrupt signals, so we
        # terminate the whole process group first and then force-kill if needed.
        timed_out = True
        _kill_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=kill_timeout)
        except subprocess.TimeoutExpired as kill_exc:
            kill_stuck = True
            stdout = _decode_process_text(kill_exc.stdout or exc.stdout)
            stderr = _decode_process_text(kill_exc.stderr or exc.stderr)
        stderr = (f"{_decode_process_text(stderr)}\nTimeoutExpired: command exceeded {timeout}s").strip()
    return {
        "command": list(command_tuple),
        "stdout": _decode_process_text(stdout),
        "stderr": _decode_process_text(stderr),
        "returncode": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "kill_stuck": kill_stuck,
        "duration_seconds": time.monotonic() - start,
    }


@dataclass
class CommandJob:
    """Track the lifecycle and output of an asynchronous subprocess command.

    Attributes:
        job_id: Unique identifier for the job.
        command: Full command tuple executed by the subprocess.
        display_command: Human-readable command tuple for UI surfaces.
        cwd: Working directory for the subprocess, if set.
        output_dir: Directory for persistent output artifacts, if set.
        metadata: Arbitrary metadata attached by the submitter.
        status: Current lifecycle status (queued, running, completed, failed, cancelled).
        created_at: ISO timestamp when the job was created.
        started_at: ISO timestamp when the job began running.
        completed_at: ISO timestamp when the job finished.
        returncode: Process exit code, or ``None`` if not finished.
        error: Error message if the job failed.
        result: Parsed JSON result extracted from stdout.
        logs: Chronological list of stdout/stderr log entries.
    """

    job_id: str
    command: tuple[str, ...]
    display_command: tuple[str, ...]
    run_id: str = ""
    pid: int | None = None
    process_identity: dict[str, str | int] | None = None
    completion_path: str | None = None
    completion_token: str | None = None
    cwd: str | None = None
    output_dir: str | None = None
    log_dir: str | None = None
    event_log_path: str | None = None
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    result: Any | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    # True when the job was rebuilt from a persisted index after a store
    # restart; such jobs carry no live process/task handles.
    restored: bool = False
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        """Return whether the job has reached a terminal status."""
        return self.status in TERMINAL_JOB_STATUSES

    def append_log(self, stream: str, text: str) -> None:
        """Append a timestamped log entry for *stream* (stdout or stderr)."""
        if text:
            self.logs.append({"time": _utc_now_iso(), "stream": stream, "text": text})

    def log_text(self, *, stream: str | None = None, limit: int | None = None) -> str:
        """Return concatenated text from log entries, optionally filtered by stream."""
        rows = [row for row in self.logs if stream is None or row.get("stream") == stream]
        if limit is not None:
            rows = rows[-limit:]
        return "".join(str(row.get("text") or "") for row in rows)

    def to_summary(self, *, log_tail: int = 40) -> dict[str, Any]:
        """Return a summary dict suitable for UI polling endpoints."""
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "pid": self.pid,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "output_dir": self.output_dir,
            "log_dir": self.log_dir,
            "event_log_path": self.event_log_path,
            "stdout_log_path": self.stdout_log_path,
            "stderr_log_path": self.stderr_log_path,
            "command": list(self.display_command),
            "cwd": self.cwd,
            "metadata": dict(self.metadata),
            "error": self.error,
            "restored": self.restored,
            "logs": self.logs[-log_tail:] if log_tail else [],
        }

    def to_result(self, *, include_logs: bool = False, log_tail: int = 200) -> dict[str, Any]:
        """Return a full result dict with optional log text."""
        payload = self.to_summary(log_tail=log_tail if include_logs else 0)
        payload["result"] = self.result
        if include_logs:
            payload["stdout"] = self.log_text(stream="stdout", limit=log_tail)
            payload["stderr"] = self.log_text(stream="stderr", limit=log_tail)
        return payload


class AsyncCommandJobStore:
    """Small in-process command runner used by local UI and MCP surfaces."""

    def __init__(
        self,
        *,
        max_log_lines: int = 4000,
        max_jobs: int | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.max_log_lines = max_log_lines
        # Optional retention bound for terminal jobs; ``None`` keeps every job
        # (legacy behaviour). Long-lived UI/MCP processes should set a bound.
        self.max_jobs = max_jobs
        # Optional on-disk JSON index (CM-28). When set, job metadata
        # (run_id/pid/output_dir/status/log paths) survives store restarts:
        # jobs are restored on construction and reconciled against live pids,
        # so an MCP stdio server killed by its client can still report — and
        # cancel — the evaluation subprocesses it left running.
        self.state_path = None if state_path is None else Path(state_path)
        self._jobs: dict[str, CommandJob] = {}
        if self.state_path is not None:
            self._restore_persisted_jobs()

    # ── Persistence (CM-28) ─────────────────────────────────────────

    def _restore_persisted_jobs(self) -> None:
        """Load the persisted job index and reconcile statuses against live pids.

        Non-terminal jobs whose recorded pid is gone are marked failed; jobs
        whose pid is still alive stay ``running`` as *restored* jobs (metadata
        and on-disk log paths available, no live process handle).  A missing
        or unreadable index starts the store empty.
        """
        assert self.state_path is not None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return
        rows = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            return
        reconciled = False
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("job_id"):
                continue
            command = tuple(str(item) for item in (row.get("command") or ()))
            job = CommandJob(
                job_id=str(row["job_id"]),
                command=command,
                display_command=command,
                restored=True,
            )
            for field_name in _PERSISTED_JOB_FIELDS:
                if field_name in ("job_id",) or row.get(field_name) is None:
                    continue
                setattr(job, field_name, row[field_name])
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping):
                job.metadata = dict(metadata)
            reconciled = self._reconcile_restored_job(job) or reconciled
            self._jobs[job.job_id] = job
        if reconciled:
            self._persist()

    @staticmethod
    def _verified_process(job: CommandJob) -> bool:
        """A PID alone is not authority to signal a recovered process group."""
        identity = job.process_identity
        return bool(
            isinstance(job.pid, int)
            and job.pid > 1
            and isinstance(identity, dict)
            and identity.get("pgid") == job.pid
            and identity.get("session_id") == job.pid
            and process_identity(job.pid) == identity
        )

    @staticmethod
    def _read_completion(job: CommandJob) -> dict[str, Any] | None:
        if not job.completion_path or not job.completion_token:
            return None
        try:
            payload = json.loads(Path(job.completion_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or (
            payload.get("job_id") != job.job_id
            or payload.get("completion_token") != job.completion_token
            or payload.get("pid") != job.pid
            or (job.process_identity is not None and payload.get("process_identity") != job.process_identity)
            or payload.get("status") not in TERMINAL_JOB_STATUSES
            or (payload.get("returncode") is not None and type(payload["returncode"]) is not int)
        ):
            return None
        return payload

    def _reconcile_restored_job(self, job: CommandJob) -> bool:
        if not job.restored:
            return False
        # An unknown failure can precede a supervisor receipt becoming visible.
        # A matching receipt remains authoritative on subsequent polls.
        if job.terminal and not (job.status == "failed" and job.returncode is None and job.completion_token):
            return False
        receipt = self._read_completion(job)
        if receipt is None and not job.terminal:
            if self._verified_process(job) and process_group_alive(job.pid):
                changed = job.status != "running"
                job.status = "running"
                return changed
            # The supervisor can publish its receipt and exit between the
            # first read and the liveness probe. Read again before declaring
            # the exit status unknown.
            receipt = self._read_completion(job)
        if receipt is not None:
            job.status = receipt["status"]
            job.returncode = receipt.get("returncode")
            job.error = receipt.get("error")
            job.completed_at = receipt.get("completed_at") or _utc_now_iso()
            return True
        if job.terminal:
            return False
        job.status = "failed"
        job.returncode = None
        if job.pid is None:
            job.error = "job lost during job-store restart (no pid recorded)"
        elif not pid_alive(job.pid):
            job.error = f"process not found after job-store restart (pid {job.pid}); exit status unknown"
        else:
            job.error = f"recorded process identity cannot be verified (pid {job.pid}); exit status unknown"
        job.completed_at = _utc_now_iso()
        return True

    def _persist(self) -> None:
        """Atomically write the metadata index for every tracked job."""
        if self.state_path is None:
            return
        rows = []
        for job in sorted(self._jobs.values(), key=lambda item: item.created_at):
            row: dict[str, Any] = {name: getattr(job, name) for name in _PERSISTED_JOB_FIELDS}
            row["command"] = list(job.display_command)
            row["metadata"] = dict(job.metadata)
            rows.append(row)
        payload = {"schema_version": JOB_STORE_STATE_SCHEMA_VERSION, "jobs": rows}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_name(self.state_path.name + f".tmp-{os.getpid()}")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp_path, self.state_path)
        except OSError:
            # Persistence is best-effort: an unwritable index must never take
            # down job submission or status transitions.
            pass

    def submit(
        self,
        command: Sequence[str],
        *,
        display_command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        output_dir: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> CommandJob:
        """Submit a command as an async job and start it immediately.

        Args:
            command: Command and arguments to execute.
            display_command: Optional human-readable override for UI surfaces.
            cwd: Working directory for the subprocess.
            env: Additional environment variables merged into ``os.environ``.
            output_dir: Directory for persistent output artifacts.
            metadata: Arbitrary metadata attached to the job.
            job_id: Optional explicit job identifier; auto-generated if ``None``.
            timeout: Optional per-job wall-clock bound in seconds. On expiry the
                subprocess group is terminated and the job is marked failed.

        Returns:
            The newly created :class:`CommandJob`.

        Raises:
            ValueError: If *job_id* already exists in the store.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if self.max_jobs is not None:
            self.prune(max_jobs=self.max_jobs)
        resolved_job_id = job_id or uuid.uuid4().hex[:12]
        if resolved_job_id in self._jobs:
            raise ValueError(f"job already exists: {resolved_job_id}")
        # Jobs are tracked in-process so UI/MCP calls can poll by id and inspect
        # both state transitions and captured stdout/stderr.
        job_metadata = dict(metadata or {})
        run_id = str(job_metadata.get("run_id") or resolved_job_id)
        log_dir: Path | None = None
        if output_dir is not None:
            output_root = Path(output_dir)
            safe_job_id = "".join(
                char if char.isalnum() or char in {"-", "_", "."} else "-" for char in resolved_job_id
            )
            log_dir = output_root / "logs" / "jobs" / (safe_job_id.strip(".-") or "job")
        job = CommandJob(
            job_id=resolved_job_id,
            run_id=run_id,
            command=tuple(str(item) for item in command),
            display_command=tuple(str(item) for item in (display_command or command)),
            cwd=str(cwd) if cwd is not None else None,
            output_dir=str(output_dir) if output_dir is not None else None,
            log_dir=None if log_dir is None else str(log_dir),
            event_log_path=None if log_dir is None else str(log_dir / "events.jsonl"),
            stdout_log_path=None if log_dir is None else str(log_dir / "stdout.log"),
            stderr_log_path=None if log_dir is None else str(log_dir / "stderr.log"),
            metadata=job_metadata,
        )
        self._jobs[job.job_id] = job
        self._write_lifecycle_event(job, "INFO", "job.queued", "WorldFoundry job queued")
        self._persist()
        job._task = asyncio.create_task(self._run(job, dict(env or {}), timeout=timeout))
        return job

    def get(self, job_id: str) -> CommandJob | None:
        """Retrieve a job by its identifier, or ``None`` if not found."""
        job = self._jobs.get(job_id)
        if job is not None and self._reconcile_restored_job(job):
            self._persist()
        return job

    def list(self) -> list[CommandJob]:
        """Return all jobs sorted by creation time (most recent first)."""
        changed = False
        for job in self._jobs.values():
            changed = self._reconcile_restored_job(job) or changed
        if changed:
            self._persist()
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def _remove_completion_receipt(self, job: CommandJob) -> None:
        if self.state_path is None or not job.completion_path or not job.completion_token:
            return
        try:
            # Restored metadata must never let retention unlink an arbitrary file.
            token = uuid.UUID(job.completion_token).hex
            if token != job.completion_token:
                return
            directory = self.state_path.absolute().with_name(self.state_path.name + ".completions")
            expected = directory / f"{token}.json"
            if Path(job.completion_path).absolute() != expected:
                return
            expected.unlink(missing_ok=True)
        except (ValueError, OSError):
            return

    def prune(self, *, max_age_seconds: float | None = None, max_jobs: int | None = None) -> int:
        """Evict terminal jobs to bound memory in long-lived UI/MCP processes.

        Running or queued jobs are never evicted. ``max_age_seconds`` drops
        terminal jobs whose completion (or creation) timestamp is older than
        the cutoff; ``max_jobs`` keeps at most that many jobs by evicting the
        oldest terminal ones first.

        Returns:
            The number of jobs removed.
        """
        removed = 0
        self.list()
        if max_age_seconds is not None:
            cutoff = time.time() - max_age_seconds
            for job_id, job in list(self._jobs.items()):
                if not job.terminal or (job._task is not None and not job._task.done()):
                    continue
                stamp = _iso_to_epoch(job.completed_at or job.created_at)
                if stamp is not None and stamp < cutoff:
                    del self._jobs[job_id]
                    self._remove_completion_receipt(job)
                    removed += 1
        if max_jobs is not None and len(self._jobs) > max_jobs:
            excess = len(self._jobs) - max_jobs
            for job in sorted(self._jobs.values(), key=lambda item: item.created_at):
                if excess <= 0:
                    break
                if not job.terminal or (job._task is not None and not job._task.done()):
                    continue
                del self._jobs[job.job_id]
                self._remove_completion_receipt(job)
                removed += 1
                excess -= 1
        if removed:
            self._persist()
        return removed

    async def cancel(self, job_id: str) -> tuple[bool, str]:
        """Cancel a running or queued job by identifier.

        Args:
            job_id: The job to cancel.

        Returns:
            ``(True, "cancelled")`` on success, or ``(False, reason)`` on failure.
        """
        job = self.get(job_id)
        if job is None:
            return False, f"unknown job: {job_id}"
        if job.terminal:
            return False, f"job is already {job.status}"
        if job.restored and not self._verified_process(job):
            return False, "recorded process identity cannot be verified; refusing to signal a saved PID"
        job.status = "cancelled"
        job.error = "cancelled by request"
        if job.started_at is None:
            self._write_lifecycle_event(job, "WARNING", "job.cancelled", "WorldFoundry job cancelled")
        await self._terminate_process(job)
        if job._task is not None and not job._task.done():
            job._task.cancel()
            await asyncio.gather(job._task, return_exceptions=True)
        job.completed_at = job.completed_at or _utc_now_iso()
        self._persist()
        return True, "cancelled"

    async def _run(self, job: CommandJob, env: Mapping[str, str], *, timeout: float | None = None) -> None:
        """Execute the job subprocess, stream logs, and set the final status."""
        # Async runner lifecycle: spawn process -> stream logs -> parse result -> set final status.
        if job.status == "cancelled":
            return
        job.status = "running"
        job.started_at = _utc_now_iso()
        process_env = os.environ.copy()
        process_env.update(env)
        try:
            if job.output_dir:
                Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            if job.log_dir:
                Path(job.log_dir).mkdir(parents=True, exist_ok=True)
            from worldfoundry.core.observability.logging_setup import log_context_environment

            process_env.update(log_context_environment(run_id=job.run_id, job_id=job.job_id, phase="job"))
            if job.event_log_path:
                # Give framework-owned child CLIs a per-job sink.  Their
                # stdout/stderr remain in the separate raw artifact files.
                process_env["WORLDFOUNDRY_LOG_FILE"] = job.event_log_path
                process_env["WORLDFOUNDRY_LOG_JSON"] = "1"
            self._write_lifecycle_event(job, "INFO", "job.started", "WorldFoundry job started")
            command = job.command
            if self.state_path is not None:
                completion_dir = self.state_path.absolute().with_name(self.state_path.name + ".completions")
                completion_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                job.completion_token = uuid.uuid4().hex
                job.completion_path = str(completion_dir / f"{job.completion_token}.json")
                command = (
                    sys.executable,
                    str(Path(__file__).with_name("job_worker.py")),
                    "--completion-path",
                    job.completion_path,
                    "--completion-token",
                    job.completion_token,
                    "--job-id",
                    job.job_id,
                    *(("--timeout", str(timeout)) if timeout is not None else ()),
                    "--",
                    *job.command,
                )
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    cwd=job.cwd,
                    env=process_env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    limit=_STREAM_BUFFER_LIMIT,
                )
            )
            try:
                job._process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                # Cancellation during spawn must still capture the child
                # handle, otherwise fork/exec can finish with no owner left.
                job._process = await spawn
                job.pid = job._process.pid
                job.process_identity = process_identity(job.pid)
                raise
            job.pid = job._process.pid
            job.process_identity = process_identity(job.pid)
            self._persist()

            async def _pump_and_wait() -> None:
                readers = [
                    asyncio.create_task(self._read_stream(job, "stdout", job._process.stdout)),
                    asyncio.create_task(self._read_stream(job, "stderr", job._process.stderr)),
                ]
                try:
                    # Process.wait() can wait for inherited pipes to close even
                    # after its leader exits. Stop remaining group members first.
                    while job._process.returncode is None:
                        for reader in readers:
                            if reader.done() and not reader.cancelled() and reader.exception() is not None:
                                reader.result()
                        await asyncio.sleep(0.03)
                    await self._terminate_process(job)
                    await asyncio.gather(*readers)
                    job.returncode = await job._process.wait()
                finally:
                    for reader in readers:
                        if not reader.done():
                            reader.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)

            if timeout is None:
                await _pump_and_wait()
            else:
                try:
                    # A durable supervisor enforces timeout independently of
                    # this controller. Allow its bounded teardown/receipt flush.
                    budget = timeout + 4 if job.completion_path else timeout
                    await asyncio.wait_for(_pump_and_wait(), timeout=budget)
                except asyncio.TimeoutError:
                    await self._terminate_process(job)
                    job.returncode = job._process.returncode
                    if job.status != "cancelled":
                        job.status = "failed"
                        job.error = f"job timed out after {timeout}s"
                        self._append_job_log(job, "stderr", f"TimeoutError: job exceeded {timeout}s\n")
                        self._write_lifecycle_event(job, "ERROR", "job.timeout", "WorldFoundry job timed out")
                    return
            if job.status == "cancelled":
                return
            job.result = _extract_json_from_logs(job.logs)
            receipt = self._read_completion(job)
            if receipt is not None:
                job.returncode = receipt.get("returncode")
                job.status = receipt["status"]
                job.error = receipt.get("error")
            else:
                job.status = "completed" if job.returncode == 0 else "failed"
            if job.status == "failed":
                job.error = job.error or f"command exited with code {job.returncode}"
                self._write_lifecycle_event(job, "ERROR", "job.failed", "WorldFoundry job failed")
            else:
                self._write_lifecycle_event(job, "INFO", "job.finished", "WorldFoundry job finished")
        except asyncio.CancelledError:
            await self._terminate_process(job)
            job.status = "cancelled"
            job.error = job.error or "cancelled"
            self._write_lifecycle_event(job, "WARNING", "job.cancelled", "WorldFoundry job cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through UI/MCP status.
            # Never leave the spawned process group running when the runner
            # itself fails, otherwise the child keeps occupying GPUs while the
            # job is reported as failed.
            try:
                await self._terminate_process(job)
            except Exception:  # noqa: BLE001 - cleanup must not mask the original error.
                pass
            job.status = "failed"
            job.error = str(exc)
            self._append_job_log(job, "stderr", f"{type(exc).__name__}: {exc}\n")
            self._write_lifecycle_event(job, "ERROR", "job.failed", "WorldFoundry job failed", exception=exc)
        finally:
            if job.completed_at is None:
                job.completed_at = _utc_now_iso()
            self._persist()

    async def _read_stream(
        self,
        job: CommandJob,
        stream_name: str,
        stream: asyncio.StreamReader | None,
    ) -> None:
        """Read lines from a subprocess stream and append them to the job log."""
        if stream is None:
            return
        while True:
            try:
                chunk = await stream.readline()
            except ValueError:
                # A single line exceeded the stream buffer limit; the reader
                # already dropped the oversized data. Record a marker and keep
                # draining instead of failing the job (and orphaning the child).
                self._append_job_log(
                    job,
                    stream_name,
                    f"[worldfoundry] dropped output line exceeding {_STREAM_BUFFER_LIMIT} bytes\n",
                )
                continue
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            self._append_job_log(job, stream_name, text)
            if len(job.logs) > self.max_log_lines:
                del job.logs[: len(job.logs) - self.max_log_lines]

    @staticmethod
    def _append_job_log(job: CommandJob, stream_name: str, text: str) -> None:
        """Keep a bounded in-memory tail while retaining full raw stream files."""

        job.append_log(stream_name, text)
        target = job.stdout_log_path if stream_name == "stdout" else job.stderr_log_path
        if target is not None and text:
            with Path(target).open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(text)

    @staticmethod
    def _write_lifecycle_event(
        job: CommandJob,
        level: str,
        event: str,
        message: str,
        *,
        exception: BaseException | None = None,
    ) -> None:
        """Persist job state transitions independently of child process output."""

        if job.event_log_path is None:
            return
        from worldfoundry.core.observability.logging_setup import write_jsonl_event

        fields: dict[str, Any] = {
            "run_id": job.run_id,
            "job_id": job.job_id,
            "status": job.status,
            "returncode": job.returncode,
            "output_dir": job.output_dir,
        }
        if job.started_at and job.completed_at:
            fields["started_at"] = job.started_at
            fields["completed_at"] = job.completed_at
        write_jsonl_event(
            job.event_log_path,
            level=level,
            event=event,
            message=message,
            logger_name=__name__,
            exception=exception,
            **fields,
        )

    async def _terminate_process(self, job: CommandJob) -> None:
        """Stop the owned group, including descendants after leader exit."""
        process = job._process
        if process is None:
            if not job.restored or not self._verified_process(job):
                return

            def stop_restored() -> None:
                # Recheck in the thread immediately before signalling. Never
                # manufacture ownership by sampling an unverified saved PID.
                if self._verified_process(job):
                    terminate_owned_process_group(job.pid, grace_seconds=5)

            await asyncio.to_thread(stop_restored)
            return
        if sys.platform != "win32":
            await asyncio.to_thread(terminate_owned_process_group, process.pid, grace_seconds=5)
        elif process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (TimeoutError, asyncio.TimeoutError):
                process.kill()
        await process.wait()


def _extract_json_from_logs(logs: Sequence[Mapping[str, Any]]) -> Any | None:
    """Attempt to parse a JSON result object from stdout log entries."""
    stdout = "".join(str(row.get("text") or "") for row in logs if row.get("stream") == "stdout")
    for candidate in _json_candidates(stdout):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str) -> list[str]:
    """Generate candidate JSON strings from raw stdout for result extraction."""
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates.extend(reversed(lines))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    return candidates


__all__ = [
    "AsyncCommandJobStore",
    "CommandJob",
    "JOB_STORE_STATE_SCHEMA_VERSION",
    "TERMINAL_JOB_STATUSES",
    "pid_alive",
    "python_module_command",
    "run_bounded_command",
]
