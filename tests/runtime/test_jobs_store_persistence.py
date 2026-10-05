"""Persistence and reconciliation tests for ``AsyncCommandJobStore`` (CM-28).

The store optionally writes a JSON metadata index (run_id/pid/output_dir/
status/log paths) so an MCP stdio server restarted by its client can still
report — and cancel — the evaluation subprocesses it left running. These
tests use real short-lived subprocesses and crafted index files with fake
pids; no GPU, network, or optional packages required.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from worldfoundry.core.execution.process import (
    process_group_alive,
    process_identity,
    terminate_owned_process_group,
    terminate_process_group,
)
from worldfoundry.runtime.jobs import (
    JOB_STORE_STATE_SCHEMA_VERSION,
    AsyncCommandJobStore,
    CommandJob,
)


def _run_to_completion(store: AsyncCommandJobStore, command: list[str], **kwargs):
    async def scenario():
        job = store.submit(command, **kwargs)
        assert job._task is not None
        await job._task
        return job

    return asyncio.run(scenario())


def _write_index(state_path: Path, jobs: list[dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"schema_version": JOB_STORE_STATE_SCHEMA_VERSION, "jobs": jobs}),
        encoding="utf-8",
    )


def _job_row(**overrides) -> dict:
    row = {
        "job_id": "job-1",
        "run_id": "run-1",
        "pid": None,
        "status": "running",
        "created_at": "2026-08-25T00:00:00+00:00",
        "started_at": "2026-08-25T00:00:01+00:00",
        "completed_at": None,
        "returncode": None,
        "error": None,
        "cwd": None,
        "output_dir": "/tmp/run-1",
        "log_dir": None,
        "event_log_path": None,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "command": ["echo", "hi"],
        "metadata": {"run_id": "run-1", "surface": "mcp"},
    }
    row.update(overrides)
    return row


def _dead_pid() -> int:
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait()
    return probe.pid


# ── Index round trip ─────────────────────────────────────────────────────


def test_completed_job_survives_store_restart(tmp_path):
    state_path = tmp_path / "jobs-index.json"
    first = AsyncCommandJobStore(state_path=state_path)
    job = _run_to_completion(
        first,
        [sys.executable, "-c", "print('done')"],
        output_dir=tmp_path / "out",
        metadata={"run_id": "run-persist"},
    )
    assert job.status == "completed"
    assert job.pid is not None
    assert state_path.is_file()

    second = AsyncCommandJobStore(state_path=state_path)
    restored = second.get(job.job_id)
    assert restored is not None
    assert restored.restored is True
    assert restored.status == "completed"
    assert restored.run_id == "run-persist"
    assert restored.returncode == 0
    assert restored.pid == job.pid
    assert restored.output_dir == str(tmp_path / "out")
    assert restored.stdout_log_path == job.stdout_log_path
    summary = restored.to_summary()
    assert summary["restored"] is True
    assert summary["metadata"]["run_id"] == "run-persist"


def test_legacy_store_without_state_path_writes_nothing(tmp_path):
    store = AsyncCommandJobStore()
    job = _run_to_completion(store, [sys.executable, "-c", "print('ok')"], output_dir=tmp_path)
    assert job.status == "completed"
    assert store.state_path is None
    assert not list(tmp_path.glob("jobs-index*"))


@pytest.mark.parametrize("late", [False, True])
def test_exit_receipt_is_reconciled_even_when_liveness_changes_between_reads(tmp_path, monkeypatch, late):
    store = AsyncCommandJobStore(state_path=tmp_path / "jobs.json")
    job = CommandJob(job_id="race", command=("unused",), display_command=("unused",), pid=99999999, restored=True,
                     completion_path=str(tmp_path / "receipt.json"), completion_token="race-token")
    job.status = "failed" if late else "running"
    store._jobs[job.job_id] = job

    def publish_receipt():
        Path(job.completion_path).write_text(json.dumps({
            "job_id": job.job_id, "pid": job.pid, "completion_token": job.completion_token,
            "status": "failed", "returncode": 7, "error": "command exited with code 7",
        }))

    if late:
        assert store.get(job.job_id).returncode is None
        publish_receipt()
    else:
        def exit_during_probe(_):
            publish_receipt()
            return False
        monkeypatch.setattr(store, "_verified_process", exit_during_probe)
    recovered = store.get(job.job_id)
    assert recovered.status == "failed"
    assert recovered.returncode == 7
    assert recovered.error == "command exited with code 7"


def test_pruning_terminal_jobs_also_removes_their_completion_receipts(tmp_path):
    store = AsyncCommandJobStore(state_path=tmp_path / "jobs.json")
    job = _run_to_completion(store, [sys.executable, "-c", "pass"])
    receipt = Path(job.completion_path)
    assert receipt.is_file()
    assert store.prune(max_jobs=0) == 1
    assert not receipt.exists()
    assert store.list() == []


def test_pruning_restored_metadata_cannot_delete_an_unrelated_file(tmp_path):
    import uuid

    store = AsyncCommandJobStore(state_path=tmp_path / "jobs.json")
    unrelated = tmp_path / "keep.json"
    unrelated.write_text("keep")
    job = CommandJob(job_id="forged", command=("unused",), display_command=("unused",), status="completed", returncode=0,
                     completion_path=str(unrelated), completion_token=uuid.uuid4().hex)
    store._jobs[job.job_id] = job
    assert store.prune(max_jobs=0) == 1
    assert unrelated.read_text() == "keep"


# ── Startup pid reconciliation ───────────────────────────────────────────


def test_restart_marks_dead_pid_running_job_failed(tmp_path):
    state_path = tmp_path / "jobs-index.json"
    dead = _dead_pid()
    _write_index(state_path, [_job_row(pid=dead, status="running")])

    store = AsyncCommandJobStore(state_path=state_path)
    job = store.get("job-1")
    assert job is not None
    assert job.status == "failed"
    assert str(dead) in (job.error or "")
    assert job.completed_at is not None
    # The reconciled status is written back so the next restart agrees.
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "failed"


def test_restart_keeps_live_pid_running_and_cancel_kills_it(tmp_path):
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    try:
        state_path = tmp_path / "jobs-index.json"
        _write_index(state_path, [_job_row(pid=orphan.pid, status="running",
                                          process_identity=process_identity(orphan.pid))])

        store = AsyncCommandJobStore(state_path=state_path)
        job = store.get("job-1")
        assert job is not None
        assert job.status == "running"
        assert job.restored is True

        success, message = asyncio.run(store.cancel("job-1"))
        assert success, message
        assert job.status == "cancelled"
        # In production the orphan's dead parent lets init reap it; here the
        # test process is still the parent, so reap via wait() to observe the
        # SIGTERM delivered through the restored-pid cancellation path.
        assert orphan.wait(timeout=10) == -signal.SIGTERM
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()


def test_restart_marks_queued_job_without_pid_failed(tmp_path):
    state_path = tmp_path / "jobs-index.json"
    _write_index(state_path, [_job_row(pid=None, status="queued", started_at=None)])
    store = AsyncCommandJobStore(state_path=state_path)
    job = store.get("job-1")
    assert job is not None
    assert job.status == "failed"
    assert "no pid recorded" in (job.error or "")


def test_corrupt_index_starts_empty_without_raising(tmp_path):
    state_path = tmp_path / "jobs-index.json"
    state_path.write_text("{not json", encoding="utf-8")
    store = AsyncCommandJobStore(state_path=state_path)
    assert store.list() == []


# ── Retention bound ──────────────────────────────────────────────────────


def test_max_jobs_bound_evicts_oldest_terminal_and_persists(tmp_path):
    state_path = tmp_path / "jobs-index.json"
    store = AsyncCommandJobStore(max_jobs=2, state_path=state_path)
    job_ids: list[str] = []
    for index in range(4):
        job = _run_to_completion(
            store,
            [sys.executable, "-c", f"print({index})"],
            job_id=f"job-{index}",
        )
        assert job.status == "completed"
        job_ids.append(job.job_id)
        # created_at has second granularity; keep eviction order deterministic.
        time.sleep(1.1)

    tracked = {job.job_id for job in store.list()}
    assert len(tracked) <= 3  # max_jobs + the in-flight submission
    assert "job-0" not in tracked
    assert "job-3" in tracked
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert {row["job_id"] for row in payload["jobs"]} == tracked


@pytest.mark.parametrize("access", ["get", "list"])
def test_restored_jobs_are_reconciled_after_the_process_exits(tmp_path, access):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"], start_new_session=True)
    state_path = tmp_path / "jobs.json"
    try:
        _write_index(state_path, [_job_row(pid=process.pid, process_identity=process_identity(process.pid))])
        store = AsyncCommandJobStore(state_path=state_path)
        assert store.get("job-1").status == "running"
        process.wait(timeout=5)
        job = store.get("job-1") if access == "get" else store.list()[0]
        assert job.status == "failed"
        assert job.returncode is None  # An unrecorded detached exit is unknown.
        assert "unknown" in job.error
    finally:
        terminate_process_group(process, grace_seconds=0)


@pytest.mark.parametrize("saved_identity", ["missing", "mismatch"])
def test_restored_unverified_pids_never_receive_signals(tmp_path, monkeypatch, saved_identity):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        identity = process_identity(process.pid) if saved_identity == "mismatch" else None
        if identity is not None:
            identity["start_ticks"] = "incorrect-incarnation"
        state_path = tmp_path / "jobs.json"
        _write_index(state_path, [_job_row(pid=process.pid, process_identity=identity)])
        signals = []
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", lambda *args: signals.append(args))
            store = AsyncCommandJobStore(state_path=state_path)
            success, _ = asyncio.run(store.cancel("job-1"))
            assert not success
            assert signals == []
            assert process.poll() is None
    finally:
        terminate_process_group(process, grace_seconds=0)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_a_restored_store_reads_the_independent_completion_receipt(tmp_path, exit_code):
    async def scenario():
        state_path = tmp_path / "jobs.json"
        owner = AsyncCommandJobStore(state_path=state_path)
        job = owner.submit([sys.executable, "-c", f"import time; time.sleep(0.2); raise SystemExit({exit_code})"])
        while job.pid is None:
            await asyncio.sleep(0.01)
        restored = AsyncCommandJobStore(state_path=state_path)
        assert restored.get(job.job_id).status == "running"
        await job._task
        recovered = restored.get(job.job_id)
        assert recovered.status == ("completed" if exit_code == 0 else "failed")
        assert recovered.returncode == exit_code
        assert recovered.completed_at is not None
        assert recovered.process_identity == job.process_identity

    asyncio.run(scenario())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process identity")
def test_completion_survives_the_controller_process_exiting(tmp_path):
    state_path = tmp_path / "jobs.json"
    script = (
        "import asyncio,os,sys; from worldfoundry.runtime.jobs import AsyncCommandJobStore;\n"
        "async def scenario():\n"
        " store=AsyncCommandJobStore(state_path=sys.argv[1]); "
        "job=store.submit([sys.executable,'-c','import time; time.sleep(0.4); raise SystemExit(7)'],job_id='detached')\n"
        " while job.pid is None: await asyncio.sleep(0.01)\n"
        " os._exit(0)\n"
        "asyncio.run(scenario())"
    )
    subprocess.run([sys.executable, "-c", script, str(state_path)], check=True, timeout=10)
    restored = AsyncCommandJobStore(state_path=state_path)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = restored.get("detached")
            # The receipt is published before the supervisor itself exits.
            # Both milestones must complete within the original deadline.
            if job.terminal and not process_group_alive(job.pid):
                break
            time.sleep(0.03)
        job = restored.get("detached")
        assert job.status == "failed"
        assert job.returncode == 7
        assert not process_group_alive(job.pid)
    finally:
        job = restored.get("detached")
        if (
            job.process_identity is not None
            and process_identity(job.pid) == job.process_identity
            and process_group_alive(job.pid)
        ):
            terminate_owned_process_group(job.pid, grace_seconds=0)


def test_supervised_timeout_is_reported_to_a_restored_store(tmp_path):
    async def scenario():
        state_path = tmp_path / "jobs.json"
        owner = AsyncCommandJobStore(state_path=state_path)
        job = owner.submit([sys.executable, "-c", "import time; time.sleep(120)"], timeout=0.15)
        while job.pid is None:
            await asyncio.sleep(0.01)
        restored = AsyncCommandJobStore(state_path=state_path)
        await job._task
        recovered = restored.get(job.job_id)
        assert recovered.status == "failed"
        assert "timed out" in recovered.error
        assert not process_group_alive(job.pid)

    asyncio.run(scenario())
