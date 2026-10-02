from __future__ import annotations

import codecs
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.runtime.conda import RuntimeCondaEnvSpec
from worldfoundry.studio.inference import dispatch as conda_dispatch


class _FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.stdin = SimpleNamespace(write=lambda data: len(data), flush=lambda: None, close=lambda: None)
        self.stdout = SimpleNamespace(close=lambda: None)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def _worker(key: tuple[str, ...], *, last_used_at: float, in_use: int = 0) -> conda_dispatch._ResidentWorker:
    return conda_dispatch._ResidentWorker(
        key=key,
        base_key=key,
        model_id=key[0],
        process=_FakeProcess(),  # type: ignore[arg-type]
        lock=threading.RLock(),
        decoder=codecs.getincrementaldecoder("utf-8")("replace"),
        command=["python"],
        created_at=last_used_at,
        last_used_at=last_used_at,
        in_use=in_use,
    )


def _context(key: tuple[str, ...]) -> conda_dispatch._ResidentRunContext:
    return conda_dispatch._ResidentRunContext(
        child_run_kwargs={},
        payload_run_kwargs={},
        env={},
        key=key,
    )


@pytest.fixture(autouse=True)
def _isolated_resident_worker_registry(monkeypatch):
    real_stop = conda_dispatch.terminate_process_group

    def stop(process, **kwargs):
        if isinstance(process, _FakeProcess):
            process.returncode = -15
        else:
            real_stop(process, **kwargs)

    monkeypatch.setattr(conda_dispatch, "terminate_process_group", stop)
    monkeypatch.setattr(conda_dispatch, "_ensure_resident_worker_reaper", lambda: None)
    with conda_dispatch._RESIDENT_WORKERS_LOCK:
        conda_dispatch._RESIDENT_WORKERS.clear()
    yield
    with conda_dispatch._RESIDENT_WORKERS_LOCK:
        conda_dispatch._RESIDENT_WORKERS.clear()


def test_resident_worker_request_timeout_defaults_to_six_hours_and_zero_disables(monkeypatch):
    monkeypatch.delenv(conda_dispatch.RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(conda_dispatch.RESIDENT_WORKER_MAX_WORKERS_ENV, raising=False)
    monkeypatch.delenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, raising=False)

    assert conda_dispatch._resident_worker_request_timeout() == 21600.0
    assert conda_dispatch._resident_worker_max_workers() == 2
    assert conda_dispatch._resident_worker_idle_ttl() == 900.0

    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, "0")

    assert conda_dispatch._resident_worker_request_timeout() == 0.0


def test_expired_idle_worker_is_replaced(monkeypatch, tmp_path):
    key = ("model",)
    expired = _worker(key, last_used_at=10.0)
    conda_dispatch._RESIDENT_WORKERS[key] = expired
    stopped: list[conda_dispatch._ResidentWorker] = []
    started = _worker(key, last_used_at=1_000.0)
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "100")
    monkeypatch.setattr(conda_dispatch.time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))
    monkeypatch.setattr(conda_dispatch, "_start_resident_worker", lambda **kwargs: started)

    result = conda_dispatch._resident_worker_for(
        model_id="model",
        spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
        workspace_root=str(tmp_path),
        context=_context(key),
        log_callback=None,
    )

    assert result is started
    assert result.in_use == 1
    assert conda_dispatch._RESIDENT_WORKERS == {key: started}
    assert stopped == [expired]


def test_worker_limit_evicts_least_recently_used_idle_worker(monkeypatch, tmp_path):
    oldest = _worker(("oldest",), last_used_at=10.0)
    recent = _worker(("recent",), last_used_at=20.0)
    conda_dispatch._RESIDENT_WORKERS.update({oldest.key: oldest, recent.key: recent})
    stopped: list[conda_dispatch._ResidentWorker] = []
    replacement = _worker(("new",), last_used_at=30.0)
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_MAX_WORKERS_ENV, "2")
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))
    monkeypatch.setattr(conda_dispatch, "_start_resident_worker", lambda **kwargs: replacement)

    result = conda_dispatch._resident_worker_for(
        model_id="new",
        spec=RuntimeCondaEnvSpec(model_id="new", env_name="test", env_root=tmp_path),
        workspace_root=str(tmp_path),
        context=_context(replacement.key),
        log_callback=None,
    )

    assert result is replacement
    assert len(conda_dispatch._RESIDENT_WORKERS) == 2
    assert oldest.key not in conda_dispatch._RESIDENT_WORKERS
    assert recent.key in conda_dispatch._RESIDENT_WORKERS
    assert stopped == [oldest]


def test_worker_limit_never_evicts_busy_workers(monkeypatch, tmp_path):
    first = _worker(("first",), last_used_at=10.0, in_use=1)
    second = _worker(("second",), last_used_at=20.0, in_use=1)
    conda_dispatch._RESIDENT_WORKERS.update({first.key: first, second.key: second})
    stopped: list[conda_dispatch._ResidentWorker] = []
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_MAX_WORKERS_ENV, "2")
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "1")
    monkeypatch.setattr(conda_dispatch.time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))
    monkeypatch.setattr(
        conda_dispatch,
        "_start_resident_worker",
        lambda **kwargs: pytest.fail("a worker must not start above the configured limit"),
    )

    with pytest.raises(conda_dispatch._ResidentWorkerUnavailable, match="slots are busy"):
        conda_dispatch._resident_worker_for(
            model_id="new",
            spec=RuntimeCondaEnvSpec(model_id="new", env_name="test", env_root=tmp_path),
            workspace_root=str(tmp_path),
            context=_context(("new",)),
            log_callback=None,
        )

    assert conda_dispatch._RESIDENT_WORKERS == {first.key: first, second.key: second}
    assert stopped == []


def test_resident_worker_lease_is_released_when_dispatch_raises(monkeypatch, tmp_path):
    key = ("model",)
    worker = _worker(key, last_used_at=10.0)
    conda_dispatch._RESIDENT_WORKERS[key] = worker
    context = _context(key)
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(conda_dispatch, "_prepare_resident_run_context", lambda **kwargs: context)

    def _raise_on_send(*args, **kwargs):
        raise RuntimeError("request failed")

    monkeypatch.setattr(conda_dispatch, "_send_resident_worker_request", _raise_on_send)

    with pytest.raises(RuntimeError, match="request failed"):
        conda_dispatch._run_manager_payload_in_resident_conda(
            model_id="model",
            spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
            workspace_root=str(tmp_path),
            run_kwargs={},
            dispatch_root=Path(tmp_path) / "dispatch",
        )

    assert worker.in_use == 0


def test_waiting_for_worker_lock_honors_cancel_when_timeout_is_disabled():
    worker = _worker(("model",), last_used_at=10.0)
    worker.lock = threading.Lock()  # type: ignore[assignment]
    worker.lock.acquire()
    checks = 0

    def _cancel_requested() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    try:
        with pytest.raises(RuntimeError, match="cancelled while waiting"):
            conda_dispatch._acquire_resident_worker_request_lock(
                worker,
                deadline=None,
                request_timeout=0.0,
                cancel_requested=_cancel_requested,
            )
    finally:
        worker.lock.release()

    assert checks >= 2


def test_waiting_for_worker_lock_uses_dispatch_deadline():
    worker = _worker(("model",), last_used_at=10.0)
    worker.lock = threading.Lock()  # type: ignore[assignment]
    worker.lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="0.0s"):
            conda_dispatch._acquire_resident_worker_request_lock(
                worker,
                deadline=time.monotonic() + 0.01,
                request_timeout=0.01,
                cancel_requested=None,
            )
    finally:
        worker.lock.release()


def test_resident_dispatch_passes_original_deadline_to_result_wait(monkeypatch, tmp_path):
    key = ("model",)
    worker = _worker(key, last_used_at=100.0)
    context = _context(key)
    captured: dict[str, float | None] = {}
    sentinel = object()
    monkeypatch.setattr(conda_dispatch, "_resident_worker_request_timeout", lambda: 10.0)
    monkeypatch.setattr(conda_dispatch.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(conda_dispatch, "_prepare_resident_run_context", lambda **kwargs: context)
    monkeypatch.setattr(conda_dispatch, "_resident_worker_for", lambda **kwargs: worker)

    def _acquire(worker_arg, *, deadline, request_timeout, cancel_requested):
        captured["acquire_deadline"] = deadline
        captured["acquire_timeout"] = request_timeout
        worker_arg.lock.acquire()

    def _wait(worker_arg, **kwargs):
        captured["wait_deadline"] = kwargs["deadline"]
        captured["wait_timeout"] = kwargs["request_timeout"]
        return sentinel

    monkeypatch.setattr(conda_dispatch, "_acquire_resident_worker_request_lock", _acquire)
    monkeypatch.setattr(conda_dispatch, "_send_resident_worker_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(conda_dispatch, "_wait_for_resident_worker_result", _wait)

    result = conda_dispatch._run_manager_payload_in_resident_conda(
        model_id="model",
        spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
        workspace_root=str(tmp_path),
        run_kwargs={},
        dispatch_root=Path(tmp_path) / "dispatch-deadline",
    )

    assert result is sentinel
    assert captured == {
        "acquire_deadline": 110.0,
        "acquire_timeout": 10.0,
        "wait_deadline": 110.0,
        "wait_timeout": 10.0,
    }


def test_max_zero_disables_reuse_of_existing_worker(monkeypatch, tmp_path):
    existing = _worker(("model",), last_used_at=10.0)
    conda_dispatch._RESIDENT_WORKERS[existing.key] = existing
    stopped: list[conda_dispatch._ResidentWorker] = []
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_MAX_WORKERS_ENV, "0")
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))

    with pytest.raises(conda_dispatch._ResidentWorkerUnavailable, match="capacity is disabled"):
        conda_dispatch._resident_worker_for(
            model_id="model",
            spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
            workspace_root=str(tmp_path),
            context=_context(existing.key),
            log_callback=None,
        )

    assert conda_dispatch._RESIDENT_WORKERS == {}
    assert stopped == [existing]


def test_lowered_cap_is_enforced_before_existing_worker_reuse(monkeypatch, tmp_path):
    requested = _worker(("requested",), last_used_at=30.0)
    oldest = _worker(("oldest",), last_used_at=10.0)
    recent = _worker(("recent",), last_used_at=20.0)
    conda_dispatch._RESIDENT_WORKERS.update(
        {requested.key: requested, oldest.key: oldest, recent.key: recent}
    )
    stopped: list[conda_dispatch._ResidentWorker] = []
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_MAX_WORKERS_ENV, "2")
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))

    result = conda_dispatch._resident_worker_for(
        model_id="requested",
        spec=RuntimeCondaEnvSpec(model_id="requested", env_name="test", env_root=tmp_path),
        workspace_root=str(tmp_path),
        context=_context(requested.key),
        log_callback=None,
    )

    assert result is requested
    assert requested.in_use == 1
    assert len(conda_dispatch._RESIDENT_WORKERS) == 2
    assert oldest.key not in conda_dispatch._RESIDENT_WORKERS
    assert stopped == [oldest]


def test_reaper_collects_expired_and_dead_idle_workers_but_not_busy(monkeypatch):
    expired = _worker(("expired",), last_used_at=10.0)
    dead = _worker(("dead",), last_used_at=999.0)
    dead.process.returncode = 1
    busy = _worker(("busy",), last_used_at=10.0, in_use=1)
    conda_dispatch._RESIDENT_WORKERS.update(
        {expired.key: expired, dead.key: dead, busy.key: busy}
    )
    stopped: list[conda_dispatch._ResidentWorker] = []
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "100")
    monkeypatch.setattr(conda_dispatch, "_shutdown_resident_worker", lambda worker, force=False: stopped.append(worker))

    count = conda_dispatch._reap_resident_workers_once(now=1_000.0)

    assert count == 2
    assert conda_dispatch._RESIDENT_WORKERS == {busy.key: busy}
    assert stopped == [expired, dead]


def test_worker_start_and_write_os_errors_are_resident_unavailable(monkeypatch, tmp_path):
    context = _context(("model",))
    monkeypatch.setattr(
        conda_dispatch.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot spawn")),
    )
    with pytest.raises(conda_dispatch._ResidentWorkerUnavailable, match="cannot spawn"):
        conda_dispatch._start_resident_worker(
            model_id="model",
            spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
            workspace_root=str(tmp_path),
            context=context,
            log_callback=None,
        )

    worker = _worker(("write",), last_used_at=10.0)

    class _BrokenInput:
        def fileno(self):
            raise OSError("write failed")

        def write(self, data):
            raise OSError("write failed")

        def flush(self):
            pass

    worker.process.stdin = _BrokenInput()  # type: ignore[attr-defined]
    with pytest.raises(conda_dispatch._ResidentWorkerUnavailable, match="pipe closed"):
        conda_dispatch._send_resident_worker_request(worker, request_payload={"request_id": "x"})


def test_shutdown_reaps_after_terminate_and_kill(monkeypatch):
    class _Pipe:
        def __init__(self) -> None:
            self.closed = False

        def write(self, data):
            return len(data)

        def flush(self):
            pass

        def close(self):
            self.closed = True

    class _StubbornProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.stdin = _Pipe()
            self.stdout = _Pipe()
            self.wait_calls: list[float | None] = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) <= 2:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -9
            return self.returncode

    process = _StubbornProcess()
    worker = _worker(("stubborn",), last_used_at=10.0)
    worker.process = process  # type: ignore[assignment]
    terminations: list[float] = []

    def stop(process, *, grace_seconds):
        terminations.append(grace_seconds)
        process.returncode = -9

    monkeypatch.setattr(
        conda_dispatch,
        "terminate_process_group",
        stop,
    )

    conda_dispatch._shutdown_resident_worker(worker, force=False)

    assert terminations == [2]
    assert process.wait_calls == [2]
    assert process.stdin.closed
    assert process.stdout.closed


def test_start_log_callback_runs_outside_lifecycle_lock_and_cannot_leak_lease(monkeypatch, tmp_path):
    key = ("model",)
    started = _worker(key, last_used_at=10.0)
    monkeypatch.setenv(conda_dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.setattr(conda_dispatch, "_start_resident_worker", lambda **kwargs: started)
    monkeypatch.setattr(conda_dispatch, "_ensure_resident_worker_reaper", lambda: None)

    def _reentrant_log(stream: str, text: str) -> None:
        assert conda_dispatch._RESIDENT_WORKERS_LIFECYCLE_LOCK.acquire(blocking=False)
        conda_dispatch._RESIDENT_WORKERS_LIFECYCLE_LOCK.release()
        raise RuntimeError("log callback failed")

    with pytest.raises(RuntimeError, match="log callback failed"):
        conda_dispatch._resident_worker_for(
            model_id="model",
            spec=RuntimeCondaEnvSpec(model_id="model", env_name="test", env_root=tmp_path),
            workspace_root=str(tmp_path),
            context=_context(key),
            log_callback=_reentrant_log,
        )

    assert started.in_use == 0
