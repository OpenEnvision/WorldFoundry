"""CPU regressions for GPU admission, bounded dispatch and worker teardown."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from worldfoundry.core.execution.process import process_group_alive, terminate_process_group
from worldfoundry.runtime.device_pool import CudaDeviceLeasePool
from worldfoundry.studio.inference import dispatch as dispatch


@pytest.fixture(autouse=True)
def isolated_workers(monkeypatch):
    monkeypatch.delenv(dispatch.STUDIO_CONDA_CHILD_ENV, raising=False)
    monkeypatch.setenv(dispatch.AUTO_GPU_PLACEMENT_ENV, "1")
    monkeypatch.setenv(dispatch.RESIDENT_WORKERS_ENV, "1")
    monkeypatch.setenv(dispatch.RESIDENT_WORKER_IDLE_TTL_ENV, "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(dispatch, "_ensure_resident_worker_reaper", lambda: None)
    with dispatch._RESIDENT_WORKERS_LOCK:
        assert not dispatch._RESIDENT_WORKERS
    yield
    dispatch._shutdown_all_resident_workers()


def _options(tmp_path):
    return dict(model_id="regression-model",
                spec=dispatch._LocalRuntimeSpec(model_id="regression-model", env_name=Path(sys.prefix).name,
                                                env_root=Path(sys.prefix).parent),
                workspace_root=str(tmp_path), dispatch_root=tmp_path / "dispatch",
                run_kwargs={"device": "cuda"}, log_callback=lambda *_: None)


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_resident_fallback_cannot_bypass_an_occupied_gpu(tmp_path, monkeypatch, device, reason):
    pool = CudaDeviceLeasePool(("0",))
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    monkeypatch.setenv(dispatch.INFERENCE_TIMEOUT_ENV, "0.15")
    monkeypatch.setattr(dispatch, "_run_manager_payload_in_resident_conda",
                        lambda **_: (_ for _ in ()).throw(dispatch._ResidentWorkerUnavailable("pool full")))
    monkeypatch.setattr(dispatch.subprocess, "Popen",
                        lambda *_, **__: pytest.fail("occupied GPU reached subprocess launch"))
    start = time.monotonic()
    options = _options(tmp_path)
    options["run_kwargs"] = {"device": device}
    if reason == "cancel":
        options["cancel_requested"] = lambda: time.monotonic() - start >= 0.05
    with pool.acquire():
        with pytest.raises(TimeoutError if reason == "timeout" else RuntimeError,
                           match="timed out|cancelled"):
            dispatch.run_manager_payload_in_conda(**options)
        assert pool.available_count == 0
    assert pool.available_count == 1


def test_waiting_for_resident_gpu_is_cancellable_without_holding_lifecycle_lock(tmp_path, monkeypatch):
    pool = CudaDeviceLeasePool(("0",))
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    cancelled = threading.Event()
    errors = []
    options = _options(tmp_path)
    options.pop("run_kwargs")
    options.pop("dispatch_root")
    options["context"] = dispatch._ResidentRunContext({}, {}, {}, ("regression-model",), 1)
    options["cancel_requested"] = cancelled.is_set

    def allocate():
        try:
            dispatch._resident_worker_for(**options)
        except Exception as exc:
            errors.append(exc)

    with pool.acquire():
        thread = threading.Thread(target=allocate)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while pool.waiting_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pool.waiting_count == 1
            assert dispatch._RESIDENT_WORKERS_LIFECYCLE_LOCK.acquire(timeout=0.1)
            dispatch._RESIDENT_WORKERS_LIFECYCLE_LOCK.release()
        finally:
            cancelled.set()
            thread.join(timeout=2)
        assert not thread.is_alive()
    assert len(errors) == 1 and "cancelled" in str(errors[0])


def test_resident_gpu_queue_uses_the_request_deadline(tmp_path, monkeypatch):
    pool = CudaDeviceLeasePool(("0",))
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    options = _options(tmp_path)
    options.pop("run_kwargs")
    options.pop("dispatch_root")
    options["context"] = dispatch._ResidentRunContext({}, {}, {}, ("regression-model",), 1)
    with pool.acquire():
        with pytest.raises(TimeoutError):
            dispatch._resident_worker_for(**options, deadline=time.monotonic() + 0.1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("failure", ["callback", "select", "timeout", "cancel"])
def test_one_shot_failures_stop_descendants_before_releasing_gpu(tmp_path, monkeypatch, failure):
    pool = CudaDeviceLeasePool(("0",), lock_dir=tmp_path / "locks")
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    monkeypatch.setenv(dispatch.INFERENCE_TIMEOUT_ENV, "20")
    monkeypatch.setattr(dispatch, "_run_manager_payload_in_resident_conda", lambda **_: None)
    real_popen = subprocess.Popen
    processes = []
    ready = tmp_path / "descendant-ready"
    if failure == "timeout":
        real_monotonic = time.monotonic
        # Expire the request only once the SIGTERM-ignoring descendant exists.
        # Keep time advancing so teardown's grace-period clocks still work.
        monkeypatch.setattr(dispatch.time, "monotonic",
                            lambda: real_monotonic() + (60 if ready.exists() else 0))
    child_script = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(120)"
    )
    parent_script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}]); "
        f"ready=Path({str(ready)!r});\n"
        "while not ready.exists(): time.sleep(0.01)\n"
        "print('ready',flush=True); time.sleep(120)"
    )

    def launch(_command, **kwargs):
        process = real_popen([sys.executable, "-c", parent_script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(dispatch.subprocess, "Popen", launch)
    options = _options(tmp_path)
    cancelled = threading.Event()

    def log(stream, _text):
        if stream == "stdout":
            if failure == "callback":
                raise ValueError("observer failed")
            if failure == "cancel":
                cancelled.set()

    options["log_callback"] = log
    options["cancel_requested"] = cancelled.is_set
    if failure == "select":
        real_select = dispatch.select.select

        def fail_after_descendant_starts(*args):
            if ready.exists():
                raise ValueError("select failed")
            return real_select(*args)

        monkeypatch.setattr(dispatch.select, "select", fail_after_descendant_starts)
    expected_error, message = {
        "callback": (ValueError, "observer failed"),
        "select": (ValueError, "select failed"),
        "timeout": (TimeoutError, "inference request timed out"),
        "cancel": (RuntimeError, "inference dispatch cancelled"),
    }[failure]
    try:
        with pytest.raises(expected_error, match=message):
            dispatch.run_manager_payload_in_conda(**options)
        assert len(processes) == 1
        assert ready.exists()
        assert not process_group_alive(processes[0].pid)
        assert pool.available_count == 1
        with CudaDeviceLeasePool(("0",), lock_dir=tmp_path / "locks").acquire(deadline=time.monotonic() + 1):
            pass
    finally:
        for process in processes:
            terminate_process_group(process, grace_seconds=0)


def test_allocation_log_failure_returns_the_lease_without_spawning(tmp_path, monkeypatch):
    pool = CudaDeviceLeasePool(("0",))
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    monkeypatch.setattr(dispatch, "_run_manager_payload_in_resident_conda", lambda **_: None)
    options = _options(tmp_path)
    options["log_callback"] = lambda *_: (_ for _ in ()).throw(ValueError("observer failed"))
    with pytest.raises(ValueError, match="observer failed"):
        dispatch.run_manager_payload_in_conda(**options)
    assert pool.available_count == 1


def test_reaper_start_failure_stops_the_worker_before_releasing_its_lease(tmp_path, monkeypatch):
    import codecs

    pool = CudaDeviceLeasePool(("0",), lock_dir=tmp_path / "locks")
    monkeypatch.setattr(dispatch, "_automatic_gpu_pool", lambda: pool)
    spawned = []

    def start_worker(**kwargs):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                   start_new_session=True)
        spawned.append(process)
        lease = kwargs["device_lease"]
        lease.process_group_id = process.pid
        return dispatch._ResidentWorker(key=kwargs["context"].key, base_key=kwargs["base_key"],
                                        model_id=kwargs["model_id"], process=process,
                                        lock=threading.RLock(), decoder=codecs.getincrementaldecoder("utf-8")(),
                                        command=[], created_at=time.monotonic(), last_used_at=time.monotonic(),
                                        device_lease=lease)

    monkeypatch.setattr(dispatch, "_start_resident_worker", start_worker)
    monkeypatch.setattr(dispatch, "_ensure_resident_worker_reaper",
                        lambda: (_ for _ in ()).throw(RuntimeError("cannot start thread")))
    options = _options(tmp_path)
    options.pop("run_kwargs")
    options.pop("dispatch_root")
    options["context"] = dispatch._ResidentRunContext({}, {}, {}, ("regression-model",), 1)
    try:
        with pytest.raises(RuntimeError, match="cannot start thread"):
            dispatch._resident_worker_for(**options)
        assert len(spawned) == 1
        assert not process_group_alive(spawned[0].pid)
        assert pool.available_count == 1
        assert not dispatch._RESIDENT_WORKERS
    finally:
        for process in spawned:
            terminate_process_group(process, grace_seconds=0)


@pytest.mark.parametrize("configured", [False, True])
def test_managed_local_dispatch_uses_the_exact_current_interpreter(monkeypatch, configured):
    current = dispatch._LocalRuntimeSpec(model_id="regression-model", env_name=Path(sys.prefix).name,
                                         env_root=Path(sys.prefix).parent)
    monkeypatch.setattr(dispatch, "workspace_runtime_spec", lambda _: current if configured else None)
    monkeypatch.setattr(dispatch, "_force_subprocess_for_model", lambda _: False)
    assert dispatch.dispatch_spec_for_inference("regression-model") is None
    spec = dispatch.dispatch_spec_for_inference("regression-model", force_subprocess=True)
    assert str(spec.python_executable) == sys.executable
    assert dispatch.dispatch_spec_for_inference("regression-model", backend="api_init", force_subprocess=True) is None
    monkeypatch.setenv(dispatch.STUDIO_CONDA_CHILD_ENV, "1")
    assert dispatch.dispatch_spec_for_inference("regression-model", force_subprocess=True) is None
