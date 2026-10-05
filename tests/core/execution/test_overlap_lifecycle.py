"""Exercise recovery and contention in the public overlap runner contract."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from worldfoundry.core.execution.realtime.overlap import HostThreadOverlap, SynchronousOverlap


@pytest.mark.parametrize("runner_type", [SynchronousOverlap, HostThreadOverlap])
def test_worker_error_does_not_poison_the_next_request(runner_type):
    runner = runner_type()
    failure = ValueError("decode failed")
    outputs = []

    def fail():
        raise failure

    try:
        if runner_type is SynchronousOverlap:
            with pytest.raises(ValueError, match="decode failed"):
                runner.submit(fail)
        else:
            runner.submit(fail)
        assert runner.wait(timeout_s=5)
        assert runner.last_error is failure
        with pytest.raises(ValueError, match="decode failed"):
            runner.wait(raise_error=True)
        runner.submit(lambda: outputs.append("next frame"))
        assert runner.wait(timeout_s=5, raise_error=True)
        assert outputs == ["next frame"]
        assert runner.last_error is None
    finally:
        runner.close()


@pytest.mark.parametrize("failure_stage", ["construction", "start"])
@pytest.mark.parametrize("failure_type", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_thread_launch_failure_leaves_runner_idle_and_reusable(monkeypatch, failure_stage, failure_type):
    runner = HostThreadOverlap()
    failure = failure_type("cannot launch presentation thread")
    outputs = []

    def fail(*args, **kwargs):
        raise failure

    with monkeypatch.context() as patch:
        if failure_stage == "construction":
            patch.setattr(threading, "Thread", fail)
        else:
            patch.setattr(threading.Thread, "start", fail)
        with pytest.raises(failure_type, match="cannot launch"):
            runner.submit(lambda: outputs.append("must not execute"))

    assert not runner.pending
    assert runner.wait(timeout_s=0)
    assert runner.last_error is failure
    with pytest.raises(failure_type, match="cannot launch"):
        runner.wait(raise_error=True)
    runner.close()
    runner.submit(lambda: outputs.append("recovered"))
    try:
        assert runner.wait(timeout_s=5, raise_error=True)
        assert outputs == ["recovered"]
        assert runner.last_error is None
    finally:
        runner.close()


def test_failure_after_thread_started_does_not_release_its_pending_slot(monkeypatch):
    runner = HostThreadOverlap()
    release = threading.Event()
    started = threading.Event()
    original_start = threading.Thread.start

    def work():
        started.set()
        assert release.wait(timeout=5)

    def fail_after_start(thread):
        original_start(thread)
        raise RuntimeError("failure after thread launch")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", fail_after_start)
            with pytest.raises(RuntimeError, match="after thread launch"):
                runner.submit(work)
        assert started.wait(timeout=5)
        assert runner.pending
        assert not runner.wait(timeout_s=0)
        with pytest.raises(RuntimeError, match="pending overlap"):
            runner.submit(lambda: None)
        release.set()
        assert runner.wait(timeout_s=5)
        runner.submit(lambda: None)
        assert runner.wait(timeout_s=5, raise_error=True)
    finally:
        release.set()
        runner.close()


def test_concurrent_submitters_admit_exactly_one_callback():
    runner = HostThreadOverlap()
    callers = 8
    barrier = threading.Barrier(callers)
    release = threading.Event()
    started = threading.Event()
    executed = []

    def submit(index):
        barrier.wait(timeout=5)

        def work():
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release callback")
            executed.append(index)

        try:
            runner.submit(work)
        except RuntimeError as exc:
            assert "pending overlap" in str(exc)
            return None
        return index

    try:
        with ThreadPoolExecutor(max_workers=callers) as pool:
            futures = [pool.submit(submit, index) for index in range(callers)]
            admitted = [value for future in futures if (value := future.result(timeout=5)) is not None]
        assert started.wait(timeout=5)
        assert len(admitted) == 1
        assert runner.pending
        assert not runner.wait(timeout_s=0)
        release.set()
        assert runner.wait(timeout_s=5, raise_error=True)
        assert executed == admitted
    finally:
        release.set()
        runner.close()


def test_close_without_wait_preserves_pending_work_until_it_finishes():
    runner = HostThreadOverlap()
    release = threading.Event()
    outputs = []

    def work():
        assert release.wait(timeout=5)
        outputs.append("finished")

    try:
        runner.submit(work)
        runner.close(wait=False)
        assert runner.pending
        assert outputs == []
        release.set()
        runner.close(wait=True)
        assert not runner.pending
        assert outputs == ["finished"]
        runner.close()
    finally:
        release.set()
        runner.close()


@pytest.mark.parametrize("timeout", [-1, -0.001])
def test_negative_wait_timeout_is_rejected_without_discarding_work(timeout):
    runner = HostThreadOverlap()
    with pytest.raises(ValueError, match="non-negative"):
        runner.wait(timeout_s=timeout)
    runner.submit(lambda: None)
    assert runner.wait(timeout_s=5, raise_error=True)
    runner.close()
