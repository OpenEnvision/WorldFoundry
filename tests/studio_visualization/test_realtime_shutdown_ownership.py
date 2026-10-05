"""Bounded teardown must still issue cleanup and own its late result."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends.world_realtime import ResidentWorldRuntime, _run_by_shutdown_deadline


def _runtime(postprocess, manager=None):
    return ResidentWorldRuntime(
        manager=manager or SimpleNamespace(), entry=find_entry("sana-wm"),
        launch_config=StudioLaunchConfig(model_id="sana-wm", frontend="world"),
        fps=16, postprocess=postprocess,
    )


def test_expired_budget_still_issues_cleanup_and_consumes_immediate_failure(caplog):
    async def exercise():
        issued = []
        unhandled = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))

        async def cleanup():
            issued.append(True)
            raise RuntimeError("expired cleanup failed")

        retained = set()
        await _run_by_shutdown_deadline(
            cleanup(), deadline=loop.time() - 1, retained=retained, label="expired cleanup",
        )
        assert issued == [True]
        assert not retained
        await asyncio.sleep(0)
        assert not unhandled

    asyncio.run(exercise())
    assert "expired cleanup failed" in caplog.text


@pytest.mark.parametrize("configured", [False, True])
def test_reset_preserves_primary_failure_and_attempts_model_cleanup(configured):
    failure = RuntimeError("postprocess reset failed")
    actions = []

    def fail():
        raise failure

    runtime = _runtime(SimpleNamespace(reset=fail), SimpleNamespace(run_realtime=lambda **kwargs: actions.append(kwargs)))
    request = object() if configured else None
    runtime._base_request = request
    runtime._configured = configured

    async def exercise():
        with pytest.raises(RuntimeError) as caught:
            await runtime.reset()
        assert caught.value is failure
        assert not runtime._configured
        assert runtime._base_request is None
        assert len(actions) == int(configured)
        if configured:
            assert actions[0]["request"] is request
            assert actions[0]["action"] == "reset"

    try:
        asyncio.run(exercise())
    finally:
        runtime._executor.shutdown(wait=True)


def test_timed_out_reset_retains_late_failure_and_completes_model_cleanup(monkeypatch, caplog):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")
    entered, release = threading.Event(), threading.Event()
    actions = []

    def reset():
        entered.set()
        assert release.wait(2)
        raise RuntimeError("late postprocess reset failed")

    runtime = _runtime(SimpleNamespace(reset=reset), SimpleNamespace(run_realtime=lambda **kwargs: actions.append(kwargs)))
    request = object()
    runtime._base_request = request
    runtime._configured = True

    async def exercise():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        await asyncio.wait_for(runtime.reset(), timeout=0.2)
        assert entered.is_set()
        assert runtime._base_request is None and not runtime._configured
        retained = tuple(runtime._shutdown_tasks)
        assert len(retained) == 1 and not retained[0].done()
        assert not actions
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*retained, return_exceptions=True), timeout=0.2)
        assert str(results[0]) == "late postprocess reset failed"
        assert actions[0]["request"] is request and actions[0]["action"] == "reset"
        await asyncio.sleep(0)
        assert not runtime._shutdown_tasks and not unhandled

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        runtime._executor.shutdown(wait=True)
    assert "late postprocess reset failed" in caplog.text


def test_close_drains_accepted_cleanup_queued_behind_uncancelable_inference(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")
    entered, release = threading.Event(), threading.Event()
    order = []

    def inference():
        entered.set()
        assert release.wait(2)
        order.append("inference")

    runtime = _runtime(
        SimpleNamespace(reset=lambda: order.append("postprocess")),
        SimpleNamespace(run_realtime=lambda **kwargs: order.append("model-reset")),
    )
    runtime._configured = True
    runtime._base_request = object()

    async def exercise():
        running = asyncio.create_task(runtime._run(inference))
        while not entered.is_set():
            await asyncio.sleep(0)
        await asyncio.wait_for(runtime.close(), timeout=0.2)
        assert runtime._closed and not runtime._configured
        assert not order
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            await runtime._run(lambda: None)
        retained = tuple(runtime._shutdown_tasks)
        assert retained
        release.set()
        await asyncio.wait_for(asyncio.gather(running, *retained, return_exceptions=True), timeout=0.2)
        await asyncio.sleep(0)
        assert order == ["inference", "postprocess", "model-reset"]
        assert not runtime._shutdown_tasks

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        runtime._executor.shutdown(wait=True)
