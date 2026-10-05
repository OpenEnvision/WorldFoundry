"""Closing admission must preserve reset work already owned by the runtime."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends.world_realtime import ResidentWorldRuntime


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")
    entry = find_entry("sana-wm")
    value = ResidentWorldRuntime(
        manager=SimpleNamespace(), entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"), fps=16,
        postprocess=SimpleNamespace(reset=lambda: None),
    )
    yield value
    value._executor.shutdown(wait=True, cancel_futures=True)


async def _wait(event):
    assert await asyncio.to_thread(event.wait, 5), "worker did not reach the barrier"


async def _reject_new_work(runtime, effects):
    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        await runtime._run(lambda: effects.append("new-work"))
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.preload()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.configure(prompt="A", image_path="", video_path="")
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.generate([], seed=43)
    with pytest.raises(RuntimeError, match="closed"):
        await runtime._warmup()


def test_concurrent_close_rejects_new_work_and_drains_accepted_reset_once(runtime):
    entered, release = threading.Event(), threading.Event()
    order = []

    def inference():
        entered.set()
        assert release.wait(5)
        order.append("inference")

    runtime._postprocess.reset = lambda: order.append("postprocess")
    runtime.manager.run_realtime = lambda **kwargs: order.append(kwargs["action"])
    runtime._base_request, runtime._configured = object(), True

    async def exercise():
        running = asyncio.create_task(runtime._run(inference))
        try:
            await _wait(entered)
            await asyncio.wait_for(asyncio.gather(runtime.close(), runtime.close()), 0.5)
            assert not order and runtime._shutdown_tasks
            await asyncio.wait_for(_reject_new_work(runtime, order), 0.2)
            release.set()
            await asyncio.wait_for(asyncio.gather(running, *tuple(runtime._shutdown_tasks), return_exceptions=True), 1)
            await asyncio.sleep(0)
            assert order == ["inference", "postprocess", "reset"]
            assert not runtime._shutdown_tasks and runtime._reset_request is None
            await runtime.close()
            await asyncio.wait_for(_reject_new_work(runtime, order), 0.2)
        finally:
            release.set()
            await asyncio.gather(running, *tuple(runtime._shutdown_tasks), return_exceptions=True)

    asyncio.run(exercise())


def test_cancelled_close_still_owns_cleanup_and_keeps_admission_closed(runtime):
    entered, release = threading.Event(), threading.Event()
    order = []

    def inference():
        entered.set()
        assert release.wait(5)
        order.append("inference")

    runtime._postprocess.reset = lambda: order.append("postprocess")
    runtime.manager.run_realtime = lambda **kwargs: order.append(kwargs["action"])
    runtime._base_request, runtime._configured = object(), True

    async def exercise():
        running = asyncio.create_task(runtime._run(inference))
        closing = None
        try:
            await _wait(entered)
            closing = asyncio.create_task(runtime.close(deadline=asyncio.get_running_loop().time() + 5))
            while runtime._reset_task is None:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await asyncio.wait_for(_reject_new_work(runtime, order), 0.2)
            assert runtime._shutdown_tasks and not order
            release.set()
            await asyncio.wait_for(asyncio.gather(running, *tuple(runtime._shutdown_tasks), return_exceptions=True), 1)
            await asyncio.sleep(0)
            assert order == ["inference", "postprocess", "reset"]
            assert not runtime._shutdown_tasks
        finally:
            release.set()
            tasks = [running, *tuple(runtime._shutdown_tasks)]
            if closing is not None:
                tasks.append(closing)
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())


def test_late_close_cleanup_failure_is_consumed_and_model_reset_still_runs(runtime, caplog):
    entered, release = threading.Event(), threading.Event()
    order = []

    def reset():
        entered.set()
        assert release.wait(5)
        order.append("postprocess")
        raise ValueError("late close cleanup failed")

    runtime._postprocess.reset = reset
    runtime.manager.run_realtime = lambda **kwargs: order.append(kwargs["action"])
    runtime._base_request, runtime._configured = object(), True

    async def exercise():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        try:
            await runtime.close()
            await _wait(entered)
            await asyncio.wait_for(_reject_new_work(runtime, order), 0.2)
            assert runtime._shutdown_tasks
            release.set()
            await asyncio.wait_for(asyncio.gather(*tuple(runtime._shutdown_tasks), return_exceptions=True), 1)
            await asyncio.sleep(0)
            assert order == ["postprocess", "reset"]
            assert not runtime._shutdown_tasks and not unhandled
        finally:
            release.set()
            await asyncio.gather(*tuple(runtime._shutdown_tasks), return_exceptions=True)

    asyncio.run(exercise())
    assert "late close cleanup failed" in caplog.text
