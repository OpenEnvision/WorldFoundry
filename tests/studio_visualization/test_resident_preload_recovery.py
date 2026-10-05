from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import PreparedInputs
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends.world_realtime import ResidentWorldRuntime


class Manager:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_started = threading.Event()
        self.reset_release = threading.Event()
        self.release.set()
        self.reset_release.set()
        self.calls = []
        self.dirty = False
        self.failure = None
        self.reset_failure = None

    def run_realtime(self, *, entry, request, action):
        self.calls.append(action)
        if action == "configure":
            self.dirty = True
            self.started.set()
            assert self.release.wait(5)
            if self.failure is not None:
                raise self.failure
        elif action == "reset":
            self.reset_started.set()
            assert self.reset_release.wait(5)
            if self.reset_failure is not None:
                raise self.reset_failure
            self.dirty = False
        return {"video": np.zeros((2, 4, 5, 3), dtype=np.uint8)}


@pytest.fixture
def setup(tmp_path):
    entry = find_entry("infinite-world")
    manager = Manager()
    resident = ResidentWorldRuntime(
        manager=manager, entry=entry,
        launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"), fps=16,
    )
    base = PreparedInputs(
        prompt="A", input_path="", image=None, image_path=None, video_path=None,
        last_frame=None, last_frame_path=None, reference_images=[], reference_image_paths=[],
        interactions=[], camera_view=None, task_type="", intrinsics=None, meta_path="",
        panorama_path="", scene_name="", fps=16, num_frames=2, output_dir=str(tmp_path),
        output_path=str(tmp_path / "unused.mp4"), call_kwargs={}, load_kwargs={}, model_ref="",
        backend="from_pretrained", endpoint="", api_key="", device="cpu",
    )
    resident._build_request = lambda **values: replace(base, prompt=values["prompt"], image=values["image"])
    yield resident, manager
    manager.release.set()
    manager.reset_release.set()
    manager.reset_failure = None
    resident._executor.shutdown(wait=True, cancel_futures=True)


async def wait_event(event):
    assert await asyncio.to_thread(event.wait, 2), "worker did not reach the barrier"


@pytest.mark.parametrize("cancel_first", [True, False])
def test_cancelling_one_preload_waiter_keeps_the_shared_load_alive(setup, cancel_first):
    resident, manager = setup
    manager.release.clear()

    async def exercise():
        first = asyncio.create_task(resident.preload())
        await wait_event(manager.started)
        second = asyncio.create_task(resident.preload())
        await asyncio.sleep(0)
        cancelled, survivor = (first, second) if cancel_first else (second, first)
        try:
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert not resident.ready
            manager.release.set()
            await survivor
            assert resident.ready and resident.preload_error is None
            assert manager.calls.count("configure") == 1
            await resident.preload()
            assert manager.calls.count("configure") == 1
        finally:
            manager.release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            await resident.close()

    asyncio.run(exercise())


def test_a_preload_waiter_timeout_does_not_cancel_the_model_load(setup):
    resident, manager = setup
    manager.release.clear()

    async def exercise():
        first = asyncio.create_task(resident.preload())
        await wait_event(manager.started)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(first, 0.01)
            manager.release.set()
            await resident.preload()
            assert resident.ready and manager.calls.count("configure") == 1
        finally:
            manager.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await resident.close()

    asyncio.run(exercise())


def test_preload_failure_is_shared_and_retry_clears_the_error_after_cleanup(setup):
    resident, manager = setup
    manager.release.clear()
    failure = ValueError("load failed after mutation")
    manager.failure = failure

    async def exercise():
        first = asyncio.create_task(resident.preload())
        await wait_event(manager.started)
        second = asyncio.create_task(resident.preload())
        await asyncio.sleep(0)
        try:
            manager.release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            assert results == [failure, failure]
            assert not resident.ready and resident.preload_error == str(failure)
            assert not manager.dirty and manager.calls == ["configure", "reset"]
            manager.failure = None
            await resident.preload()
            assert resident.ready and resident.preload_error is None
            assert manager.calls.count("configure") == 2
        finally:
            manager.release.set()
            manager.failure = None
            await resident.close()

    asyncio.run(exercise())


def test_retry_waits_for_failed_attempt_cleanup_to_finish(setup):
    resident, manager = setup
    manager.failure = ValueError("configure failed")
    manager.reset_release.clear()

    async def exercise():
        first = asyncio.create_task(resident.preload())
        try:
            await wait_event(manager.reset_started)
            retry = asyncio.create_task(resident.preload())
            await asyncio.sleep(0)
            assert not first.done() and not retry.done()
            assert manager.calls.count("configure") == 1
            manager.reset_release.set()
            errors = await asyncio.gather(first, retry, return_exceptions=True)
            assert errors == [manager.failure, manager.failure]
            manager.failure = None
            await resident.preload()
            assert resident.ready
        finally:
            manager.reset_release.set()
            manager.failure = None
            await asyncio.gather(first, return_exceptions=True)
            await resident.close()

    asyncio.run(exercise())


def test_failed_cleanup_blocks_a_new_load_until_reset_succeeds(setup):
    resident, manager = setup
    manager.failure = ValueError("primary load error")
    manager.reset_failure = ValueError("reset failed")

    async def exercise():
        try:
            with pytest.raises(ValueError, match="primary load error") as caught:
                await resident.preload()
            assert manager.dirty and resident._reset_request is not None
            if callable(getattr(caught.value, "add_note", None)):
                assert any("reset failed" in note for note in caught.value.__notes__)
            manager.failure = None
            with pytest.raises(ValueError, match="reset failed"):
                await resident.preload()
            assert manager.calls.count("configure") == 1
            manager.reset_failure = None
            await resident.preload()
            assert resident.ready and resident.preload_error is None
            assert manager.calls.count("configure") == 2
        finally:
            manager.failure = manager.reset_failure = None
            await resident.close()

    asyncio.run(exercise())


def test_cancelled_owned_preload_is_cleaned_before_retry(setup):
    resident, manager = setup
    manager.release.clear()

    async def exercise():
        first = asyncio.create_task(resident.preload())
        await wait_event(manager.started)
        try:
            resident._preload_task.cancel()
            await asyncio.sleep(0)
            manager.release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not manager.dirty
            await resident.preload()
            assert resident.ready and manager.calls.count("configure") == 2
        finally:
            manager.release.set()
            await asyncio.gather(first, return_exceptions=True)
            await resident.close()

    asyncio.run(exercise())


def test_close_during_preload_is_bounded_and_cleans_the_worker_after_it_returns(setup, monkeypatch):
    resident, manager = setup
    manager.release.clear()
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.02")

    async def exercise():
        caller = asyncio.create_task(resident.preload())
        await wait_event(manager.started)
        try:
            await asyncio.wait_for(resident.close(), 0.5)
            assert not resident.ready
            with pytest.raises(RuntimeError, match="closed"):
                await resident.preload()
        finally:
            manager.release.set()
            await asyncio.gather(caller, *tuple(resident._shutdown_tasks), return_exceptions=True)
        assert not resident.ready and not resident._configured
        assert not manager.dirty and "reset" in manager.calls

    asyncio.run(exercise())


@pytest.mark.parametrize("failure_stage", ["stream", "flush"])
def test_preload_warmup_failure_resets_state_and_can_retry(setup, tmp_path, failure_stage):
    resident, manager = setup
    seed = tmp_path / "seed.png"
    Image.new("RGB", (5, 4)).save(seed)
    resident.warmup_image_path, resident.warmup_chunks = str(seed), 2
    original = manager.run_realtime
    finish = resident._postprocess.finish
    fail = True

    def run(**kwargs):
        if fail and failure_stage == "stream" and kwargs["action"] == "stream":
            raise ValueError("warmup stream failed")
        return original(**kwargs)

    def flush():
        if fail and failure_stage == "flush":
            raise ValueError("warmup flush failed")
        return finish()

    manager.run_realtime = run
    resident._postprocess.finish = flush

    async def exercise():
        nonlocal fail
        try:
            with pytest.raises(ValueError, match="warmup .* failed"):
                await resident.preload()
            assert not manager.dirty and not resident.ready
            assert resident._postprocess.input_spec is None
            fail = False
            await resident.preload()
            assert resident.ready and resident.preload_error is None
            assert not manager.dirty and resident._postprocess.chunk_index == 0
        finally:
            fail = False
            await resident.close()

    asyncio.run(exercise())


def test_postprocess_cleanup_failure_must_be_retried_before_loading(setup):
    resident, manager = setup
    manager.failure = ValueError("load failed")
    reset = resident._postprocess.reset
    failing = True

    def reset_postprocess():
        if failing:
            raise ValueError("postprocess reset failed")
        return reset()

    resident._postprocess.reset = reset_postprocess

    async def exercise():
        nonlocal failing
        try:
            with pytest.raises(ValueError, match="load failed"):
                await resident.preload()
            manager.failure = None
            with pytest.raises(ValueError, match="postprocess reset failed"):
                await resident.preload()
            assert manager.calls.count("configure") == 1
            failing = False
            await resident.preload()
            assert resident.ready and resident.preload_error is None
        finally:
            failing = False
            manager.failure = None
            await resident.close()

    asyncio.run(exercise())


def test_preload_preserves_a_primary_error_without_python311_exception_notes(setup):
    class LegacyError(ValueError):
        add_note = None

    resident, manager = setup
    manager.failure = LegacyError("primary load error")
    manager.reset_failure = ValueError("secondary reset error")

    async def exercise():
        try:
            with pytest.raises(LegacyError, match="primary load error"):
                await resident.preload()
            manager.failure = manager.reset_failure = None
            await resident.preload()
            assert resident.ready
        finally:
            manager.failure = manager.reset_failure = None
            await resident.close()

    asyncio.run(exercise())


def test_preload_warmup_timeout_can_retry_after_the_worker_is_clean(setup, tmp_path, monkeypatch):
    from worldfoundry.core.execution.realtime.prewarm import PrewarmTimeoutError

    resident, manager = setup
    seed = tmp_path / "seed.png"
    Image.new("RGB", (5, 4)).save(seed)
    resident.warmup_image_path, resident.warmup_chunks = str(seed), 2
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PREWARM_TIMEOUT_SECONDS", "0.000001")

    async def exercise():
        try:
            with pytest.raises(PrewarmTimeoutError):
                await resident.preload()
            assert not manager.dirty and not resident.ready
            monkeypatch.setenv("WORLDFOUNDRY_REALTIME_PREWARM_TIMEOUT_SECONDS", "0")
            await resident.preload()
            assert resident.ready and resident.preload_error is None
        finally:
            await resident.close()

    asyncio.run(exercise())
