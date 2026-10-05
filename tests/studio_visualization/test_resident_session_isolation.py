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


def request(tmp_path):
    return PreparedInputs(prompt="A", input_path="seed.png", image=Image.new("RGB", (5, 4)),
                          image_path="seed.png", video_path=None, last_frame=None, last_frame_path=None,
                          reference_images=[], reference_image_paths=[], interactions=[], camera_view=None,
                          task_type="", intrinsics=None, meta_path="", panorama_path="", scene_name="",
                          fps=16, num_frames=2, output_dir=str(tmp_path), output_path=str(tmp_path / "unused.mp4"),
                          call_kwargs={"seed": 0}, load_kwargs={}, model_ref="", backend="from_pretrained",
                          endpoint="", api_key="", device="cpu")


class Manager:
    def __init__(self):
        self.calls = []
        self.fail = None

    def run_realtime(self, *, entry, request, action):
        self.calls.append((action, request))
        if self.fail == action:
            raise ValueError(f"{action} failed")
        return {"video": np.zeros((2, 4, 5, 3), dtype=np.uint8)}


def runtime(tmp_path, *, queued=False, postprocess=None):
    entry = find_entry("longvie-2" if queued else "infinite-world")
    manager = Manager()
    result = ResidentWorldRuntime(manager=manager, entry=entry,
                                  launch_config=StudioLaunchConfig(model_id=entry.model_id, frontend="world"),
                                  fps=16, postprocess=postprocess)
    base = request(tmp_path)
    result._build_request = lambda **values: replace(base, prompt=values["prompt"])
    return result, manager, base


def test_queued_requests_forward_each_seed_without_changing_the_base_request(tmp_path):
    resident, manager, base = runtime(tmp_path, queued=True)
    resident._base_request, resident._configured = base, True

    async def exercise():
        try:
            await resident.generate([], seed=99)
            await resident.generate([], seed=100, dense_video_path="new-depth.mp4", sparse_video_path="new-track.mp4")
            assert [item.call_kwargs["seed"] for action, item in manager.calls if action in {"run", "stream"}] == [99, 100]
            assert base.call_kwargs["seed"] == 0
        finally:
            await resident.close()

    asyncio.run(exercise())


def test_postprocess_reset_failure_still_resets_model_and_invalidates_session(tmp_path):
    class Postprocess:
        def reset(self):
            raise ValueError("postprocess reset failed")

    resident, manager, base = runtime(tmp_path, postprocess=Postprocess())
    resident._base_request, resident._configured = base, True

    async def exercise():
        try:
            with pytest.raises(ValueError, match="postprocess reset failed"):
                await resident.reset()
            assert not resident._configured and resident._base_request is None
            assert [action for action, _ in manager.calls] == ["reset"]
        finally:
            resident._executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(exercise())


def test_failed_model_reset_blocks_reuse_until_cleanup_succeeds(tmp_path):
    resident, manager, base = runtime(tmp_path)
    resident._base_request, resident._configured = base, True
    manager.fail = "reset"

    async def exercise():
        try:
            with pytest.raises(ValueError, match="reset failed"):
                await resident.reset()
            with pytest.raises(RuntimeError, match="not configured"):
                await resident.generate([], seed=43)
            with pytest.raises(ValueError, match="reset failed"):
                await resident.configure(prompt="B", image_path="", video_path="")
            manager.fail = None
            await resident.configure(prompt="A", image_path="", video_path="")
            await resident.generate([], seed=43)
        finally:
            manager.fail = None
            await resident.close()

    asyncio.run(exercise())


def test_failed_generation_requires_a_new_configuration(tmp_path):
    resident, manager, base = runtime(tmp_path)
    resident._base_request, resident._configured = base, True
    manager.fail = "stream"

    async def exercise():
        try:
            with pytest.raises(ValueError, match="stream failed"):
                await resident.generate([], seed=43)
            manager.fail = None
            with pytest.raises(RuntimeError, match="not configured"):
                await resident.generate([], seed=43)
            await resident.configure(prompt="A", image_path="", video_path="")
            assert any(action == "reset" for action, _ in manager.calls)
        finally:
            await resident.close()

    asyncio.run(exercise())


def test_cancelled_generation_cannot_commit_into_the_next_session(tmp_path):
    resident, manager, base = runtime(tmp_path)
    resident._base_request, resident._configured = base, True
    started, release = threading.Event(), threading.Event()
    original = manager.run_realtime

    def blocking(**kwargs):
        if kwargs["action"] == "stream":
            started.set()
            assert release.wait(5)
        return original(**kwargs)

    manager.run_realtime = blocking

    async def exercise():
        try:
            pending = asyncio.create_task(resident.generate([], seed=43))
            while not started.is_set():
                await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            release.set()
            await resident.configure(prompt="B", image_path="", video_path="")
            assert [action for action, _ in manager.calls].index("reset") < [action for action, _ in manager.calls].index("configure")
            assert resident._base_request.prompt == "B" and resident._first_stream_step
        finally:
            release.set()
            await resident.close()

    asyncio.run(exercise())


def test_closed_resident_runtime_rejects_configuration_and_preload(tmp_path):
    resident, _, _ = runtime(tmp_path)

    async def exercise():
        await resident.preload()
        await resident.close()
        assert not resident.ready
        with pytest.raises(RuntimeError, match="closed"):
            await resident.preload()
        with pytest.raises(RuntimeError, match="closed"):
            await resident.configure(prompt="B", image_path="", video_path="")

    asyncio.run(exercise())


def test_timed_out_reset_keeps_new_configuration_waiting_for_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_SHUTDOWN_TIMEOUT_SECONDS", "0.01")
    resident, manager, base = runtime(tmp_path)
    resident._base_request, resident._configured = base, True
    started, release = threading.Event(), threading.Event()
    original = manager.run_realtime

    def blocking(**kwargs):
        if kwargs["action"] == "reset":
            started.set()
            assert release.wait(5)
        return original(**kwargs)

    manager.run_realtime = blocking

    async def exercise():
        pending = None
        try:
            await resident.reset()
            assert started.is_set() and not resident._configured
            pending = asyncio.create_task(resident.configure(prompt="B", image_path="", video_path=""))
            await asyncio.sleep(0.02)
            assert not pending.done()
            assert not any(action == "configure" for action, _ in manager.calls)
            release.set()
            await asyncio.wait_for(pending, 1)
            assert [action for action, _ in manager.calls][0] == "reset"
            assert resident._base_request.prompt == "B"
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await resident.close()

    asyncio.run(exercise())


def test_warmup_flushes_tail_records_stage_times_and_resets_both_states(tmp_path):
    from worldfoundry.core.media.processing.postprocess import (
        VideoPostprocessChain,
        VideoPostProcessor,
        VideoPostProcessorSession,
        VideoPostprocessStream,
    )

    events = []

    class TailSession(VideoPostProcessorSession):
        def process(self, chunk):
            return []

        def flush(self):
            events.append("flush")
            return []

        def close(self):
            events.append("close")

    class TailProcessor(VideoPostProcessor):
        def start(self, spec):
            events.append("start")
            return TailSession()

    resident, manager, _ = runtime(tmp_path, postprocess=VideoPostprocessStream(
        chain=VideoPostprocessChain((TailProcessor(),)), fps=16,
    ))
    seed = tmp_path / "warmup.png"
    Image.new("RGB", (5, 4)).save(seed)
    resident.warmup_image_path, resident.warmup_chunks = str(seed), 2

    async def exercise():
        try:
            await resident._warmup()
            assert "flush" in events
            assert [action for action, _ in manager.calls] == ["configure", "stream", "stream", "reset"]
            assert [req.call_kwargs["seed"] for action, req in manager.calls if action == "stream"] == [41000, 41001]
            assert resident._postprocess.input_spec is None
            assert resident._postprocess.chunk_index == 0 and resident._base_request is None
            assert len(resident.warmup_metrics["steps"]) == 2
            assert all(value >= 0 for step in resident.warmup_metrics["steps"] for value in step.values())
            assert resident.warmup_metrics["flush_ms"] >= 0
        finally:
            await resident.close()

    asyncio.run(exercise())


def test_warmup_retains_primary_failure_when_reset_also_fails(tmp_path):
    resident, manager, _ = runtime(tmp_path)
    seed = tmp_path / "warmup.png"
    Image.new("RGB", (5, 4)).save(seed)
    resident.warmup_image_path, resident.warmup_chunks = str(seed), 2
    original = manager.run_realtime

    def failing(**kwargs):
        if kwargs["action"] in {"stream", "reset"}:
            raise ValueError(kwargs["action"] + " failed")
        return original(**kwargs)

    manager.run_realtime = failing

    async def exercise():
        try:
            with pytest.raises(ValueError, match="stream failed") as caught:
                await resident._warmup()
            if hasattr(BaseException, "add_note"):
                assert any("reset failed" in note for note in caught.value.__notes__)
            assert not resident._configured and resident._reset_request is not None
        finally:
            manager.run_realtime = original
            await resident.close()

    asyncio.run(exercise())


def test_secondary_reset_error_does_not_mask_the_primary_without_exception_notes(tmp_path):
    class LegacyError(ValueError):
        add_note = None

    class Postprocess:
        def reset(self):
            raise LegacyError("primary reset failed")

    resident, manager, base = runtime(tmp_path, postprocess=Postprocess())
    resident._base_request, resident._configured = base, True
    manager.fail = "reset"

    async def exercise():
        try:
            with pytest.raises(LegacyError, match="primary reset failed"):
                await resident.reset()
            assert [action for action, _ in manager.calls] == ["reset"]
            assert not resident._configured and resident._reset_request is base
        finally:
            resident._executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(exercise())
