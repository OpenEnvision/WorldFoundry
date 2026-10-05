"""Actual CUDA frames through the resident generation and presentation boundary."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import torch

from worldfoundry.core.execution.realtime.frame_prefetch import LazyCudaFrame
from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.execution import StudioManager
from worldfoundry.studio.serving.realtime.media import (
    ChunkPresentationBuffer,
    FrameQueuePolicy,
    LatestFrameBuffer,
)
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.backends import world_realtime as backend

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("queue_kind", ["latest", "presentation"])
@pytest.mark.parametrize("resolution_mode", ["native", "matching", "oversize"])
def test_resident_generation_preserves_cuda_identity_and_observes_resolution(
    monkeypatch, tmp_path, queue_kind, resolution_mode,
):
    assert torch.cuda.is_available(), "NVENC generation contracts require an actual CUDA device"
    device = torch.device("cuda:0")
    height, width = 96, 160
    reference = np.arange(2 * height * width * 3, dtype=np.uint8).reshape(2, height, width, 3)
    source = torch.from_numpy(reference).to(device)
    extracted: list[LazyCudaFrame] = []
    original_extract = backend.realtime_frames_from_result

    def record_extract(result, *, preserve_cuda=False):
        assert preserve_cuda, "identity postprocess must retain the selected device transport"
        frames = original_extract(result, preserve_cuda=preserve_cuda)
        extracted.extend(frames)
        return frames

    def reject_host(self):
        raise AssertionError("the generation/presentation boundary materialized host pixels")

    monkeypatch.setattr(backend, "realtime_frames_from_result", record_extract)
    monkeypatch.setattr(LazyCudaFrame, "to_numpy", reject_host)

    class ModelManager(StudioManager):
        def __init__(self):
            super().__init__(workspace_root=str(tmp_path))
            self.actions = []

        def run_realtime(self, *, entry, request, action):
            self.actions.append(action)
            if action == "reset":
                return {}
            assert action == "stream"
            return {"videos": source}

    class Channel:
        readyState = "open"

        def __init__(self):
            self.messages = []
            self.active = None

        def send(self, raw):
            payload = json.loads(raw)
            self.messages.append(payload)
            if payload["type"] == "chunk_done":
                self.active.closed = True

    async def exercise():
        model_manager = ModelManager()
        runtime = backend.ResidentWorldRuntime(
            manager=model_manager,
            entry=find_entry("sana-wm"),
            launch_config=StudioLaunchConfig(model_id="sana-wm", frontend="world", device="cuda:0"),
            fps=16,
        )
        runtime._base_request = runtime._build_request(
            prompt="CUDA contract", image=None, input_path="", video_path="",
        )
        runtime._configured = True
        runtime.preserve_cuda_frames = True
        buffer = (
            LatestFrameBuffer(maxsize=2, policy=FrameQueuePolicy.ORDERED_QUALITY)
            if queue_kind == "latest" else
            ChunkPresentationBuffer(fps=1000, maximum_fps=1000, policy=FrameQueuePolicy.ORDERED_QUALITY)
        )
        buffer.preserve_cuda_frames = True
        requested = {
            "native": {"mode": "native"},
            "matching": {"width": width, "height": height},
            "oversize": {"width": width * 2, "height": height * 2},
        }[resolution_mode]
        resolution = backend.OutputResolutionState.from_value(requested)
        channel = Channel()
        active = backend._ActivePeer(
            peer=None,
            channel=channel,
            frames=buffer,
            resampler=backend.RealtimeControlResampler(fps=16, start_time=0.0),
            pending_steps=1,
            output_resolution=resolution,
        )
        channel.active = active
        peers = backend.RealtimePeerManager(runtime=runtime, fps=16, chunk_frames=2)
        peers._active = active
        active.first_action.set()
        try:
            await asyncio.wait_for(peers._generation_worker(active), timeout=5)
            assert not [message for message in channel.messages if message["type"] == "error"]
            chunks = [message for message in channel.messages if message["type"] == "chunk_done"]
            assert len(chunks) == 1
            assert chunks[0]["frames"] == 2
            assert chunks[0]["resolution"]["width"] == width
            assert chunks[0]["resolution"]["height"] == height
            assert chunks[0]["resolution_revision"] == int(resolution_mode == "oversize")
            assert resolution.source_dimensions == (width, height)
            assert resolution.dimensions == ((width, height) if resolution_mode == "matching" else None)
            assert model_manager.actions == ["stream"]
            assert runtime._cuda_transport_enabled
            assert runtime._postprocess.processor_names == ("identity",)
            assert runtime._postprocess.chunk_index == 1
            assert runtime._postprocess.input_spec.dtype == "uint8"
            assert len(extracted) == 2
            # The resident identity processor and real queue retain the exact
            # owned frames produced by the production result extractor.
            source.fill_(0)
            for index in range(2):
                queued = await asyncio.wait_for(buffer.get(), timeout=1)
                assert queued is extracted[index]
                assert queued.ndim == 3
                assert queued.shape == (height, width, 3)
                event = queued.to_cuda_event()
                assert event is not None
                event.synchronize()
                pixels = queued.to_cuda_tensor()
                assert pixels.is_cuda and pixels.dtype == torch.uint8
                np.testing.assert_array_equal(pixels.cpu().numpy(), reference[index])
        finally:
            buffer.close()
            await runtime.close()

    asyncio.run(exercise())
