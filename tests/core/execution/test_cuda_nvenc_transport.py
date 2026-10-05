"""Real CUDA pixels and ownership contracts; no native encoder is simulated here."""

from __future__ import annotations

import asyncio
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldfoundry.core.execution.realtime.frame_prefetch import LazyCudaFrame
from worldfoundry.core.media.processing.postprocess import (
    IdentityVideoPostProcessor,
    VideoPostprocessChain,
    VideoPostprocessStream,
)
from worldfoundry.studio.serving.realtime.media import (
    ChunkPresentationBuffer,
    FrameQueuePolicy,
    LatestFrameBuffer,
    realtime_frames_from_result,
)
from worldfoundry.studio.serving.realtime.nvenc import _rgb_frame_to_abgr_cuda
from worldfoundry.studio.visualization.backends.world_realtime import ResidentWorldRuntime

pytestmark = pytest.mark.gpu


def _require_cuda():
    assert torch.cuda.is_available(), "CUDA transport contracts require an actual CUDA device"
    return torch.device("cuda:0")


@pytest.mark.parametrize("queue_kind", ["latest", "presentation"])
def test_device_transport_waits_for_producer_owns_pixels_and_avoids_host_materialization(monkeypatch, queue_kind):
    device = _require_cuda()
    reference = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    producer = torch.cuda.Stream(device=device)
    # Exercise a noncontiguous HWC decoder view, including distinct RGB bytes.
    source = torch.empty((2, 3, 4, 5), dtype=torch.uint8, device=device).permute(0, 2, 3, 1)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(5_000_000)
        source.copy_(torch.from_numpy(reference).to(device))
        ready = torch.cuda.Event()
        ready.record(producer)
    lazy = [LazyCudaFrame(source, index, source_event=ready) for index in range(2)]

    def reject_host(self):
        raise AssertionError("NVENC transport unexpectedly materialized CPU pixels")

    monkeypatch.setattr(LazyCudaFrame, "to_numpy", reject_host)
    frames = realtime_frames_from_result({"frames": lazy}, preserve_cuda=True)
    assert frames[0].shape == (4, 5, 3)
    assert str(frames[0].dtype) == "uint8"
    # Original decoder storage can be reused after its ordered snapshot.
    source.fill_(0)
    runtime = object.__new__(ResidentWorldRuntime)
    runtime.preserve_cuda_frames = True
    runtime._postprocess = VideoPostprocessStream(chain=VideoPostprocessChain((IdentityVideoPostProcessor(),)), fps=16)
    assert runtime._cuda_transport_enabled
    frames = runtime._postprocess_frames(frames)
    assert runtime._postprocess.chunk_index == 1
    assert runtime._postprocess.input_spec.dtype == "uint8"

    async def exercise():
        buffer = (
            LatestFrameBuffer(maxsize=2, policy=FrameQueuePolicy.ORDERED_QUALITY)
            if queue_kind == "latest" else
            ChunkPresentationBuffer(fps=1000, maximum_fps=1000, policy=FrameQueuePolicy.ORDERED_QUALITY)
        )
        buffer.preserve_cuda_frames = True
        assert await buffer.put_chunk(frames) == 2
        for index in range(2):
            queued = await asyncio.wait_for(buffer.get(), 1)
            assert isinstance(queued, LazyCudaFrame)
            rgba = _rgb_frame_to_abgr_cuda(queued, torch)
            assert rgba.is_cuda and rgba.is_contiguous()
            actual = rgba.cpu().numpy()
            np.testing.assert_array_equal(actual[..., :3], reference[index])
            np.testing.assert_array_equal(actual[..., 3], np.full((4, 5), 255, np.uint8))
        buffer.close()
        with pytest.raises(EOFError):
            await buffer.get()

    asyncio.run(exercise())


def test_raw_device_result_snapshot_survives_decoder_reuse_and_latest_drop():
    device = _require_cuda()
    source = torch.full((3, 2, 3, 3), 23, dtype=torch.uint8, device=device)
    frames = realtime_frames_from_result({"videos": source}, preserve_cuda=True)
    source.fill_(99)
    dropped_refs = [weakref.ref(frame) for frame in frames[:2]]

    async def exercise():
        buffer = LatestFrameBuffer(maxsize=1)
        buffer.preserve_cuda_frames = True
        await buffer.put_chunk(frames)
        frames.clear()
        assert buffer.dropped_frames == 2
        assert all(reference() is None for reference in dropped_refs)
        frame = await buffer.get()
        np.testing.assert_array_equal(_rgb_frame_to_abgr_cuda(frame, torch).cpu().numpy()[..., :3], np.full((2, 3, 3), 23, np.uint8))
        await buffer.put_chunk([frame])
        held = weakref.ref(frame)
        del frame
        assert held() is not None
        buffer.close()
        assert held() is None

    asyncio.run(exercise())


def test_device_transport_declines_custom_postprocess_and_rejects_wrong_encoder_device():
    device = _require_cuda()
    runtime = object.__new__(ResidentWorldRuntime)
    runtime.preserve_cuda_frames = True
    runtime._postprocess = SimpleNamespace(chain=VideoPostprocessChain((IdentityVideoPostProcessor(),)))
    assert not runtime._cuda_transport_enabled
    source = torch.zeros((2, 2, 3), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="does not match NVENC device"):
        _rgb_frame_to_abgr_cuda(source, torch, gpu_id=1)
