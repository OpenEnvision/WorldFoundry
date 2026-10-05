"""Real CUDA event timing on a non-current device and a side stream."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.core.media.processing.postprocess import (
    VideoPostprocessChain,
    VideoPostProcessor,
    VideoPostProcessorSession,
    VideoPostprocessStream,
)

pytestmark = pytest.mark.gpu


class _CudaProcessor(VideoPostProcessor):
    name = "cuda-arithmetic-validation"
    processing_device = "cuda:1"

    def start(self, spec):
        return _CudaSession()


class _CudaSession(VideoPostProcessorSession):
    def flush(self):
        return []

    def process(self, chunk):
        for _ in range(8):
            chunk.frames = chunk.frames * 1.001 + 0.01
        return [chunk]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_profile_times_actual_work_on_tensor_device_and_side_stream():
    previous_device = torch.cuda.current_device()
    previous_stream = torch.cuda.current_stream(1)
    side = torch.cuda.Stream(device=1)
    try:
        torch.cuda.set_stream(side)
        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(1) == side
        frames = torch.ones(1, 3, 4, 64, 64, device="cuda:1")
        expected = frames
        for _ in range(8):
            expected = expected * 1.001 + 0.01
        stream = VideoPostprocessStream(chain=VideoPostprocessChain((_CudaProcessor(),)), fps=16, profile_cuda=True)
        [output] = stream.process(frames, layout="bcthw")
        torch.testing.assert_close(output.frames, expected, rtol=0, atol=0)
        assert stream.last_stats.gpu_elapsed_ms > 0
        assert torch.cuda.current_device() == 0
        stream.finish()
    finally:
        torch.cuda.set_stream(previous_stream)
        torch.cuda.set_device(previous_device)
