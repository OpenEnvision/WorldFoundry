"""Real CPU/CUDA frames, producer events and failed asynchronous callbacks."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import torch

from worldfoundry.core.execution.realtime.frame_prefetch import CudaHostPrefetch, LazyCudaFrame
from worldfoundry.core.execution.realtime.overlap import CudaStreamOverlap


def _batch(device, noncontiguous):
    batch = torch.arange(2 * 4 * 6 * 3, dtype=torch.int64, device=device).to(torch.uint8).reshape(2, 4, 6, 3)
    return batch.transpose(1, 2) if noncontiguous else batch


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_cpu_fallback_materializes_once_and_releases_source(noncontiguous):
    batch = _batch("cpu", noncontiguous)
    expected = batch[1].numpy().copy()
    source = weakref.ref(batch)
    frame = LazyCudaFrame(batch, 1)
    frame.prefetch_to_numpy()
    actual = frame.to_numpy()
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert frame.to_numpy() is actual
    assert frame.to_cuda_event() is None
    with pytest.raises(RuntimeError, match="already materialized"):
        frame.to_cuda_tensor()
    del batch
    gc.collect()
    assert source() is None


def test_numpy_copy_contract_does_not_alias_an_explicit_copy():
    frame = LazyCudaFrame(_batch("cpu", False), 0)
    cached = frame.to_numpy()
    assert frame.__array__(dtype=np.uint8, copy=False) is cached
    copied = frame.__array__(copy=True)
    copied[:] = 0
    assert cached.any()
    converted = frame.__array__(dtype=np.float32)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, cached)
    with pytest.raises(ValueError, match="copy=False"):
        frame.__array__(dtype=np.float32, copy=False)


def test_declined_prefetch_is_safe_to_call_repeatedly():
    prefetch = CudaHostPrefetch(torch.ones(2, 3))
    assert not prefetch.start()
    assert prefetch.started
    assert not prefetch.start()
    with pytest.raises(RuntimeError, match="not started successfully"):
        prefetch.to_numpy()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_cuda_host_pixels_wait_for_producer_and_preserve_frame_order(noncontiguous, prefetch):
    producer = torch.cuda.Stream()
    batch = torch.empty((2, 4, 6, 3), dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(producer):
        producer.wait_stream(torch.cuda.current_stream())
        batch[0].fill_(17)
        batch[1].fill_(231)
        event = torch.cuda.Event()
        event.record(producer)
    if noncontiguous:
        batch = batch.transpose(1, 2)
    reference = weakref.ref(batch)
    frames = [LazyCudaFrame(batch, index, source_event=event) for index in range(2)]
    if prefetch:
        for frame in frames:
            frame.prefetch_to_numpy()
            assert frame._prefetch is not None, "real CUDA prefetch unexpectedly fell back"
    for frame, value in zip(frames, [17, 231]):
        actual = frame.to_numpy()
        assert actual.flags.c_contiguous
        assert actual.dtype == np.uint8
        assert actual.shape == (6, 4, 3) if noncontiguous else actual.shape == (4, 6, 3)
        np.testing.assert_array_equal(actual, np.full(actual.shape, value, dtype=np.uint8))
        assert frame.to_numpy() is actual
    del batch
    gc.collect()
    assert reference() is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_prefetch_failure_uses_the_correct_blocking_fallback(monkeypatch):
    batch = _batch("cuda", True)
    expected = batch[1].cpu().numpy().copy()
    frame = LazyCudaFrame(batch, 1)
    real_empty = torch.empty

    def decline_pinned_allocation(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("pinned-memory allocation unavailable")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", decline_pinned_allocation)
    frame.prefetch_to_numpy()
    assert frame._prefetch is None
    np.testing.assert_array_equal(frame.to_numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_failed_cuda_callback_drains_queued_work_before_reuse():
    runner = CudaStreamOverlap(device="cuda:0")
    finished = torch.cuda.Event()
    value = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    def fail_after_enqueue():
        torch.cuda._sleep(100_000_000)
        value.fill_(19)
        finished.record()
        raise ValueError("decode aborted after enqueue")

    try:
        with pytest.raises(ValueError, match="decode aborted"):
            runner.submit(fail_after_enqueue)
        assert finished.query(), "failure returned while CUDA still held request buffers"
        assert not runner.pending
        with pytest.raises(ValueError, match="decode aborted"):
            runner.wait(raise_error=True)
        runner.submit(lambda: value.add_(4))
        assert runner.wait(raise_error=True)
        assert value.item() == 23
        assert runner.last_error is None
    finally:
        runner.close(wait=False)
        torch.cuda.synchronize()
