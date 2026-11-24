# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""LingBot pixel conversion, producer ordering and deferred CPU/CUDA transport."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from worldfoundry.core.execution.realtime.frame_prefetch import CudaHostPrefetch, LazyCudaFrame
from worldfoundry.core.media.processing.postprocess import VideoChunk
from worldfoundry.pipelines.lingbot_world_v2.pipeline_lingbot_world_v2 import LingBotWorldV2Pipeline
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import (
    LingBotWorldV2RealtimeSession,
    LingBotWorldV2SmallRealtimeSession,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("session_type", [LingBotWorldV2RealtimeSession, LingBotWorldV2SmallRealtimeSession])
@pytest.mark.parametrize("chunk_index", [0, 1])
@pytest.mark.parametrize("preserve_cuda", [False, True])
@torch.no_grad()
def test_deferred_decode_preserves_pixels_order_and_storage(
    monkeypatch: pytest.MonkeyPatch,
    session_type: type[LingBotRealtimeSession],
    chunk_index: int,
    preserve_cuda: bool,
) -> None:
    assert torch.cuda.is_available()
    session = object.__new__(session_type)
    session.rank = 0
    session.autoregressive_index = chunk_index
    session._decoder_needs_scaling = False
    count = session.next_output_frames()
    small = session_type is LingBotWorldV2SmallRealtimeSession
    source = torch.linspace(-1 if small else 0, 1, count * 3 * 4 * 5).reshape(count, 3, 4, 5)
    normalized = source.permute(0, 2, 3, 1).clone()
    expected = (
        (normalized.add_(1).mul_(127.5) if small else normalized.mul_(255))
        .round_()
        .clamp_(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    storage = source.permute(1, 0, 2, 3).unsqueeze(0) if small else source.unsqueeze(0)
    storage = storage.to("cuda")
    session._decoder = cast(Any, SimpleNamespace(decode=lambda z=None: storage if z is not None else None))
    latent = torch.zeros(16, session.chunk_latent_frames, 1, 1, device="cuda")
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    copies: list[CudaHostPrefetch] = []
    start = CudaHostPrefetch.start

    def record_copy(copy: CudaHostPrefetch) -> bool:
        assert start(copy), "CPU transport must execute the asynchronous pinned-memory copy"
        assert copy._host_tensor is not None and copy._host_tensor.is_pinned()
        copies.append(copy)
        return True

    def reject_cpu(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("deferred decode/transport used a blocking CPU copy or device-wide barrier")

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "cpu", reject_cpu)
        scoped.setattr(torch.cuda, "synchronize", reject_cpu)
        scoped.setattr(CudaHostPrefetch, "start", record_copy)
        if preserve_cuda:
            scoped.setattr(LazyCudaFrame, "to_numpy", reject_cpu)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(2_000_000)
            frames = session._decode_for_transport(latent)
            assert isinstance(frames, list)
            # Reusing the decoder's own storage must not change the uint8 output.
            storage.zero_()
        result = {"video_chunk": VideoChunk(frames=frames, layout="frame-list", value_range="0,255")}
        transported = realtime_frames_from_result(result, preserve_cuda=preserve_cuda)
    if preserve_cuda:
        assert not copies
        assert all(isinstance(frame, LazyCudaFrame) for frame in transported)
        actual = np.stack([frame.to_numpy() for frame in transported])
    else:
        assert len(copies) == count
        actual = np.stack(transported)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rank", [0, 1])
@torch.no_grad()
def test_v2_public_realtime_defers_frames_and_uses_only_decode_event(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
) -> None:
    assert torch.cuda.is_available()

    class Model:
        def __call__(self, *, x: list[torch.Tensor], **kwargs: Any) -> list[torch.Tensor]:
            return [torch.zeros_like(x[0])]

    session = object.__new__(LingBotWorldV2SmallRealtimeSession)
    session.core = SimpleNamespace(model=Model())
    session.device = torch.device("cuda:0")
    session.rank, session.sp_size, session.fps = rank, 1, 16
    session.param_dtype = torch.float32
    session._state_lock = threading.RLock()
    session._state_ready_event = None
    session.configured = True
    session._context = [torch.zeros(1, device=session.device)]
    session._vae_state = {}
    session._self_cache, session._cross_cache = [], []
    session._generator = torch.Generator(device=session.device).manual_seed(7)
    session._timesteps = torch.tensor([1000.0], device=session.device)
    session._sigmas = torch.tensor([1.0], device=session.device)
    session._text_condition = session._condition_cache = None
    session._cross_attention_initialized = False
    session._decoder = None
    session.camera = cast(Any, SimpleNamespace(reset=lambda: None))
    session._pending_first_condition = torch.zeros(1, device=session.device)
    session._prefetched_conditions = [torch.zeros(1, device=session.device)]
    session.autoregressive_index = 0
    session.chunk_sequence_length, session.max_sequence_length, session.cache_sequence_length = 4, 18, 18
    session.frame_sequence_length = 1
    monkeypatch.setattr(session, "_camera_condition", lambda *args: torch.zeros(1, device=session.device))
    monkeypatch.setattr(session, "_draw_initial_noise", lambda: torch.zeros(16, 4, 2, 2, device=session.device))
    monkeypatch.setattr(session, "_prepare_rotary_frequencies", lambda *args, **kwargs: None)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        assert rank == 0
        return torch.full(
            (session.next_output_frames(), 3, 4, 3),
            51 + session.autoregressive_index,
            dtype=torch.uint8,
            device=session.device,
        )

    def reject_sync(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("realtime generation must not synchronize unrelated device work or copy pixels")

    monkeypatch.setattr(session, "_decode_pixels", decode)
    pipeline = object.__new__(LingBotWorldV2Pipeline)
    pipeline._realtime_session = session
    results = []
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.cuda, "synchronize", reject_sync)
        scoped.setattr(torch.Tensor, "cpu", reject_sync)
        for _ in range(2):
            results.append(pipeline.stream_realtime(interactions=["forward"]))
    assert session.autoregressive_index == 2
    assert session._state_ready_event is not None
    session.reset()
    assert not session.configured and session.autoregressive_index == 0
    for index, result in enumerate(results):
        if rank != 0:
            assert result is None
            continue
        assert result is not None
        chunk = result["video_chunk"]
        count = 13 if index == 0 else 16
        assert chunk.frames is result["video"]
        assert chunk.layout == "frame-list" and chunk.frame_count == count
        assert result["realtime_metrics"]["decode_ms"] >= 0
        pixels = np.stack(realtime_frames_from_result(result))
        np.testing.assert_array_equal(pixels, np.full((count, 3, 4, 3), 51 + index, np.uint8))
