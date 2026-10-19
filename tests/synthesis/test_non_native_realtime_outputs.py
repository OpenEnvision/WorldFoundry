"""Real session orchestration with small CPU tensors and substituted model/decoder calls."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.core.media.codecs.video import video_chunk_to_uint8_frames
from worldfoundry.studio.serving.realtime.media import realtime_frames_from_result
from worldfoundry.synthesis.visual_generation.dreamx_world.realtime import DreamXWorldRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import (
    LingBotWorldV2RealtimeSession,
    LingBotWorldV2SmallRealtimeSession,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rank", [0, 1])
def test_dreamx_camera_chunk_describes_only_new_rgb_frames(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    source = np.arange(5 * 3 * 4 * 3, dtype=np.uint8).reshape(5, 3, 4, 3)
    decodes: list[int] = []
    broadcasts: list[Any] = []

    class Pipe:
        def __call__(self, **kwargs: Any) -> Any:
            assert kwargs["output_type"] == "latent"
            return SimpleNamespace(videos=object())

        def decode_latents(self, latent: Any) -> np.ndarray:
            assert rank == 0
            decodes.append(rank)
            return source.transpose(3, 0, 1, 2)[None]

    session = object.__new__(DreamXWorldRealtimeSession)
    session.pipe = Pipe()
    session.rank = rank
    session.world_size = 2
    session.device = torch.device("cpu")
    session.checkpoint = Path("/models/dreamx")
    session.wan_model_path = Path("/models/wan")
    session.model_frames = 5
    session.fps = 16
    session.height = session.width = 64
    session.num_inference_steps = 1
    session.guidance_scale = 1.0
    session.boundary = 0.9
    session._configured = True
    session._prewarmed = False
    session._image = Image.new("RGB", (64, 64))
    session._prompt = "scene"
    session._seed = 7
    session._chunk_index = 0
    monkeypatch.setattr(session, "_encode_prompt", lambda prompt: ([torch.zeros(1)], [torch.zeros(1)]))
    monkeypatch.setattr(session, "_camera_condition", lambda *args: {})

    def broadcast(frame: np.ndarray | None) -> Image.Image:
        broadcasts.append(frame)
        return Image.fromarray(source[-1])

    monkeypatch.setattr(session, "_broadcast_last_frame", broadcast)
    result = session.generate(interactions=["forward"])
    assert session._chunk_index == 1
    assert result["realtime_spec"]["first_chunk_frames"] == 4
    assert session._image.getpixel((0, 0)) == tuple(source[-1, 0, 0].tolist())
    if rank == 0:
        chunk = result["video_chunk"]
        assert chunk.frames is result["frames"] is result["video"]
        assert (chunk.layout, chunk.value_range, chunk.frame_count) == ("thwc", "0,255", 4)
        np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), source[1:])
        np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), source[1:])
        np.testing.assert_array_equal(broadcasts[0], source[-1])
        assert decodes == [0]
    else:
        assert result["frames"] is result["video"] is result["video_chunk"] is None
        with pytest.raises(RuntimeError, match="no in-memory RGB frames"):
            realtime_frames_from_result(result)
        assert decodes == []
        assert broadcasts == [None]


@pytest.mark.parametrize(
    "session_cls", [LingBotRealtimeSession, LingBotWorldV2RealtimeSession, LingBotWorldV2SmallRealtimeSession]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_lingbot_inherited_output_contract_keeps_rank_and_cadence(
    monkeypatch: pytest.MonkeyPatch, session_cls: type[LingBotRealtimeSession], rank: int
) -> None:
    class Event:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def record(self) -> None:
            pass

        def elapsed_time(self, other: Any) -> float:
            return 0.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    class Model:
        def __call__(self, *, x: list[torch.Tensor], **kwargs: Any) -> list[torch.Tensor]:
            return [torch.zeros_like(x[0])]

    session = object.__new__(session_cls)
    session.core = SimpleNamespace(model=Model())
    session.device = torch.device("cpu")
    session.rank = rank
    session.sp_size = 2
    session.fps = 12
    session.param_dtype = torch.float32
    session._state_lock = threading.RLock()
    session._state_ready_event = None
    session.configured = True
    session._context = [torch.zeros(1)]
    session._vae_state = {}
    session._self_cache = []
    session._cross_cache = []
    session._generator = torch.Generator().manual_seed(7)
    session._timesteps = torch.tensor([1000.0])
    session._sigmas = torch.tensor([1.0])
    session._text_condition = None
    session._condition_cache = None
    session._cross_attention_initialized = False
    session._pending_first_condition = torch.zeros(1)
    session._prefetched_conditions = [torch.zeros(1)]
    session.autoregressive_index = 0
    session.chunk_sequence_length = 3
    session.max_sequence_length = 18
    session.cache_sequence_length = 18
    session.frame_sequence_length = 1
    monkeypatch.setattr(session, "_camera_condition", lambda *args: torch.zeros(1))
    monkeypatch.setattr(session, "_draw_initial_noise", lambda: torch.zeros(16, 3, 2, 2))
    monkeypatch.setattr(session, "_prepare_rotary_frequencies", lambda *args, **kwargs: None)
    decoded: list[np.ndarray] = []

    def decode(latent: torch.Tensor) -> np.ndarray:
        assert rank == 0
        frames = np.full((session.next_output_frames(), 3, 4, 3), 191, dtype=np.uint8)
        decoded.append(frames)
        return frames

    monkeypatch.setattr(session, "_decode", decode)
    for count in (session.first_output_frames, session.steady_output_frames):
        result = session.generate(interactions=["forward"])
        if rank == 0:
            assert result is not None
            chunk = result["video_chunk"]
            assert chunk.frames is result["video"] is decoded[-1]
            assert (chunk.layout, chunk.value_range, chunk.frame_count) == ("thwc", "0,255", count)
            np.testing.assert_array_equal(video_chunk_to_uint8_frames(chunk), decoded[-1])
            np.testing.assert_array_equal(np.stack(realtime_frames_from_result(result)), decoded[-1])
        else:
            assert result is None
            assert decoded == []
    assert session.autoregressive_index == 2
