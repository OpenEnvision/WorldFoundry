# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Settled I2V conditions retain original encode geometry and request isolation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.conditioning import (
    WanZeroConditionCache,
    wan21_encoder_history_frames,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming import (
    Encoder3d,
    WanVAE_,
)
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import (
    LingBotWorldV2RealtimeSession,
    LingBotWorldV2SmallRealtimeSession,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("residual_blocks,history", [(1, 80), (2, 112)])
def test_encoder_history_follows_temporal_convolutions(residual_blocks: int, history: int) -> None:
    encoder = Encoder3d(dim=4, z_dim=8, num_res_blocks=residual_blocks, temperal_downsample=[False, True, True])
    assert wan21_encoder_history_frames(encoder) == history
    with pytest.raises(TypeError, match="canonical"):
        wan21_encoder_history_frames(torch.nn.Conv3d(3, 4, 3))
    encoder = Encoder3d(dim=4, z_dim=8, temperal_downsample=[False, False, True])
    with pytest.raises(ValueError, match="four-frame"):
        wan21_encoder_history_frames(encoder)


def test_cache_requires_inference_and_retains_whole_group_boundaries() -> None:
    encoder = Encoder3d(dim=4, z_dim=8, temperal_downsample=[False, True, True])
    cache = WanZeroConditionCache(encoder)
    value = torch.randn(20, 16, 2, 2)
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval no-grad"):
        cache.record(value, 13, first=True)
    encoder.eval()
    with pytest.raises(RuntimeError, match="eval no-grad"):
        cache.get(64)
    with torch.no_grad():
        cache.record(value, 13, first=True)
        cache.record(value, 64, first=False)
        cache.record(value, 64, first=False)
        assert cache.encoded_zero_frames == 140
        # The group crossed the history boundary but still contained a transient.
        assert cache.get(64) is None
        cache.record(value, 64, first=False)
        assert cache.get(64) is value
        assert cache.get(16) is None
        assert cache.nbytes == value.numel() * value.element_size()
        cache.record(value, 13, first=True)
        assert cache.get(64) is None and cache.nbytes == cache.reuse_calls == 0


class _TinyWan:
    def __init__(self, device: str) -> None:
        self.model = (
            WanVAE_(
                dim=4,
                z_dim=4,
                dim_mult=[1, 2, 2, 2],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False, True, True],
            )
            .eval()
            .to(device)
        )
        self.scale = [torch.zeros(4, device=device), torch.ones(4, device=device)]
        self.calls = 0

    def prepare_streaming_encode(self) -> dict[str, Any]:
        return self.model.prepare_streaming_encode()

    def encode_streaming(self, video: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        self.calls += 1
        return self.model.encode_streaming(video.unsqueeze(0), self.scale, state).float().squeeze(0)


def _session(
    monkeypatch: pytest.MonkeyPatch,
    vae: _TinyWan,
    session_type: type[LingBotRealtimeSession] = LingBotWorldV2SmallRealtimeSession,
    *,
    sp_size: int = 1,
) -> LingBotRealtimeSession:
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_REALTIME_HEIGHT", "16")
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_REALTIME_WIDTH", "16")
    monkeypatch.setattr(session_type, "_load_decoder", lambda _self: None)
    monkeypatch.setattr(session_type, "_compile_model_if_enabled", lambda _self: None)
    scheduler = SimpleNamespace(
        timesteps=torch.arange(1000),
        sigmas=torch.linspace(1, 0, 1000),
        set_timesteps=lambda *_args, **_kwargs: None,
    )
    core = SimpleNamespace(
        model=SimpleNamespace(blocks=[]),
        vae=vae,
        text_encoder=None,
        scheduler=scheduler,
        config=None,
        device=vae.scale[0].device,
        rank=0,
        sp_size=sp_size,
        param_dtype=torch.float32,
        pipe_dtype=torch.float32,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
        num_train_timesteps=1000,
    )
    session = session_type(core)
    monkeypatch.setattr(session, "_encode_prompt", lambda _prompt: [torch.zeros(1, 4)])
    monkeypatch.setattr(session, "_prepare_camera_grid", lambda: None)
    monkeypatch.setattr(session, "_initialize_model_caches", lambda: None)
    return session


@pytest.mark.parametrize(
    "session_type,sp_size,enabled",
    [
        (LingBotRealtimeSession, 1, False),
        (LingBotWorldV2RealtimeSession, 1, False),
        (LingBotWorldV2SmallRealtimeSession, 1, True),
        (LingBotWorldV2SmallRealtimeSession, 2, False),
    ],
)
def test_reuse_admission(
    monkeypatch: pytest.MonkeyPatch,
    session_type: type[LingBotRealtimeSession],
    sp_size: int,
    enabled: bool,
) -> None:
    session = _session(monkeypatch, _TinyWan("cpu"), session_type, sp_size=sp_size)
    assert (session._new_condition_cache() is not None) is enabled


@pytest.mark.parametrize("prefetch", [0, 8])
@pytest.mark.parametrize("replenish", [1, 4])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@torch.no_grad()
def test_conditions_match_through_settling_prefetch_reset_and_changed_images(
    monkeypatch: pytest.MonkeyPatch,
    prefetch: int,
    replenish: int,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("WORLDFOUNDRY_REALTIME_CONDITION_PREFETCH_CHUNKS", str(prefetch))
    torch.manual_seed(23)
    vae = _TinyWan(device)
    reference, candidate = _session(monkeypatch, vae), _session(monkeypatch, vae)
    reference.condition_replenish_chunks = candidate.condition_replenish_chunks = replenish
    monkeypatch.setattr(reference, "_new_condition_cache", lambda: None)
    images = [Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)) for value in (50, 210, 50)]
    outputs: list[list[torch.Tensor]] = []
    old_cache = None
    for image in images:
        reference.configure(image=image, prompt="room")
        candidate.configure(image=image, prompt="room")
        assert candidate._condition_cache is not None and candidate._condition_cache is not old_cache
        assert reference._vae_state is not candidate._vae_state
        assert reference._pending_first_condition is not None
        assert candidate._pending_first_condition is not None
        assert torch.equal(reference._pending_first_condition, candidate._pending_first_condition)
        actual = [candidate._pending_first_condition.clone()]
        for _ in range(20):
            chunks = []
            for session in (reference, candidate):
                if not session._prefetched_conditions:
                    session._prefetch_steady_conditions(replenish)
                chunks.append(session._prefetched_conditions.pop(0))
            assert torch.isfinite(chunks[1]).all() and torch.count_nonzero(chunks[1]) > 0
            assert torch.equal(chunks[0], chunks[1])
            actual.append(chunks[1].clone())
        outputs.append(actual)
        old_cache = candidate._condition_cache
        assert old_cache.reuse_calls > 0 and old_cache.nbytes > 0
        calls = vae.calls
        candidate._prefetch_steady_conditions(replenish)
        assert vae.calls == calls
        candidate.reset()
        assert candidate._condition_cache is None and candidate._vae_state is None
        assert candidate._prefetched_conditions == []
    assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[2]))
    assert not torch.equal(outputs[0][0], outputs[1][0])
