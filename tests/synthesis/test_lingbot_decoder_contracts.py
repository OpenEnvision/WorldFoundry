# SPDX-FileCopyrightText: WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""LingBot decoder loading, numerical conventions and stream-local scratch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from worldfoundry.synthesis.visual_generation.lingbot_world import realtime
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import LingBotWorldV2RealtimeSession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("session_cls", [realtime.LingBotRealtimeSession, LingBotWorldV2RealtimeSession])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_decoder_converts_only_raw_checkpoint_inputs(
    session_cls: type[realtime.LingBotRealtimeSession], raw: bool, dtype: torch.dtype
) -> None:
    session = object.__new__(session_cls)
    session.rank = 0
    session.autoregressive_index = 0
    session._decoder_needs_scaling = raw
    mean = torch.linspace(-1.5, 1.5, 16)
    std = torch.linspace(0.25, 2.5, 16)
    session.core = SimpleNamespace(vae=SimpleNamespace(mean=mean, std=std)) if raw else SimpleNamespace()
    captured: list[torch.Tensor] = []

    class Decoder:
        def decode(self, z: torch.Tensor | None = None) -> torch.Tensor | None:
            if z is None:
                return None
            captured.append(z.clone())
            return torch.full((1, session.next_output_frames(), 3, 2, 3), 0.5)

    session._decoder = Decoder()  # ty: ignore[invalid-assignment]
    latent = torch.linspace(-2, 3, 16 * session.chunk_latent_frames * 6, dtype=dtype).reshape(
        16, session.chunk_latent_frames, 2, 3
    )
    original = latent.clone()
    expected = latent.permute(1, 0, 2, 3)[None].to(torch.bfloat16)
    if raw:
        expected = expected * std.to(torch.bfloat16)[None, None, :, None, None]
        expected = expected + mean.to(torch.bfloat16)[None, None, :, None, None]
    for index in (0, 1):
        session.autoregressive_index = index
        frames = session._decode(latent)
        torch.testing.assert_close(captured[-1], expected, rtol=0, atol=0)
        assert captured[-1].dtype == torch.bfloat16
        assert frames.shape == (session.next_output_frames(), 2, 3, 3)
        np.testing.assert_array_equal(frames, np.full_like(frames, 128))
    torch.testing.assert_close(latent, original, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["taew2_1.pth", "lighttaew2_1.pth", "renamed.safetensors"])
@pytest.mark.parametrize("convention,raw", [("raw", True), ("normalized", False)])
def test_explicit_convention_overrides_checkpoint_name(
    monkeypatch: pytest.MonkeyPatch, name: str, convention: str, raw: bool
) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", convention)
    assert realtime.LingBotRealtimeSession._decoder_uses_raw_latents(Path(name)) is raw


@pytest.mark.parametrize("convention", ["auto", "unit", "raw-latents"])
def test_invalid_convention_rejects_before_checkpoint_loading(monkeypatch: pytest.MonkeyPatch, convention: str) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", convention)
    session = object.__new__(realtime.LingBotRealtimeSession)
    session._decoder_model = None
    monkeypatch.setattr(session, "_decoder_checkpoint", lambda: Path("missing/lighttaew2_1.pth"))
    with pytest.raises(ValueError, match="must be raw or normalized"):
        session._load_decoder()
    assert session._decoder_model is None


@pytest.mark.parametrize("name", ["taew2_1.pth", "taew2_1.safetensors", "lighttaew2_1.pth", "lighttaew2_1.safetensors"])
def test_decoder_loader_reads_actual_checkpoint_and_freezes_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    class TinyTAE(torch.nn.Module):
        def __init__(self, *, checkpoint_path: None) -> None:
            super().__init__()
            assert checkpoint_path is None
            self.gain = torch.nn.Parameter(torch.zeros(1))

        def patch_tgrow_layers(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return state

    path = tmp_path / name
    state = {"gain": torch.tensor([1.25])}
    if path.suffix == ".safetensors":
        save_file(state, str(path))
    else:
        torch.save(state, path)
    monkeypatch.delenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", raising=False)
    monkeypatch.setattr(realtime, "TAEHV", TinyTAE)
    streams: list[Any] = []

    def stream(model: torch.nn.Module) -> Any:
        wrapper = SimpleNamespace(taehv=model)
        streams.append(wrapper)
        return wrapper

    monkeypatch.setattr(realtime, "StreamingTAEHV", stream)
    session = object.__new__(realtime.LingBotRealtimeSession)
    session._decoder_model = None
    session._decoder = None
    session.device = torch.device("cpu")
    monkeypatch.setattr(session, "_decoder_checkpoint", lambda: path)
    session._load_decoder()
    model = session._decoder_model
    assert model is not None and session._decoder is streams[0]
    assert model is streams[0].taehv
    assert not model.training
    assert not next(model.parameters()).requires_grad
    torch.testing.assert_close(next(model.parameters()), torch.tensor([1.25], dtype=torch.bfloat16), rtol=0, atol=0)
    raw = name.startswith("light")
    assert session._decoder_needs_scaling is raw
    monkeypatch.setenv("WORLDFOUNDRY_LINGBOT_TAE_LATENT_FORMAT", "normalized" if raw else "raw")
    session._load_decoder()
    assert session._decoder_model is model
    assert len(streams) == 1
    assert session._decoder_needs_scaling is raw


@pytest.fixture(scope="module")
def tiny_autoencoder() -> realtime.TAEHV:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = realtime.TAEHV(checkpoint_path=None, decoder_space_upscale=(False, False, False))
    return model.eval().requires_grad_(False).to(torch.bfloat16)


@pytest.mark.parametrize("session_cls", [realtime.LingBotRealtimeSession, LingBotWorldV2RealtimeSession])
@pytest.mark.parametrize("raw", [False, True])
@torch.no_grad()
def test_real_tae_streams_share_weights_not_temporal_memory(
    tiny_autoencoder: realtime.TAEHV, session_cls: type[realtime.LingBotRealtimeSession], raw: bool
) -> None:
    rng = torch.random.get_rng_state().clone()
    model = tiny_autoencoder
    mean = torch.linspace(-0.5, 0.5, 16, dtype=torch.bfloat16)
    std = torch.linspace(0.5, 1.5, 16, dtype=torch.bfloat16)

    def session() -> realtime.LingBotRealtimeSession:
        result = object.__new__(session_cls)
        result.rank = 0
        result.autoregressive_index = 0
        result.core = SimpleNamespace(vae=SimpleNamespace(mean=mean, std=std))
        result._decoder_needs_scaling = raw
        result._decoder_model = model
        result._decoder = realtime.StreamingTAEHV(model)
        return result

    a, b = session(), session()
    a_decoder, b_decoder = a._decoder, b._decoder
    assert isinstance(a_decoder, realtime.StreamingTAEHV) and isinstance(b_decoder, realtime.StreamingTAEHV)
    assert a_decoder is not b_decoder
    assert a._decoder_model is b._decoder_model is model
    weights = tuple(model.parameters())
    count = session_cls.chunk_latent_frames
    latents = torch.linspace(-0.75, 1.25, 16 * count * 2, dtype=torch.bfloat16).reshape(16, count * 2, 1, 1)
    original = latents.clone()
    z = latents.permute(1, 0, 2, 3)[None]
    if raw:
        z = z * std[None, None, :, None, None] + mean[None, None, :, None, None]
    reference = model.decode_video(z, parallel=False, show_progress_bar=False)[0]
    expected = reference.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
    first = a._decode(latents[:, :count])
    b._decode(-latents[:, :count])
    a.autoregressive_index = 1
    second = a._decode(latents[:, count:])
    np.testing.assert_array_equal(np.concatenate((first, second)), expected)
    a_decoder.reset()
    a.autoregressive_index = 0
    np.testing.assert_array_equal(a._decode(latents[:, :count]), first)
    assert not a_decoder.decoder_work_queue
    assert a_decoder.decoder_memory is not b_decoder.decoder_memory
    assert all(left is right for left, right in zip(weights, model.parameters()))
    torch.testing.assert_close(latents, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
