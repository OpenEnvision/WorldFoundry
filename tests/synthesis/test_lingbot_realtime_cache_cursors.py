# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Request-owned host cursors preserve eager causal cache arithmetic."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from worldfoundry.synthesis.visual_generation.lingbot_world import realtime
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2.realtime import LingBotWorldV2SmallRealtimeSession

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("session_type", [LingBotRealtimeSession, LingBotWorldV2SmallRealtimeSession])
@pytest.mark.parametrize("mode", ["eager", "compiled", "sp", "fsdp"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cursor_admission_and_request_reset(
    monkeypatch: pytest.MonkeyPatch,
    session_type: type[LingBotRealtimeSession],
    mode: str,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = SimpleNamespace(config=SimpleNamespace(dim=64, num_heads=4, num_layers=2), blocks=[])
    if mode == "compiled":
        model = SimpleNamespace(_orig_mod=model)
    monkeypatch.setattr(realtime, "_is_fsdp_module", lambda _model: mode == "fsdp")
    allocated: list[list[dict[str, torch.Tensor]]] = []

    def allocate(*, num_layers: int, shape: list[int], dtype: torch.dtype, device: torch.device) -> Any:
        layers = [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
            }
            for _ in range(num_layers)
        ]
        allocated.append(layers)
        return layers

    cursor_copies: list[torch.Tensor] = []
    original_cpu = torch.Tensor.cpu

    def to_cpu(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        cursor_copies.append(value)
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", to_cpu)
    monkeypatch.setattr(session_type, "_load_decoder", lambda _self: None)
    monkeypatch.setattr(session_type, "_compile_model_if_enabled", lambda _self: None)
    core = SimpleNamespace(
        model=model,
        _initialize_self_kv_cache=allocate,
        sp_size=2 if mode == "sp" else 1,
        device=torch.device(device),
        param_dtype=torch.bfloat16,
        pipe_dtype=torch.bfloat16,
        vae=None,
        text_encoder=None,
        scheduler=None,
        config=None,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
    )
    session = session_type(core)
    session.cache_sequence_length = 12
    monkeypatch.setattr(session, "_new_cross_attention_cache", lambda: [{"is_init": False}])
    session._initialize_model_caches()
    first = session._self_cache
    assert first is not None
    assert len(cursor_copies) == (4 if mode == "eager" else 0)
    for cache in first:
        assert cache["k"].device.type == cache["v"].device.type == device
        for key in ("global_end_index", "local_end_index"):
            assert cache[key].device.type == ("cpu" if mode == "eager" else device)
            assert cache[key].shape == (1,) and cache[key].dtype == torch.int64
            assert cache[key].item() == 0
            cache[key].fill_(12)
    session._initialize_model_caches()
    assert session._self_cache is not None and session._self_cache is not first
    for cache in session._self_cache:
        assert cache["global_end_index"].item() == cache["local_end_index"].item() == 0
        assert torch.count_nonzero(cache["k"]) == torch.count_nonzero(cache["v"]) == 0
    assert session._cross_cache == [{"is_init": False}]


@pytest.mark.parametrize("variant", ["fast", "v2"])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param("cuda", torch.float16, marks=pytest.mark.gpu),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.gpu),
    ],
)
@torch.no_grad()
def test_host_cursors_preserve_repeated_steps_eviction_and_sink(
    variant: str,
    device: str,
    dtype: torch.dtype,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from worldfoundry.base_models.diffusion_model.models.networks.wan.reference_22 import rope_params

    if variant == "v2":
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot_world_v2 import (
            CausalWanSelfAttention,
        )
    else:
        from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot.fast import (
            CausalWanSelfAttention,
        )

    torch.manual_seed(11)
    attention = (
        CausalWanSelfAttention(dim=64, num_heads=4, local_attn_size=6, sink_size=1, qk_norm=False)
        .to(device=device, dtype=dtype)
        .eval()
    )
    setattr(attention, "attention_backend", "torch")
    freqs = torch.cat([rope_params(64, 8), rope_params(64, 4), rope_params(64, 4)], dim=1).to(device)

    def new_cache(index_device: str) -> dict[str, torch.Tensor]:
        return {
            "k": torch.zeros(1, 12, 4, 16, device=device, dtype=dtype),
            "v": torch.zeros(1, 12, 4, 16, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=index_device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=index_device),
        }

    reference, candidate = new_cache(device), new_cache("cpu")
    sink = None
    for chunk in range(8):
        for step in range(4):
            x = torch.randn(1, 4, 64, device=device, dtype=dtype)
            kwargs = {
                "seq_lens": torch.tensor([4]),
                "grid_sizes": torch.tensor([[2, 1, 2]]),
                "freqs": freqs,
                "current_start": 4 * chunk,
                "frame_seqlen": 2,
                "max_attention_size": 12,
            }
            expected = attention(x, kv_cache=reference, **kwargs)
            actual = attention(x, kv_cache=candidate, **kwargs)
            assert torch.equal(actual, expected) and torch.count_nonzero(actual) > 0
            assert torch.equal(reference["k"], candidate["k"]) and torch.equal(reference["v"], candidate["v"])
            for key in ("global_end_index", "local_end_index"):
                assert reference[key].item() == candidate[key].item()
            if chunk == 0 and step == 3:
                sink = candidate["k"][:, :2].clone()
            if chunk >= 1:
                assert sink is not None and torch.equal(sink, candidate["k"][:, :2])
    assert candidate["global_end_index"].item() == 32 and candidate["local_end_index"].item() == 12
