# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Static text and rotary inputs preserve the complete causal denoiser."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import lingbot_world_v2 as network
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import StaticConditionCache
from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import LingBotRealtimeSession
from worldfoundry.synthesis.visual_generation.lingbot_world_v2 import realtime as sessions

pytestmark = pytest.mark.unit


def _model(device: str = "cpu") -> network.WanModelFast:
    model = (
        network.WanModelFast(
            model_type="t2v",
            in_dim=4,
            out_dim=4,
            dim=64,
            ffn_dim=96,
            freq_dim=16,
            text_dim=16,
            text_len=8,
            num_heads=4,
            num_layers=2,
            local_attn_size=6,
            sink_size=2,
        )
        .eval()
        .to(device)
    )
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.03)
    for block in model.blocks:
        block.self_attn.attention_backend = "torch"
    return model


def _caches(device: str, dtype: torch.dtype) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    own = [
        {
            "k": torch.zeros(1, 24, 4, 16, device=device, dtype=dtype),
            "v": torch.zeros(1, 24, 4, 16, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }
        for _ in range(2)
    ]
    cross = [
        {
            "k": torch.zeros(1, 8, 4, 16, device=device, dtype=dtype),
            "v": torch.zeros(1, 8, 4, 16, device=device, dtype=dtype),
            "is_init": torch.zeros(1, dtype=torch.bool, device=device),
        }
        for _ in range(2)
    ]
    return own, cross


def _original_rope(value: torch.Tensor, grid: torch.Tensor, frequencies: torch.Tensor, start: int) -> torch.Tensor:
    heads, pairs = value.shape[-2], value.shape[-1] // 2
    tables = frequencies.split([pairs - 2 * (pairs // 3), pairs // 3, pairs // 3], dim=1)
    outputs = []
    for index, (frames, height, width) in enumerate(grid.tolist()):
        tokens = frames * height * width
        hidden = torch.view_as_complex(value[index, :tokens].to(torch.float64).reshape(tokens, heads, -1, 2))
        expanded = torch.cat(
            [
                tables[0][start : start + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                tables[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                tables[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(tokens, 1, -1)
        rotated = torch.view_as_real(hidden * expanded).flatten(2)
        outputs.append(torch.cat([rotated, value[index, tokens:]]))
    return torch.stack(outputs).type_as(value)


@pytest.mark.parametrize("start", [0, 12, 80])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_prepared_rope_preserves_fp64_multiply_padding_and_batch(start: int, dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    model = _model()
    grids = torch.tensor([[2, 2, 3], [3, 1, 2]])
    inputs = torch.randn(2, 14, 4, 16).to(dtype)
    with torch.no_grad():
        prepared = model.prepare_rotary_frequencies(
            [torch.empty(4, 2, 4, 6), torch.empty(4, 3, 2, 4)],
            current_start=start * 6,
            frame_seqlen=6,
        )
    assert all(value.dtype == torch.complex128 for value in prepared)
    expected = _original_rope(inputs, grids, model.freqs, start)
    assert torch.equal(network.causal_rope_apply(inputs, grids, model.freqs, start), expected)
    assert torch.equal(network.causal_rope_apply(inputs, grids, prepared, start), expected)
    assert torch.equal(expected[0, 12:], inputs[0, 12:])
    assert torch.equal(expected[1, 6:], inputs[1, 6:])


@pytest.mark.parametrize("mode", ["text", "rope", "both"])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param("cuda", torch.float16, marks=pytest.mark.gpu),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.gpu),
    ],
)
@torch.no_grad()
def test_full_forward_retains_live_queries_prompt_replacement_and_eviction(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    device: str,
    dtype: torch.dtype,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    model = _model(device)
    keys = tuple(model.state_dict())
    baseline, candidate = _caches(device, dtype), _caches(device, dtype)
    calls: dict[str, Counter[str]] = {"raw": Counter(), "cached": Counter()}
    phase = "raw"
    assemble = network.assemble_complex_rotary_frequencies_3d

    def counted_assemble(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls[phase]["rope"] += 1
        return assemble(*args, **kwargs)

    def count_text(*_args: Any) -> None:
        calls[phase]["text"] += 1

    def count_query(*_args: Any) -> None:
        calls[phase]["query"] += 1

    monkeypatch.setattr(network, "assemble_complex_rotary_frequencies_3d", counted_assemble)
    handles = [model.text_embedding.register_forward_hook(count_text)]
    handles += [block.cross_attn.q.register_forward_hook(count_query) for block in model.blocks]
    try:
        with torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext():
            for chunk in range(6):
                if chunk in (0, 3):
                    context = torch.randn(3 + chunk, 16, device=device, dtype=dtype)
                    cache = StaticConditionCache(context.unsqueeze(0))
                    # A prompt update replaces text K/V but retains self-attention history.
                    baseline = baseline[0], _caches(device, dtype)[1]
                    candidate = candidate[0], _caches(device, dtype)[1]
                phase = "cached"
                latent = torch.randn(4, 2, 4, 4, device=device, dtype=dtype)
                prepared = (
                    model.prepare_rotary_frequencies([latent], current_start=chunk * 8, frame_seqlen=4)
                    if mode != "text"
                    else None
                )
                for step in range(3):
                    latent = torch.randn_like(latent)
                    args = dict(
                        x=[latent],
                        t=torch.tensor([500.0 - step], device=device),
                        seq_len=8,
                        current_start=chunk * 8,
                        frame_seqlen=4,
                        max_attention_size=24,
                        cross_attn_first_call=chunk in (0, 3) and step == 0,
                    )
                    phase = "raw"
                    expected = model(context=[context], kv_cache=baseline[0], crossattn_cache=baseline[1], **args)[0]
                    phase = "cached"
                    actual = model(
                        context=cache if mode != "rope" else [context],
                        rotary_frequencies=prepared,
                        kv_cache=candidate[0],
                        crossattn_cache=candidate[1],
                        **args,
                    )[0]
                    assert torch.equal(actual, expected) and torch.isfinite(actual).all()
                    assert torch.count_nonzero(actual) > 0
                    for group_a, group_b in zip(baseline, candidate):
                        for a, b in zip(group_a, group_b):
                            assert all(torch.equal(a[key], b[key]) for key in a)
        assert calls["raw"] == {"text": 18, "query": 36, "rope": 72}
        assert calls["cached"] == {
            "text": 2 if mode != "rope" else 18,
            "query": 36,
            "rope": 6 if mode != "text" else 72,
        }
        assert candidate[0][0]["global_end_index"].item() == 48
        assert candidate[0][0]["local_end_index"].item() == 24
        assert tuple(model.state_dict()) == keys
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def test_projection_cache_isolated_across_conditions_weights_and_precision() -> None:
    torch.manual_seed(23)
    model = _model()
    first, second = torch.randn(1, 3, 16), torch.randn(1, 5, 16)
    a, b = StaticConditionCache(first), StaticConditionCache(second)
    original = a.project_context(model, model._project_text_context)
    other = b.project_context(model, model._project_text_context)
    assert original is a.project_context(model, model._project_text_context)
    assert not torch.equal(original, other)
    assert torch.equal(original, model._project_text_context(first))
    assert torch.equal(other, model._project_text_context(second))
    model.text_embedding[0].weight.add_(0.25)
    replaced = StaticConditionCache(first).project_context(model, model._project_text_context)
    assert not torch.equal(original, replaced)
    assert torch.equal(replaced, model._project_text_context(first))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lower = StaticConditionCache(first).project_context(model, model._project_text_context)
        assert lower.dtype == torch.bfloat16
        assert torch.equal(lower, model._project_text_context(first))


def test_raw_training_and_prepared_inference_admission() -> None:
    torch.manual_seed(7)
    model = _model()
    latent = torch.randn(4, 2, 4, 4, requires_grad=True)
    context = torch.randn(3, 16, requires_grad=True)
    own, cross = _caches("cpu", torch.float32)
    args = dict(x=[latent], t=torch.tensor([500.0]), seq_len=8, kv_cache=own, crossattn_cache=cross, frame_seqlen=4)
    model.train()
    model(context=[context], **args)[0].square().mean().backward()
    assert latent.grad is not None and torch.count_nonzero(latent.grad) > 0
    assert context.grad is not None and torch.count_nonzero(context.grad) > 0
    gradient = model.text_embedding[0].weight.grad
    assert gradient is not None and torch.count_nonzero(gradient) > 0
    model.eval()
    with pytest.raises(RuntimeError, match="no-grad"):
        model.prepare_rotary_frequencies([latent], current_start=0, frame_seqlen=4)
    with pytest.raises(RuntimeError, match="no-grad"):
        model(context=StaticConditionCache(context.unsqueeze(0)), **args)
    with torch.no_grad():
        prepared = model.prepare_rotary_frequencies([latent], current_start=0, frame_seqlen=4)
    with pytest.raises(RuntimeError, match="no-grad"):
        model(context=[context], rotary_frequencies=prepared, **args)


def _session(
    monkeypatch: pytest.MonkeyPatch, session_type: type[LingBotRealtimeSession], mode: str
) -> LingBotRealtimeSession:
    monkeypatch.setattr(session_type, "_load_decoder", lambda _self: None)
    monkeypatch.setattr(session_type, "_compile_model_if_enabled", lambda _self: None)
    monkeypatch.setattr(sessions, "_is_fsdp_module", lambda _model: mode == "fsdp")
    model: Any = _model()
    if mode == "compiled":
        model = SimpleNamespace(_orig_mod=model)
    core = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        sp_size=2 if mode == "sp" else 1,
        param_dtype=torch.float32,
        pipe_dtype=torch.float32,
        vae=None,
        text_encoder=None,
        scheduler=None,
        config=None,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2),
    )
    return session_type(core)


@pytest.mark.parametrize(
    "session_type",
    [LingBotRealtimeSession, sessions.LingBotWorldV2RealtimeSession, sessions.LingBotWorldV2SmallRealtimeSession],
)
@pytest.mark.parametrize("mode", ["eager", "compiled", "sp", "fsdp"])
@torch.no_grad()
def test_static_input_admission(
    monkeypatch: pytest.MonkeyPatch, session_type: type[LingBotRealtimeSession], mode: str
) -> None:
    session = _session(monkeypatch, session_type, mode)
    enabled = session_type is sessions.LingBotWorldV2SmallRealtimeSession and mode == "eager"
    assert (session._new_text_condition([torch.randn(3, 16)]) is not None) == enabled
    assert (session._prepare_rotary_frequencies(torch.randn(4, 2, 4, 4), current_start=0) is not None) == enabled


@torch.no_grad()
def test_prompt_update_is_atomic_and_reset_discards_text_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, sessions.LingBotWorldV2SmallRealtimeSession, "eager")
    session.configured = True
    session._prompt = "old"
    session._context = [torch.randn(3, 16)]
    session._text_condition = session._new_text_condition(session._context)
    session._self_cache = [{"history": torch.randn(2)}]
    session._cross_cache = [{"history": torch.randn(2)}]
    session._vae_state = {"history": object()}
    session._generator = torch.Generator().manual_seed(3)
    old = (session._context, session._text_condition, session._cross_cache)
    video = (session._self_cache, session._vae_state, session._generator)
    new_context = [torch.randn(5, 16)]
    monkeypatch.setattr(session, "_encode_prompt", lambda _prompt: new_context)

    def fail_allocate() -> Any:
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(session, "_new_cross_attention_cache", fail_allocate)
    with pytest.raises(RuntimeError, match="allocation failed"):
        session.update_prompt("new")
    assert all(a is b for a, b in zip(old, (session._context, session._text_condition, session._cross_cache)))
    monkeypatch.setattr(session, "_new_cross_attention_cache", lambda: [{"empty": True}])
    assert session.update_prompt("new")
    assert session._context is new_context and session._text_condition is not old[1]
    assert all(a is b for a, b in zip(video, (session._self_cache, session._vae_state, session._generator)))
    text = session._text_condition
    assert not session.update_prompt("new") and session._text_condition is text
    session.reset()
    assert session._text_condition is None and session._context is None
