from __future__ import annotations

import copy

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.evoke.model import (
    EvokeAttention,
    EvokeAttnProcessor,
    EvokeTransformer3DModel,
)


def attention_pair(device="cpu", dtype=torch.float32, amplify=False, *, heads=2, head_dim=16):
    torch.manual_seed(17)
    baseline = (
        EvokeAttention(
            dim=heads * head_dim,
            heads=heads,
            dim_head=head_dim,
            processor=EvokeAttnProcessor(),
            restrict_self_attn=True,
            is_amplify_history=amplify,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    optimized = copy.deepcopy(baseline)
    baseline.processor.enable_cache(use_arena=False)
    optimized.processor.enable_cache(use_arena=True)
    return baseline, optimized


def rotary(batch, length, *, offset=0, device="cpu", head_dim=16):
    phase = (torch.arange(length, device=device) + offset)[:, None] * torch.linspace(
        0.01, 0.2, head_dim // 2, device=device
    )
    phase = phase.repeat_interleave(2, dim=-1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1)[None].expand(batch, -1, -1)


def run(layer, hidden, rope, current, first):
    return layer(
        hidden,
        rotary_emb=rope,
        original_context_length=current,
        original_context_length_list=[current],
        is_first_denoising_step=first,
    )


def test_model_cache_enable_preserves_explicit_arena_preference():
    # Pipelines re-enable the cache on each run without passing an arena option.
    from types import SimpleNamespace

    processor = EvokeAttnProcessor()
    custom_calls = []
    custom = SimpleNamespace(enable_cache=lambda: custom_calls.append(True))
    model = SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=p)) for p in (processor, custom)])
    EvokeTransformer3DModel.enable_kv_cache(model)
    assert processor.cache_enabled and not processor.kv_arena_enabled
    EvokeTransformer3DModel.enable_kv_cache(model, use_arena=True)
    processor.disable_cache()
    EvokeTransformer3DModel.enable_kv_cache(model)
    assert processor.cache_enabled and processor.kv_arena_enabled
    EvokeTransformer3DModel.enable_kv_cache(model, use_arena=False)
    assert not processor.kv_arena_enabled
    assert len(custom_calls) == 4


@pytest.mark.parametrize("batch", [1, 2])
@torch.no_grad()
def test_evoke_multiblock_rope_replacement_resize_and_clear(batch):
    baseline, optimized = attention_pair()
    previous_arena = None
    for block in range(6):
        history = 8 + block % 3
        for step, current in enumerate((4, 4, 6, 6)):
            hidden = torch.randn(batch, history + current, 32)
            rope = rotary(batch, history + current, offset=block * 100)
            expected = run(baseline, hidden, rope, current, step == 0)
            actual = run(optimized, hidden, rope, current, step == 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            arena = optimized.processor.kv_cache["arena"]
            arena.validate_persistent(
                (optimized.processor.kv_cache["key_history"], optimized.processor.kv_cache["value_history"])
            )
            if step == 1 or step == 3:
                assert arena is previous_arena
            else:
                assert arena is not previous_arena
            previous_arena = arena
        baseline.processor.clear_cache()
        optimized.processor.clear_cache()
        assert optimized.processor.kv_cache is None
    optimized.processor.disable_cache()
    assert optimized.processor.kv_cache is None
    assert not optimized.processor.cache_enabled


@torch.no_grad()
def test_first_step_replaces_same_shape_history_and_rope_without_explicit_clear():
    baseline, optimized = attention_pair()
    previous = None
    for offset in (0, 200, 50):
        hidden = torch.randn(1, 12, 32)
        rope = rotary(1, 12, offset=offset)
        torch.testing.assert_close(
            run(optimized, hidden, rope, 4, True), run(baseline, hidden, rope, 4, True), rtol=0, atol=0
        )
        assert optimized.processor.kv_cache["arena"] is not previous
        previous = optimized.processor.kv_cache["arena"]


@torch.no_grad()
def test_interleaved_sessions_and_new_batch_after_clear():
    sessions = [attention_pair(), attention_pair()]
    arenas = []
    for step in range(4):
        for index, (baseline, optimized) in enumerate(sessions):
            hidden = torch.randn(1, 12, 32) + index
            rope = rotary(1, 12, offset=100 * index)
            torch.testing.assert_close(
                run(optimized, hidden, rope, 4, step == 0),
                run(baseline, hidden, rope, 4, step == 0),
                rtol=0,
                atol=0,
            )
            if step == 0:
                arenas.append(optimized.processor.kv_cache["arena"])
            assert optimized.processor.kv_cache["arena"] is arenas[index]
    assert arenas[0].static[0].data_ptr() != arenas[1].static[0].data_ptr()
    baseline, optimized = sessions[0]
    baseline.processor.clear_cache()
    optimized.processor.clear_cache()
    hidden, rope = torch.randn(2, 12, 32), rotary(2, 12, offset=500)
    torch.testing.assert_close(
        run(optimized, hidden, rope, 4, True), run(baseline, hidden, rope, 4, True), rtol=0, atol=0
    )
    assert optimized.processor.kv_cache["arena"] is not arenas[0]


@pytest.mark.parametrize("history", [0, 8])
@torch.no_grad()
def test_optional_rope_and_empty_history(history):
    baseline, optimized = attention_pair()
    for step in range(3):
        hidden = torch.randn(1, history + 4, 32)
        torch.testing.assert_close(
            run(optimized, hidden, None, 4, step == 0),
            run(baseline, hidden, None, 4, step == 0),
            rtol=0,
            atol=0,
        )
        assert optimized.processor.kv_cache["arena"].layout.static_tokens == history


@torch.no_grad()
def test_history_amplification_retains_original_path():
    baseline, optimized = attention_pair(amplify=True)
    for step in range(3):
        hidden = torch.randn(1, 12, 32)
        rope = rotary(1, 12)
        torch.testing.assert_close(
            run(optimized, hidden, rope, 4, step == 0), run(baseline, hidden, rope, 4, step == 0)
        )
        assert "arena" not in optimized.processor.kv_cache


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_training_does_not_use_mutable_arena_storage(grad_enabled):
    _, optimized = attention_pair()
    optimized.train()
    hidden = torch.randn(1, 12, 32, requires_grad=True)
    with torch.set_grad_enabled(grad_enabled):
        output = run(optimized, hidden, rotary(1, 12), 4, True)
    assert "arena" not in optimized.processor.kv_cache
    if grad_enabled:
        output.sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,head_dim", [(2, 16), (40, 128)])
@torch.no_grad()
def test_cuda_real_evoke_attention_long_rollout(dtype, heads, head_dim):
    baseline, optimized = attention_pair("cuda", dtype, heads=heads, head_dim=head_dim)
    results = []
    for block in range(8):
        history, current = 128 + 64 * (block % 3), 32
        rope = rotary(1, history + current, offset=block * 500, device="cuda", head_dim=head_dim)
        for step in range(4):
            hidden = torch.randn(1, history + current, heads * head_dim, device="cuda", dtype=dtype)
            results.append(
                (run(optimized, hidden, rope, current, step == 0), run(baseline, hidden, rope, current, step == 0))
            )
    torch.cuda.synchronize()
    for actual, expected in results:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
