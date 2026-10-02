from __future__ import annotations

import copy

import pytest
import torch

from scripts.benchmark_model_attention import legacy_hrdt, legacy_longcat
from worldfoundry.synthesis.action_generation.h_rdt import modeling as hrdt
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules.attention import (
    Attention as LongCatAttention,
)
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules.rope_3d import (
    RotaryPositionalEmbedding,
)


@pytest.fixture(autouse=True)
def clean_optimization_environment(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_HRDT_GQA_OPTIMIZATION", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_LONGCAT_KV_OPTIMIZATION", raising=False)


def hrdt_layer(*, cross=False, kv_heads=2, backend="math", device="cpu", dtype=torch.float32, enabled=True):
    config = dict(hidden_size=64, num_heads=4, num_kv_heads=kv_heads, norm_eps=1e-6)
    cls = hrdt.CrossAttention if cross else hrdt.Attention
    return cls(config, attention_backend=backend, enable_gqa_optimization=enabled).to(device, dtype).eval()


@pytest.mark.parametrize("model", ["hrdt-self", "hrdt-cross", "longcat"])
@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("false", False), ("1", True), ("true", True)])
def test_optimization_opt_in_and_explicit_override(monkeypatch, model, value, expected):
    if model == "longcat":
        env_name, attribute = "WORLDFOUNDRY_LONGCAT_KV_OPTIMIZATION", "enable_kv_optimization"

        def build(**kwargs):
            return LongCatAttention(64, 4, cp_split_hw=(1, 1), **kwargs)

    else:
        env_name, attribute = "WORLDFOUNDRY_HRDT_GQA_OPTIMIZATION", "enable_gqa_optimization"
        cls = hrdt.CrossAttention if model == "hrdt-cross" else hrdt.Attention

        def build(**kwargs):
            return cls(dict(hidden_size=64, num_heads=4, num_kv_heads=2, norm_eps=1e-6), **kwargs)

    if value is not None:
        monkeypatch.setenv(env_name, value)
    layer = build()
    assert getattr(layer, attribute) is expected
    assert getattr(build(**{attribute: False}), attribute) is False
    assert getattr(build(**{attribute: True}), attribute) is True
    monkeypatch.setenv(env_name, "0" if expected else "1")
    assert getattr(layer, attribute) is expected
    assert getattr(build(), attribute) is not expected


def test_model_optimization_switches_are_independent(monkeypatch):
    for hrdt_enabled, longcat_enabled in ((True, False), (False, True)):
        monkeypatch.setenv("WORLDFOUNDRY_HRDT_GQA_OPTIMIZATION", str(int(hrdt_enabled)))
        monkeypatch.setenv("WORLDFOUNDRY_LONGCAT_KV_OPTIMIZATION", str(int(longcat_enabled)))
        assert hrdt_layer(enabled=None).enable_gqa_optimization is hrdt_enabled
        assert LongCatAttention(64, 4, cp_split_hw=(1, 1)).enable_kv_optimization is longcat_enabled


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("mask_kind", ["self", "none", "padding", "empty"])
@pytest.mark.parametrize("enabled", [None, False, True])
@torch.no_grad()
def test_hrdt_real_module_parity_and_head_layout(monkeypatch, kv_heads, mask_kind, enabled):
    torch.manual_seed(13)
    layer = hrdt_layer(cross=mask_kind != "self", kv_heads=kv_heads, enabled=enabled)
    hidden, condition = torch.randn(2, 7, 64), torch.randn(2, 11, 64)
    mask = None
    if mask_kind in {"padding", "empty"}:
        mask = torch.ones(2, 11, dtype=torch.bool)
        mask[0, 3:] = False
        mask[1] = mask_kind != "empty"
    expected = legacy_hrdt(layer, hidden, None if mask_kind == "self" else condition, mask)
    calls = []
    sdpa = hrdt.scaled_dot_product_attention

    def observe(query, key, value, **kwargs):
        calls.append((key.shape[1], value.shape[1], value.stride(-1), kwargs["enable_gqa"]))
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(hrdt, "scaled_dot_product_attention", observe)
    actual = layer(hidden) if mask_kind == "self" else layer(hidden, condition, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5 if enabled else 0, atol=2e-6 if enabled else 0)
    assert torch.isfinite(actual).all()
    expected_heads = kv_heads if enabled else 4
    assert calls[0][:2] == (expected_heads, expected_heads)
    assert calls[0][3] == bool(enabled and kv_heads < 4)
    if enabled and kv_heads < 4:
        assert calls[0][2] == 1


@pytest.mark.parametrize("enabled", [False, True])
@torch.no_grad()
def test_hrdt_single_example_mask_layout(monkeypatch, enabled):
    layer = hrdt_layer(cross=True, enabled=enabled)
    sdpa = hrdt.scaled_dot_product_attention
    mask_dims = []

    def observe(query, key, value, **kwargs):
        mask_dims.append(kwargs["attn_mask"].ndim)
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(hrdt, "scaled_dot_product_attention", observe)
    layer(torch.randn(1, 7, 64), torch.randn(1, 11, 64), torch.ones(1, 11, dtype=torch.bool))
    assert mask_dims == [2 if enabled else 4]


@pytest.mark.parametrize("backends", [("cudnn",), ("cudnn", "efficient")])
def test_hrdt_cudnn_selection_retains_expanded_contract(monkeypatch, backends):
    monkeypatch.setenv("WORLDFOUNDRY_NATIVE_SDPA_PRIORITY", ",".join(backends))
    layer = hrdt_layer(backend="auto")
    sdpa = hrdt.scaled_dot_product_attention

    def observe(query, key, value, **kwargs):
        assert key.shape[1] == value.shape[1] == 4
        assert kwargs["enable_gqa"] is False
        assert kwargs["backends"] == backends
        # Check the adapter on CPU; this does not claim cuDNN kernel coverage.
        kwargs["backends"] = ("math",)
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(hrdt, "scaled_dot_product_attention", observe)
    layer(torch.randn(1, 7, 64))


def test_hrdt_gradients_match_explicit_heads():
    baseline = hrdt_layer(cross=True)
    optimized = copy.deepcopy(baseline)
    hidden = torch.randn(2, 7, 64, requires_grad=True)
    condition = torch.randn(2, 11, 64, requires_grad=True)
    hidden_new, condition_new = hidden.detach().clone().requires_grad_(), condition.detach().clone().requires_grad_()
    legacy_hrdt(baseline, hidden, condition).square().sum().backward()
    optimized(hidden_new, condition_new).square().sum().backward()
    torch.testing.assert_close(hidden_new.grad, hidden.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(condition_new.grad, condition.grad, rtol=3e-5, atol=3e-6)
    for old, new in zip(baseline.parameters(), optimized.parameters()):
        torch.testing.assert_close(new.grad, old.grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("batch,cache_batch", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("enabled", [None, False, True])
@torch.no_grad()
def test_longcat_real_cached_attention_matches_padded_query(monkeypatch, batch, cache_batch, enabled):
    torch.manual_seed(31)
    layer = LongCatAttention(64, 4, enable_flashattn3=True, cp_split_hw=(1, 1), enable_kv_optimization=enabled).eval()
    rope = layer.rope_3d.forward
    rope_calls = []

    def observe(query, key, grid_size, **kwargs):
        rope_calls.append((query.shape[-2], kwargs.get("query_start", 0)))
        return rope(query, key, grid_size, **kwargs)

    monkeypatch.setattr(layer.rope_3d, "forward", observe)
    for frames, height, width, history_frames in ((1, 2, 3, 2), (2, 3, 2, 1), (1, 2, 3, 2)):
        shape = (frames, height, width)
        hidden = torch.randn(batch, frames * height * width, 64)
        cache = tuple(torch.randn(cache_batch, 4, history_frames * height * width, 16) for _ in range(2))
        before = tuple(t.clone() for t in cache)
        expected = legacy_longcat(layer, hidden, shape, history_frames, cache)
        actual = layer.forward_with_kv_cache(hidden, shape=shape, num_cond_latents=history_frames, kv_cache=cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        query_tokens, history_tokens = frames * height * width, history_frames * height * width
        assert rope_calls[-1] == ((query_tokens, history_tokens) if enabled else (query_tokens + history_tokens, 0))
        for old, current in zip(before, cache):
            torch.testing.assert_close(old, current, rtol=0, atol=0)


@pytest.mark.parametrize("query_start", [0, 6, 12])
def test_longcat_rope_query_segment_matches_full_rotation_and_gradient(query_start):
    rope = RotaryPositionalEmbedding(16, cp_split_hw=(1, 1))
    full_query = torch.randn(2, 4, 18, 16, requires_grad=True)
    key = torch.randn_like(full_query)
    query = full_query[:, :, query_start : query_start + 6].detach().clone().requires_grad_()
    expected_q, expected_k = rope(full_query, key, (3, 2, 3))
    actual_q, actual_k = rope(query, key, (3, 2, 3), query_start=query_start)
    torch.testing.assert_close(actual_q, expected_q[:, :, query_start : query_start + 6], rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    actual_q.sum().backward()
    expected_q[:, :, query_start : query_start + 6].sum().backward()
    torch.testing.assert_close(query.grad, full_query.grad[:, :, query_start : query_start + 6], rtol=0, atol=0)


@pytest.mark.parametrize("query_start", [-1, 13])
def test_longcat_rope_rejects_out_of_grid_query(query_start):
    rope = RotaryPositionalEmbedding(16, cp_split_hw=(1, 1))
    with pytest.raises(ValueError, match="inside the RoPE grid"):
        rope(torch.randn(1, 4, 6, 16), torch.randn(1, 4, 18, 16), (3, 2, 3), query_start=query_start)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["auto", "efficient"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
@torch.no_grad()
def test_cuda_hrdt_self_and_masked_cross(dtype, backend, batch, enabled):
    hidden = torch.randn(batch, 64, 64, device="cuda", dtype=dtype)
    condition = torch.randn(batch, 256, 64, device="cuda", dtype=dtype)
    for cross in (False, True):
        layer = hrdt_layer(cross=cross, backend=backend, device="cuda", dtype=dtype, enabled=enabled)
        mask = torch.rand(batch, 256, device="cuda") > 0.2 if cross else None
        if mask is not None and batch > 1:
            mask[1] = False
        expected = legacy_hrdt(layer, hidden, condition if cross else None, mask)
        actual = layer(hidden, condition, mask) if cross else layer(hidden)
        torch.testing.assert_close(actual, expected, rtol=0.01 if enabled else 0, atol=0.005 if enabled else 0)
        assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enabled", [False, True])
@torch.no_grad()
def test_cuda_longcat_multistep_and_cache_replacement(dtype, enabled):
    layer = (
        LongCatAttention(256, 4, enable_flashattn3=True, cp_split_hw=(1, 1), enable_kv_optimization=enabled)
        .to("cuda", dtype)
        .eval()
    )
    results = []
    for history_frames in (4, 8, 4):
        cache = tuple(torch.randn(1, 4, history_frames * 64, 64, device="cuda", dtype=dtype) for _ in range(2))
        for _ in range(4):
            hidden = torch.randn(2, 64, 256, device="cuda", dtype=dtype)
            expected = legacy_longcat(layer, hidden, (1, 8, 8), history_frames, cache)
            actual = layer.forward_with_kv_cache(
                hidden, shape=(1, 8, 8), num_cond_latents=history_frames, kv_cache=cache
            )
            results.append((actual, expected))
    for actual, expected in results:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
