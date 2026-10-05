"""Projection fusion must preserve the actual causal/cache attention algorithm."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import causal_action_21 as wan
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    fuse_qkv_projections,
    project_fused_qkv,
    qkv_fusion_report,
)


def _cache():
    return {
        "k": torch.zeros(1, 24, 2, 12),
        "v": torch.zeros(1, 24, 2, 12),
        "global_end_index": torch.tensor(0),
        "local_end_index": torch.tensor(0),
    }


@pytest.mark.parametrize("strategy,threshold", [("packed", 8), ("split", 8), ("auto", 8), ("auto", 9)])
def test_fusion_preserves_repeated_denoising_and_sink_cache_rollover(monkeypatch, strategy, threshold):
    def cpu_attention(query, key, value):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(wan, "attention", cpu_attention)
    torch.manual_seed(81)
    dense = wan.CausalWanSelfAttention(24, 2, local_attn_size=4, sink_size=1).eval()
    fused = copy.deepcopy(dense)
    original_forward = fused.forward.__func__
    assert fuse_qkv_projections(fused, strategy=strategy, split_threshold=threshold) == 1
    assert fused.forward.__func__ is original_forward
    assert not hasattr(fused, "q")
    dense_cache, fused_cache = _cache(), _cache()
    grid = torch.tensor([2, 1, 4])
    freqs = wan.rope_params(16, dense.head_dim)

    with torch.no_grad():
        for start in [0, 0, 8, 8, 16, 24, 24, 32]:
            hidden = torch.randn(1, 8, 24)
            kwargs = dict(
                seq_lens=torch.tensor([8]), grid_sizes=grid, freqs=freqs, block_mask=None, current_start=start
            )
            expected = dense(hidden, kv_cache=dense_cache, **kwargs)
            actual = fused(hidden, kv_cache=fused_cache, **kwargs)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            for name in dense_cache:
                torch.testing.assert_close(fused_cache[name], dense_cache[name], atol=2e-6, rtol=2e-5)
            assert int(fused_cache["global_end_index"]) == start + 8
            assert int(fused_cache["local_end_index"]) == min(start + 8, 24)

    report = qkv_fusion_report(fused)
    assert report["eager_projection_calls"] == 8
    selected = "split" if strategy == "split" or strategy == "auto" and threshold == 8 else "packed"
    assert report[f"eager_{selected}_projection_calls"] == 8


@pytest.mark.parametrize("tokens", [127, 128, 129])
def test_fusion_preserves_uncached_padding_and_rope(monkeypatch, tokens):
    def cpu_flex(*, query, key, value, block_mask):
        positions = torch.arange(query.shape[2])
        allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] < tokens)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)

    monkeypatch.setattr(wan, "flex_attention", cpu_flex)
    torch.manual_seed(83)
    dense = wan.CausalWanSelfAttention(24, 2).eval()
    fused = copy.deepcopy(dense)
    assert fuse_qkv_projections(fused) == 1
    hidden = torch.randn(1, tokens, 24)
    kwargs = dict(
        seq_lens=torch.tensor([tokens]),
        grid_sizes=torch.tensor([1, 1, tokens]),
        freqs=wan.rope_params(tokens, 12),
        block_mask=None,
    )
    with torch.no_grad():
        torch.testing.assert_close(fused(hidden, **kwargs), dense(hidden, **kwargs), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("strategy", ["packed", "split"])
def test_causal_projection_compiles_fullgraph_without_replacing_forward(strategy):
    torch.manual_seed(87)
    attention = wan.CausalWanSelfAttention(24, 2).eval()
    hidden = torch.randn(1, 8, 24)
    expected = attention.q(hidden), attention.k(hidden), attention.v(hidden)
    assert fuse_qkv_projections(attention, strategy=strategy) == 1
    compiled = torch.compile(lambda value: project_fused_qkv(attention.qkv, value), backend="eager", fullgraph=True)
    for actual, reference in zip(compiled(hidden), expected):
        torch.testing.assert_close(actual, reference)
    assert qkv_fusion_report(attention)["compiled_graph_traces"] > 0
