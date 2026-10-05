"""Sanity tests for MAGI-2 portable fallback operators."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.magi2 import config, ops


def test_config_derived_shapes() -> None:
    c = config.Magi2PreviewConfig()
    assert c.num_heads_q == 24 and c.num_heads_kv == 24  # full MHA
    assert c.adapter_dim == 3072 * 4  # MHC widens by num_stream
    r = config.Magi2RefinerConfig()
    assert r.num_heads_q == 32 and r.num_heads_kv == 8  # GQA


def test_probes_disabled() -> None:
    assert ops.can_use_fused_mh_moe() is False
    assert ops.can_use_fused_attn_sink() is False


def test_swiglu7_interleaved_matches_reference() -> None:
    x = torch.randn(4, 16)
    g, l = x[..., ::2], x[..., 1::2]
    gc = g.clamp(max=7.0)
    ref = (gc * torch.sigmoid(1.702 * gc)) * (l.clamp(-7.0, 7.0) + 1.0)
    assert torch.allclose(ops.swiglu7_interleaved(x), ref, atol=1e-5)


def test_swiglu7_interleaved_chunked_matches_reference(monkeypatch) -> None:
    monkeypatch.setattr(ops, "_SWIGLU7_MAX_CHUNK_ELEMENTS", 16)
    x = torch.randn(2, 5, 8)
    g, l = x[..., ::2], x[..., 1::2]
    gc = g.clamp(max=7.0)
    ref = (gc * torch.sigmoid(1.702 * gc)) * (l.clamp(-7.0, 7.0) + 1.0)

    actual = ops.swiglu7_interleaved(x)

    assert actual.shape == (2, 5, 4)
    assert torch.allclose(actual, ref, atol=1e-5)


def test_swiglu7_interleaved_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="even interleaved"):
        ops.swiglu7_interleaved(torch.randn(3, 7))


def test_swiglu7_split_matches_reference() -> None:
    gate, up = torch.randn(4, 8), torch.randn(4, 8)
    gc = gate.clamp(max=7.0)
    ref = (gc * torch.sigmoid(1.702 * gc)) * (up.clamp(-7.0, 7.0) + 1.0)
    assert torch.allclose(ops.swiglu7_split(gate, up), ref, atol=1e-5)


def test_swiglu7_split_chunked_matches_reference(monkeypatch) -> None:
    monkeypatch.setattr(ops, "_SWIGLU7_MAX_CHUNK_ELEMENTS", 16)
    gate, up = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    gc = gate.clamp(max=7.0)
    ref = (gc * torch.sigmoid(1.702 * gc)) * (up.clamp(-7.0, 7.0) + 1.0)

    actual = ops.swiglu7_split(gate, up)

    assert actual.shape == gate.shape
    assert torch.allclose(actual, ref, atol=1e-5)


def test_swiglu7_split_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        ops.swiglu7_split(torch.randn(3, 8), torch.randn(4, 8))


def test_routing_topk_normalized() -> None:
    H, S, E, K = 12, 5, 256, 6
    logits = torch.randn(H, S, E)
    probs, idx = ops.compute_topk_probs_and_indices(logits, K)
    assert probs.shape == (H, S, K) and idx.shape == (H, S, K)
    assert torch.allclose(probs.sum(-1), torch.ones(H, S), atol=1e-4)


def test_global_sort_csr() -> None:
    H, S, E, K = 4, 6, 32, 3
    logits = torch.randn(H, S, E)
    probs, idx = ops.compute_topk_probs_and_indices(logits, K)
    gather_ids, probs_sorted, offsets = ops.flash_mh_moe_global_sort(probs, idx, E)
    assert offsets.shape == (H * E + 1,)
    assert int(offsets[-1]) == H * S * K
    assert gather_ids.shape == (H * S * K,)


def test_moe_gemm_fallback_shape_and_finite() -> None:
    S, H, d_head, d_expert, num_experts = 8, 2, 4, 6, 3
    HE = H * num_experts
    x = torch.randn(S, H, d_head)
    Wg = torch.randn(HE, d_head, d_expert) * 0.1
    Wu = torch.randn(HE, d_head, d_expert) * 0.1
    Wd = torch.randn(HE, d_expert, d_head) * 0.1
    logits = torch.randn(H, S, num_experts)
    probs, idx = ops.compute_topk_probs_and_indices(logits, 2)
    gather_ids, probs_sorted, offsets = ops.flash_mh_moe_global_sort(probs, idx, num_experts)
    y = ops.flash_mh_moe_fwd(x, gather_ids, probs_sorted, offsets, Wg, Wu, Wd)
    assert y.shape == (S, H, d_head)
    assert torch.isfinite(y).all()


def test_attention_sink_attenuates() -> None:
    q, k, v = torch.randn(6, 2, 8), torch.randn(6, 2, 8), torch.randn(6, 2, 8)
    cu = torch.tensor([0, 6])
    out = ops.attention_with_sink(q, k, v, torch.zeros(2), cu_seqlens=cu)
    assert out.shape == (6, 2, 8) and torch.isfinite(out).all()
    # A large positive sink logit attenuates the output (more mass on the null key).
    out_big = ops.attention_with_sink(q, k, v, torch.full((2,), 20.0), cu_seqlens=cu)
    assert out_big.abs().mean() < out.abs().mean() + 1e-3
