"""CPU tests for the merged-QKV build transform.

Parity/speedup on real Wan SelfAttention is covered by the H100 microbenchmark;
here we lock the structural transform and its no-op/idempotence behavior without
needing RoPE freqs or a GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    _is_fusible_self_attention,
    fuse_qkv_projections,
    qkv_fusion_report,
)


class _AttnLike(nn.Module):
    """Minimal duck-type of Wan SelfAttention for structural testing."""

    def __init__(self, dim: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = 4
        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.o = nn.Linear(dim, dim, bias=bias)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.attn = nn.Identity()


# Give the duck-type the class name the transform gates on.
_AttnLike.__name__ = "SelfAttention"


class _Merged(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)


def test_fuse_concatenates_weights_exactly() -> None:
    m = _AttnLike(dim=64)
    q_w = m.q.weight.detach().clone()
    k_w = m.k.weight.detach().clone()
    v_w = m.v.weight.detach().clone()
    q_b = m.q.bias.detach().clone()
    n = fuse_qkv_projections(m)
    assert n == 1
    assert hasattr(m, "qkv") and not hasattr(m, "q")
    assert m.qkv.weight.shape == (3 * 64, 64)
    torch.testing.assert_close(m.qkv.weight[:64], q_w)
    torch.testing.assert_close(m.qkv.weight[64:128], k_w)
    torch.testing.assert_close(m.qkv.weight[128:], v_w)
    torch.testing.assert_close(m.qkv.bias[:64], q_b)


def test_fused_forward_matches_separate() -> None:
    m = _AttnLike(dim=64)
    # attn=Identity returns (q,k,v) not usable; instead validate the projection
    # math by comparing chunked fused output to separate projections directly.
    x = torch.randn(2, 8, 64)
    q_ref = m.q(x)
    k_ref = m.k(x)
    v_ref = m.v(x)
    fuse_qkv_projections(m)
    qkv = m.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)


def test_runtime_report_requires_projection_execution() -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    assert qkv_fusion_report(m) == {
        "fused_blocks": 1,
        "eager_projection_calls": 0,
        "compiled_graph_traces": 0,
        "request_compiled_graph_traces": 0,
        "lifetime_eager_projection_calls": 0,
        "lifetime_compiled_graph_traces": 0,
        "strategy": "packed",
        "split_threshold": 8192,
        "eager_packed_projection_calls": 0,
        "eager_split_projection_calls": 0,
        "compiled_packed_graph_traces": 0,
        "compiled_split_graph_traces": 0,
        "lifetime_eager_packed_projection_calls": 0,
        "lifetime_eager_split_projection_calls": 0,
        "lifetime_compiled_packed_graph_traces": 0,
        "lifetime_compiled_split_graph_traces": 0,
        "execution": "installed (runtime-pending)",
    }

    m.qkv(torch.randn(2, 8, 64))

    report = qkv_fusion_report(m)
    assert report is not None
    assert report["eager_projection_calls"] == 1
    assert report["execution"] == "eager-packed-projection-executed"


def test_auto_strategy_dispatches_on_both_sides_of_cutoff() -> None:
    m = _AttnLike(dim=64)
    q_weight = m.q.weight.detach().clone()
    k_weight = m.k.weight.detach().clone()
    v_weight = m.v.weight.detach().clone()
    q_bias = m.q.bias.detach().clone()
    k_bias = m.k.bias.detach().clone()
    v_bias = m.v.bias.detach().clone()
    assert (
        fuse_qkv_projections(m, strategy="auto", split_threshold=16) == 1
    )

    small = torch.randn(1, 8, 64)
    q_small, k_small, v_small = m.qkv.project_qkv(small)
    torch.testing.assert_close(q_small, torch.nn.functional.linear(small, q_weight, q_bias))
    torch.testing.assert_close(k_small, torch.nn.functional.linear(small, k_weight, k_bias))
    torch.testing.assert_close(v_small, torch.nn.functional.linear(small, v_weight, v_bias))
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["eager_packed_projection_calls"] == 1
    assert report["eager_split_projection_calls"] == 0

    m._worldfoundry_qkv_fusion.reset_request_window()
    large = torch.randn(1, 16, 64)
    q_large, k_large, v_large = m.qkv.project_qkv(large)
    torch.testing.assert_close(q_large, torch.nn.functional.linear(large, q_weight, q_bias))
    torch.testing.assert_close(k_large, torch.nn.functional.linear(large, k_weight, k_bias))
    torch.testing.assert_close(v_large, torch.nn.functional.linear(large, v_weight, v_bias))
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["eager_packed_projection_calls"] == 0
    assert report["eager_split_projection_calls"] == 1
    assert report["execution"] == "eager-split-projection-executed"


@pytest.mark.parametrize("strategy", ("auto", "packed", "split"))
def test_qkv_strategy_is_reconfigurable_without_repacking(strategy: str) -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    weight_pointer = m.qkv.weight.data_ptr()
    assert fuse_qkv_projections(m, strategy=strategy, split_threshold=32) == 0
    assert m.qkv.weight.data_ptr() == weight_pointer
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["strategy"] == strategy
    assert report["split_threshold"] == 32


def test_qkv_strategy_validation_is_fail_closed() -> None:
    m = _AttnLike(dim=64)
    with pytest.raises(ValueError, match="strategy"):
        fuse_qkv_projections(m, strategy="benchmark-at-runtime")
    with pytest.raises(ValueError, match="positive"):
        fuse_qkv_projections(m, split_threshold=0)


def test_runtime_state_survives_idempotent_reinstall() -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    m.qkv(torch.randn(1, 2, 64))
    assert fuse_qkv_projections(m) == 0
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["fused_blocks"] == 1
    assert report["eager_projection_calls"] == 1


def test_request_window_does_not_reuse_eager_projection_receipts() -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    m.qkv(torch.randn(1, 2, 64))
    state = m._worldfoundry_qkv_fusion

    state.reset_request_window()
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["eager_projection_calls"] == 0
    assert report["lifetime_eager_projection_calls"] == 1
    assert report["execution"] == "installed (runtime-pending)"

    m.qkv(torch.randn(1, 2, 64))
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["eager_projection_calls"] == 1
    assert report["lifetime_eager_projection_calls"] == 2


def test_compiled_projection_records_trace_without_graph_break() -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    compiled_projection = torch.compile(m.qkv, backend="eager", fullgraph=True)

    output = compiled_projection(torch.randn(1, 2, 64))

    assert output.shape == (1, 2, 192)
    report = qkv_fusion_report(m)
    assert report is not None
    assert report["compiled_graph_traces"] > 0
    assert report["execution"] == (
        "compiled-graph-traced (execution-pending-wrapper-receipt)"
    )


def test_idempotent_and_noop_on_merged() -> None:
    m = _AttnLike(dim=64)
    assert fuse_qkv_projections(m) == 1
    # Second pass: already fused -> no-op.
    assert fuse_qkv_projections(m) == 0
    # Already-merged module (Sana-style) is never fusible.
    merged = _Merged(dim=64)
    assert _is_fusible_self_attention(merged) is False
    assert fuse_qkv_projections(merged) == 0


def test_bias_consistency_required() -> None:
    m = _AttnLike(dim=64, bias=True)
    m.k = nn.Linear(64, 64, bias=False)  # inconsistent bias -> not fusible
    assert _is_fusible_self_attention(m) is False
