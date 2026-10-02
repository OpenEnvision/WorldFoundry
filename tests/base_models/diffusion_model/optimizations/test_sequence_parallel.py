"""Tests for the opt-in Ulysses sequence-parallel (SP) self-attention lane.

The all-to-all forward needs a live NCCL group (multi-GPU), so these CPU-only
tests lock the parts that don't: install swaps every Wan SelfAttention with the
SP processor, the divisibility constraint is detected, the audit records the
degree/backend without downgrading quality_tier (SP is exact), and the rank
freq-slice helper shards + pads correctly. The real 4×H100 numerical + scaling
run lives in a benchmark script, not CI.
"""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
    FusedRoPERuntimeState,
    fused_rope_runtime_report,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    fuse_qkv_projections,
)
from worldfoundry.base_models.diffusion_model.optimizations.sequence_parallel import (
    SequenceParallelSelfAttentionProcessor,
    _rank_slice_freqs,
    _SPState,
    enable_sequence_parallel,
    sequence_parallel_report,
)
from worldfoundry.core.model_loading.optimize import AppliedOptimizations


class _Stack(torch.nn.Module):
    def __init__(self, n: int = 3, dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([SelfAttention(dim, heads) for _ in range(n)])


def test_enable_wraps_all_self_attention() -> None:
    model = _Stack(n=3)
    state = enable_sequence_parallel(model, sp_degree=4)
    assert state.wrapped_blocks == 3
    for blk in model.blocks:
        assert isinstance(blk.get_processor(), SequenceParallelSelfAttentionProcessor)
    assert state.head_parallel is True  # 8 heads divisible by 4


def test_sequence_parallel_restores_processor_dispatch_after_qkv_fusion() -> None:
    model = _Stack(n=1)
    assert fuse_qkv_projections(model) == 1
    assert hasattr(model.blocks[0], "qkv")
    state = enable_sequence_parallel(model, sp_degree=2)
    assert state.wrapped_blocks == 1
    assert isinstance(
        model.blocks[0].get_processor(),
        SequenceParallelSelfAttentionProcessor,
    )
    assert "forward" not in model.blocks[0].__dict__


def test_rejects_degree_below_two() -> None:
    with pytest.raises(ValueError):
        enable_sequence_parallel(_Stack(), sp_degree=1)


def test_detects_indivisible_heads() -> None:
    model = _Stack(n=1, heads=6)
    with pytest.raises(ValueError, match="not divisible"):
        enable_sequence_parallel(model, sp_degree=4)


def test_report_shape() -> None:
    model = _Stack(n=2)
    state = enable_sequence_parallel(model, sp_degree=2)
    rep = sequence_parallel_report(state)
    assert rep == {
        "sp_degree": 2,
        "backend": "native-ulysses",
        "wrapped_blocks": 2,
        "head_parallel": True,
        "fused_rope_calls": 0,
        "complex_rope_calls": 0,
    }


def test_sequence_parallel_request_reset_clears_rope_execution_receipts() -> None:
    state = _SPState(
        sp_degree=2,
        fused_rope_calls=7,
        complex_rope_calls=9,
    )

    state.reset_request_window()

    assert state.fused_rope_calls == 0
    assert state.complex_rope_calls == 0


def test_rank_slice_freqs_shards_and_pads() -> None:
    freqs = torch.arange(8).view(8, 1, 1).to(torch.complex64)
    # sp=2, s_local=4: rank0 -> [0:4], rank1 -> [4:8]
    r0 = _rank_slice_freqs(freqs, sp_rank=0, s_local=4)
    r1 = _rank_slice_freqs(freqs, sp_rank=1, s_local=4)
    assert r0.shape[0] == 4 and r1.shape[0] == 4
    assert torch.equal(r0.view(-1).real, torch.arange(4).float())
    assert torch.equal(r1.view(-1).real, torch.arange(4, 8).float())


def test_rank_slice_pads_past_table() -> None:
    freqs = torch.arange(6).view(6, 1, 1).to(torch.complex64)
    # rank1 with s_local=4 wants [4:8] but table has only 6 -> pad 2 with ones.
    r1 = _rank_slice_freqs(freqs, sp_rank=1, s_local=4)
    assert r1.shape[0] == 4
    assert torch.equal(r1.view(-1).real[:2], torch.tensor([4.0, 5.0]))
    assert torch.equal(r1.view(-1).real[2:], torch.tensor([1.0, 1.0]))


def test_processor_fuses_qkv_exchange_and_preserves_selected_backend(
    monkeypatch,
) -> None:
    import torch.distributed as dist

    import worldfoundry.core.attention as attention_dispatch
    from worldfoundry.core.distributed import sequence_parallel_runtime

    module = SelfAttention(dim=32, num_heads=4).eval()
    module.attn.set_attention_backend("flash_attention_2")
    processor = SequenceParallelSelfAttentionProcessor(_SPState(sp_degree=2))
    calls: dict[str, object] = {"many": 0, "inverse": 0}

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(sequence_parallel_runtime, "get_sequence_parallel_group", lambda: None)

    def fake_many(values, group, scatter_dim, gather_dim, *, assume_even=False):
        del group
        assert (scatter_dim, gather_dim) == (2, 1)
        assert assume_even is True
        calls["many"] = int(calls["many"]) + 1
        return tuple(torch.cat([value[:, :, :2], value[:, :, :2]], dim=1) for value in values)

    def fake_inverse(value, group, scatter_dim, gather_dim, *, assume_even=False):
        del group
        assert (scatter_dim, gather_dim) == (1, 2)
        assert assume_even is True
        calls["inverse"] = int(calls["inverse"]) + 1
        return value[:, :4].repeat(1, 1, 2, 1)

    def fake_attention(q, k, v, num_heads, compatibility_mode=False, scale=None, backend=None):
        del k, v, scale
        calls["backend"] = backend
        calls["local_heads"] = num_heads
        calls["compatibility_mode"] = compatibility_mode
        return q

    monkeypatch.setattr(sequence_parallel_runtime, "all_to_all_4d_many", fake_many)
    monkeypatch.setattr(sequence_parallel_runtime, "all_to_all_4D", fake_inverse)
    monkeypatch.setattr(attention_dispatch, "packed_sequence_attention", fake_attention)

    x = torch.randn(1, 4, 32)
    freqs = torch.ones(8, 1, 4, dtype=torch.complex64)
    with torch.no_grad():
        output = processor(module, x, freqs)

    assert output.shape == x.shape
    assert calls == {
        "many": 1,
        "inverse": 1,
        "backend": "flash_attention_2",
        "local_heads": 2,
        "compatibility_mode": False,
    }


def test_processor_composes_fused_rope_with_rank_sequence_offset(
    monkeypatch,
) -> None:
    import torch.distributed as dist

    import worldfoundry.core.attention as attention_dispatch
    import worldfoundry.core.kernels as kernels
    from worldfoundry.core.distributed import sequence_parallel_runtime
    from worldfoundry.core.kernels.registry import _publish_dispatch_receipt

    module = SelfAttention(dim=32, num_heads=4).eval()
    fused_runtime = FusedRoPERuntimeState(installed_blocks=1)
    module._worldfoundry_fused_rope_runtime = fused_runtime
    state = _SPState(sp_degree=2)
    processor = SequenceParallelSelfAttentionProcessor(state)
    group = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda selected_group: 1)
    monkeypatch.setattr(
        sequence_parallel_runtime,
        "get_sequence_parallel_group",
        lambda: group,
    )

    def fake_fused_rope(q, k, q_weight, k_weight, table, **kwargs):
        del q_weight, k_weight, table
        calls.update(kwargs)
        _publish_dispatch_receipt(
            op="hidden_qk_rmsnorm_rope_3d",
            implementation="triton_hidden_qk_rmsnorm_rope_3d",
            backend="triton",
            accelerated=True,
            fallback=False,
            cache_hit=True,
            failures=[],
            quarantined=[],
        )
        return q, k

    def fake_many(values, selected_group, scatter_dim, gather_dim, *, assume_even=False):
        assert selected_group is group
        assert (scatter_dim, gather_dim, assume_even) == (2, 1, True)
        return tuple(
            torch.cat([value[:, :, :2], value[:, :, :2]], dim=1)
            for value in values
        )

    def fake_inverse(value, selected_group, scatter_dim, gather_dim, *, assume_even=False):
        assert selected_group is group
        assert (scatter_dim, gather_dim, assume_even) == (1, 2, True)
        return value[:, :4].repeat(1, 1, 2, 1)

    monkeypatch.setattr(kernels, "hidden_qk_rmsnorm_rope_3d", fake_fused_rope)
    monkeypatch.setattr(sequence_parallel_runtime, "all_to_all_4d_many", fake_many)
    monkeypatch.setattr(sequence_parallel_runtime, "all_to_all_4D", fake_inverse)
    monkeypatch.setattr(
        attention_dispatch,
        "packed_sequence_attention",
        lambda q, **kwargs: q,
    )

    x = torch.randn(1, 4, 32)
    freqs = torch.ones(8, 1, 4, dtype=torch.complex64)
    fused_table = torch.ones(8, 4, dtype=torch.complex64)
    with torch.no_grad():
        output = processor(
            module,
            x,
            freqs,
            _worldfoundry_rope_table=fused_table,
            _worldfoundry_rope_grid=(2, 2, 2),
        )

    assert output.shape == x.shape
    assert calls == {
        "num_heads": 4,
        "grid_size": (2, 2, 2),
        "eps": module.norm_q.eps,
        "sequence_offset": 4,
        "valid_tokens": 8,
    }
    assert state.fused_rope_calls == 1
    assert state.complex_rope_calls == 0
    assert fused_rope_runtime_report(fused_runtime) == {
        "installed_blocks": 1,
        "effective": "accelerated-provider-executed",
        "eager_calls": 1,
        "compiled_graph_traces": 0,
        "provider_calls": 1,
        "torch_fallback_calls": 0,
        "provider_failures": 0,
        "quarantined_skips": 0,
        "malformed_receipts": 0,
        "provider_paths": ["triton_hidden_qk_rmsnorm_rope_3d"],
        "last_dispatch": {
            "op": "hidden_qk_rmsnorm_rope_3d",
            "implementation": "triton_hidden_qk_rmsnorm_rope_3d",
            "backend": "triton",
            "accelerated": True,
            "fallback": False,
            "cache_hit": True,
            "failures": [],
            "quarantined": [],
            "reason": None,
        },
    }


def test_collective_counter_snapshot_is_copied_and_reset() -> None:
    from worldfoundry.core.distributed import sequence_parallel_runtime

    sequence_parallel_runtime.reset_sequence_parallel_collective_counters()
    snapshot = sequence_parallel_runtime.get_sequence_parallel_collective_counters()
    assert snapshot == {
        "all_to_all_calls": 0,
        "all_to_all_input_bytes": 0,
        "fused_multi_tensor_all_to_all_calls": 0,
        "unfused_multi_tensor_all_to_all_calls": 0,
        "shape_metadata_all_gather_calls": 0,
        "shape_metadata_cache_hits": 0,
        "sequence_output_all_gather_calls": 0,
    }
    snapshot["all_to_all_calls"] = 99
    assert sequence_parallel_runtime.get_sequence_parallel_collective_counters()[
        "all_to_all_calls"
    ] == 0


def test_audit_records_sp_without_quality_downgrade() -> None:
    applied = AppliedOptimizations()
    applied.record_sequence_parallel(
        requested=True,
        sp_degree=4,
        wrapped_blocks=30,
        backend="native-ulysses",
        head_parallel=True,
    )
    snap = applied.to_optimization_snapshot()
    assert snap.quality_tier == "exact"  # SP is numerically equivalent
    assert snap.requested["sequence_parallel"] == 4
    assert snap.effective["sequence_parallel_degree"] == 4
    assert snap.effective["sequence_parallel_backend"] == "native-ulysses"
    assert snap.fallbacks == ()


def test_audit_flags_indivisible_fallback() -> None:
    applied = AppliedOptimizations()
    applied.record_sequence_parallel(
        requested=True, sp_degree=4, wrapped_blocks=30, backend="xfuser-usp", head_parallel=False
    )
    snap = applied.to_optimization_snapshot()
    assert any("divisible" in f for f in snap.fallbacks)
