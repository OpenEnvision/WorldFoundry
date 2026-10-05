"""FastVideo VMoBA wiring, strict contracts, and request-window receipts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    SelfAttention,
    SelfAttentionProcessor,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    approximate_attention as approximate_module,
)
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    ApproximateSelfAttentionProcessor,
    advance_approximate_step,
    approximate_attention_lifecycle_report,
    approximate_attention_report,
    finalize_approximate_attention_request,
    install_approximate_attention,
    parse_approximate_attention,
    reset_approximate_attention,
)


class _WanBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SelfAttention(64, 1).eval()


class _WanStack(torch.nn.Module):
    def __init__(self, layers: int = 4) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_WanBlock() for _ in range(layers)])


class _CaptureVMoBAOps:
    def __init__(self) -> None:
        self.process_input_calls: list[dict[str, object]] = []
        self.attention_calls: list[dict[str, object]] = []
        self.process_output_calls: list[dict[str, object]] = []

    def process_moba_input(self, value, patch_resolution, chunk_layout):
        grid = tuple(patch_resolution)
        if isinstance(chunk_layout, int):
            chunk_size = chunk_layout * grid[1] * grid[2]
        elif len(chunk_layout) == 2:
            chunk_size = grid[0] * chunk_layout[0] * chunk_layout[1]
        else:
            chunk_size = chunk_layout[0] * chunk_layout[1] * chunk_layout[2]
        self.process_input_calls.append(
            {
                "value": value,
                "grid": grid,
                "chunk_layout": chunk_layout,
                "chunk_size": chunk_size,
            }
        )
        return value, chunk_size

    def moba_attn_varlen(self, q, k, v, **kwargs):
        self.attention_calls.append({"q": q, "k": k, "v": v, **kwargs})
        return q

    def process_moba_output(self, value, patch_resolution, chunk_layout):
        self.process_output_calls.append(
            {
                "value": value,
                "grid": tuple(patch_resolution),
                "chunk_layout": chunk_layout,
            }
        )
        return value


def _install(
    monkeypatch,
    *,
    layers: int = 4,
    ops: _CaptureVMoBAOps | None = None,
    **config_overrides,
):
    provider = _CaptureVMoBAOps() if ops is None else ops
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: provider)
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _WanStack(layers)
    test_profile = {
        "temporal_chunk_size": 2,
        "spatial_chunk_size": (3, 4),
        "st_chunk_size": (4, 6, 4),
    }
    test_profile.update(config_overrides)
    config = ApproximateAttentionConfig(
        kind="vmoba",
        first_full_step=0,
        **test_profile,
    )
    state = install_approximate_attention(model, config)
    return model, state, provider


def _run_processor(processor, attention, values, grid=(4, 6, 4)):
    with torch.no_grad():
        return processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )


def test_parser_accepts_pinned_fastvideo_vmoba_profile() -> None:
    config = parse_approximate_attention(
        {
            "kind": "vmoba",
            "temporal_chunk_size": 2,
            "temporal_topk": 3,
            "spatial_chunk_size": [3, 4],
            "spatial_topk": 20,
            "st_chunk_size": [4, 6, 4],
            "st_topk": 15,
            "moba_select_mode": "threshold",
            "moba_threshold": 0.25,
            "moba_threshold_type": "query_head",
            "first_full_layer": 0,
            "first_full_step": 12,
            "temporal_layer": 1,
            "spatial_layer": 1,
            "st_layer": 1,
        }
    )

    assert config.kind == "vmoba"
    assert config.spatial_chunk_size == (3, 4)
    assert config.st_chunk_size == (4, 6, 4)
    assert config.first_full_step == 12


def test_default_vmoba_profile_divides_wan22_ti2v_patch_grid() -> None:
    config = ApproximateAttentionConfig(kind="vmoba")
    assert config.temporal_chunk_size == 1
    assert config.spatial_chunk_size == (2, 5)
    assert config.st_chunk_size == (1, 2, 5)
    grid = (31, 22, 40)
    assert grid[0] % config.temporal_chunk_size == 0
    assert grid[1] % config.spatial_chunk_size[0] == 0
    assert grid[2] % config.spatial_chunk_size[1] == 0
    assert all(size % chunk == 0 for size, chunk in zip(grid, config.st_chunk_size))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("temporal_topk", 0, "positive integer"),
        ("spatial_chunk_size", [3, 0], "two positive"),
        ("moba_select_mode", "dense", "threshold.*topk"),
        ("moba_threshold", float("nan"), "finite"),
        ("first_full_step", -1, "non-negative"),
    ],
)
def test_parser_rejects_invalid_vmoba_profile(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_approximate_attention({"kind": "vmoba", field: value})


def test_install_fails_closed_before_mutation_when_provider_is_incomplete(
    monkeypatch,
) -> None:
    model = _WanStack(1)
    original = model.blocks[0].self_attn.get_processor()
    monkeypatch.setattr(
        approximate_module,
        "_load_sparse_ops",
        lambda: SimpleNamespace(moba_attn_varlen=lambda *args, **kwargs: None),
    )

    with pytest.raises(RuntimeError, match="process_moba_input.*process_moba_output"):
        install_approximate_attention(
            model,
            ApproximateAttentionConfig(kind="vmoba"),
        )

    assert model.blocks[0].self_attn.get_processor() is original
    assert isinstance(original, SelfAttentionProcessor)


def test_install_requires_blocks_index_path(monkeypatch) -> None:
    ops = _CaptureVMoBAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    attention = SelfAttention(64, 1)

    with pytest.raises(RuntimeError, match=r"blocks\.<index>"):
        install_approximate_attention(
            attention,
            ApproximateAttentionConfig(kind="vmoba"),
        )


def test_vmoba_routes_layer_cycle_through_all_three_provider_symbols(
    monkeypatch,
) -> None:
    model, state, ops = _install(monkeypatch)
    values = torch.arange(4 * 6 * 4 * 64, dtype=torch.float32).reshape(
        1, 4 * 6 * 4, 64
    )

    for block in model.blocks:
        processor = block.self_attn.get_processor()
        assert isinstance(processor, ApproximateSelfAttentionProcessor)
        output = _run_processor(processor, block.self_attn, values)
        torch.testing.assert_close(output, values)

    assert [call["chunk_layout"] for call in ops.process_input_calls[::3]] == [
        2,
        (3, 4),
        (4, 6, 4),
        2,
    ]
    assert len(ops.process_input_calls) == 12
    assert len(ops.attention_calls) == 4
    assert len(ops.process_output_calls) == 4
    assert ops.attention_calls[0]["cu_seqlens"].tolist() == [0, 96]
    assert ops.attention_calls[0]["max_seqlen"] == 96
    assert ops.attention_calls[0]["moba_topk"] == 2
    assert ops.attention_calls[1]["moba_topk"] == 2
    assert ops.attention_calls[2]["moba_topk"] == 1

    report = approximate_attention_report(state)
    assert report["runtime_effective"] is True
    assert report["kernel_attempts"] == 4
    assert report["sparse_calls"] == 4
    assert report["provider_path"] == "fastvideo_kernel.moba_attn_varlen"
    assert report["vmoba"]["chunk_calls"] == {
        "temporal": 2,
        "spatial": 1,
        "spatiotemporal": 1,
    }
    assert [event["layer_idx"] for event in report["vmoba"]["events"]] == [
        0,
        1,
        2,
        3,
    ]


def test_vmoba_dense_prefix_and_layer_prefix_do_not_call_provider(monkeypatch) -> None:
    ops = _CaptureVMoBAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _WanStack(2)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(
            kind="vmoba",
            first_full_step=1,
            first_full_layer=1,
            temporal_chunk_size=2,
            spatial_chunk_size=(3, 4),
            st_chunk_size=(4, 6, 4),
        ),
    )
    values = torch.randn(1, 96, 64)

    assert _run_processor(
        model.blocks[1].self_attn.get_processor(),
        model.blocks[1].self_attn,
        values,
    ) is None
    state.step = 1
    assert _run_processor(
        model.blocks[0].self_attn.get_processor(),
        model.blocks[0].self_attn,
        values,
    ) is None
    assert not ops.attention_calls
    assert state.scheduled_dense_calls == 2


def test_vmoba_missing_grid_and_invalid_chunk_divisibility_raise(monkeypatch) -> None:
    model, _, ops = _install(monkeypatch, layers=2)
    values = torch.randn(1, 96, 64)
    temporal = model.blocks[0].self_attn.get_processor()
    spatial = model.blocks[1].self_attn.get_processor()

    with pytest.raises(RuntimeError, match="3D patch grid"):
        with torch.no_grad():
            temporal._sparse_attention(
                model.blocks[0].self_attn,
                values,
                values,
                values,
                values,
            )
    with pytest.raises(RuntimeError, match="must be divisible"):
        _run_processor(
            spatial,
            model.blocks[1].self_attn,
            torch.randn(1, 5 * 4 * 5, 64),
            grid=(5, 4, 5),
        )
    assert not ops.attention_calls


def test_vmoba_malformed_provider_output_raises_without_false_receipt(
    monkeypatch,
) -> None:
    class _BadOutputOps(_CaptureVMoBAOps):
        def moba_attn_varlen(self, q, k, v, **kwargs):
            super().moba_attn_varlen(q, k, v, **kwargs)
            return q[:-1]

    ops = _BadOutputOps()
    model, state, _ = _install(monkeypatch, layers=1, ops=ops)
    values = torch.randn(1, 96, 64)

    with pytest.raises(RuntimeError, match="returned shape"):
        _run_processor(
            model.blocks[0].self_attn.get_processor(),
            model.blocks[0].self_attn,
            values,
        )

    report = approximate_attention_report(state)
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 0
    assert report["runtime_effective"] is False
    assert report["vmoba"]["events"] == []


def test_vmoba_reset_clears_only_current_request_receipts(monkeypatch) -> None:
    model, state, _ = _install(monkeypatch, layers=1)
    values = torch.randn(1, 96, 64)
    _run_processor(
        model.blocks[0].self_attn.get_processor(),
        model.blocks[0].self_attn,
        values,
    )
    assert approximate_attention_report(state)["vmoba"]["events"]

    reset_approximate_attention(state)

    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["sparse_calls"] == 0
    assert report["vmoba"]["events"] == []
    assert report["vmoba"]["chunk_calls"] == {
        "temporal": 0,
        "spatial": 0,
        "spatiotemporal": 0,
    }


@pytest.mark.parametrize(
    ("grid", "field"),
    [
        ((4, 5, 4), "spatial_chunk_size"),
        ((5, 6, 4), "st_chunk_size"),
    ],
)
def test_vmoba_preflights_full_grid_profile_before_temporal_provider(
    monkeypatch,
    grid,
    field,
) -> None:
    model, _, ops = _install(monkeypatch, layers=1)
    values = torch.randn(1, grid[0] * grid[1] * grid[2], 64)

    with pytest.raises(RuntimeError, match=field):
        _run_processor(
            model.blocks[0].self_attn.get_processor(),
            model.blocks[0].self_attn,
            values,
            grid=grid,
        )

    assert not ops.process_input_calls
    assert not ops.attention_calls


def test_vmoba_30_layer_cfg_receipt_has_exact_request_coverage(monkeypatch) -> None:
    model, state, _ = _install(
        monkeypatch,
        layers=30,
        temporal_chunk_size=1,
        spatial_chunk_size=(2, 5),
        st_chunk_size=(1, 2, 5),
    )
    grid = (31, 22, 40)
    sequence = grid[0] * grid[1] * grid[2]
    values = torch.randn(1, sequence, 64)

    for branch in ("positive", "negative"):
        advance_approximate_step(
            state,
            step=0,
            total_steps=1,
            request_id="wan22-request",
            branch=branch,
        )
        for block in model.blocks:
            _run_processor(
                block.self_attn.get_processor(),
                block.self_attn,
                values,
                grid=grid,
            )

    report = approximate_attention_report(state, "wan22-request")
    assert report["request_local"] is True
    assert report["expected_layers"] == list(range(30))
    assert report["wrapped_blocks"] == 30
    assert report["expected_calls"] == report["event_count"] == 60
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["event_totals_match_counters"] is True
    assert report["coverage"]["provider_contract_complete"] is True
    assert report["vmoba"]["chunk_calls"] == {
        "temporal": 20,
        "spatial": 20,
        "spatiotemporal": 20,
    }
    assert all(
        event["request_id"] == "wan22-request"
        and event["branch"] in {"positive", "negative"}
        and event["step"] == 0
        and event["provider_path"] == "fastvideo_kernel.moba_attn_varlen"
        and event["input_shape"] == [sequence, 1, 64]
        and event["output_shape"] == [sequence, 1, 64]
        and event["input_device"] == event["output_device"] == "cpu"
        and event["input_dtype"] == event["output_dtype"] == "torch.float32"
        and event["validated_grid"] == [31, 22, 40]
        for event in report["events"]
    )

    finalize_approximate_attention_request(state, "wan22-request")
    frozen = approximate_attention_report(state, "wan22-request")
    assert frozen["finalized"] is True
    assert frozen["release_reason"] == "completed"
    assert approximate_attention_lifecycle_report(state)["live_requests"] == 0


def test_vmoba_error_request_finalizes_and_releases_active_events(monkeypatch) -> None:
    class _BadOutputOps(_CaptureVMoBAOps):
        def moba_attn_varlen(self, q, k, v, **kwargs):
            super().moba_attn_varlen(q, k, v, **kwargs)
            return q[:-1]

    model, state, _ = _install(
        monkeypatch,
        layers=1,
        ops=_BadOutputOps(),
    )
    advance_approximate_step(
        state,
        step=0,
        total_steps=1,
        request_id="failed-vmoba",
        branch="positive",
    )
    with pytest.raises(RuntimeError, match="returned shape") as error:
        _run_processor(
            model.blocks[0].self_attn.get_processor(),
            model.blocks[0].self_attn,
            torch.randn(1, 96, 64),
        )

    finalize_approximate_attention_request(
        state,
        "failed-vmoba",
        error=error.value,
    )
    frozen = approximate_attention_report(state, "failed-vmoba")
    assert frozen["finalized"] is True
    assert frozen["completed"] is False
    assert frozen["release_reason"] == "error"
    assert frozen["error_type"] == "RuntimeError"
    assert frozen["events"][0]["execution"] == "error"
    assert approximate_attention_lifecycle_report(state) == {
        "live_requests": 0,
        "receipt_snapshots": 1,
        "max_receipt_snapshots": 32,
    }
