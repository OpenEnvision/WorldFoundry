"""Tests for the opt-in, lossy approximate self-attention lane (STA/VSA).

These lock the seam contract without a GPU or the compiled fastvideo_kernel:
- the option parser normalizes str/mapping/config forms and rejects bad input;
- installing wraps every Wan SelfAttention with the approximate processor;
- with no kernel available (CPU), the processor falls back to EXACT attention
  and is bitwise-identical to the dense path — so landing the lane never
  changes results until a user is on Hopper with the kernel built;
- the dense-boundary schedule keeps first/last steps exact;
- reset/report telemetry behaves.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    SelfAttention,
    SelfAttentionProcessor,
    WanModel,
)
from worldfoundry.base_models.diffusion_model.optimizations import approximate_attention as approximate_module
from worldfoundry.base_models.diffusion_model.optimizations import sparse_mask_attention as lightx2v_module
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    ApproximateSelfAttentionProcessor,
    _recoverable_sparse_kernel_error,
    advance_approximate_step,
    approximate_attention_lifecycle_report,
    approximate_attention_report,
    fastvideo_sla_attention_report,
    finalize_approximate_attention_request,
    install_approximate_attention,
    lightx2v_attention_report,
    parse_approximate_attention,
    prepare_lightx2v_providers,
    reset_approximate_attention,
)
from worldfoundry.base_models.diffusion_model.optimizations.sparse_linear_attention import (
    PINNED_FASTVIDEO_COMMIT,
)
from worldfoundry.base_models.diffusion_model.optimizations.sparse_mask_attention import (
    PINNED_LIGHTX2V_COMMIT,
    LightX2VSparseAdapter,
    LightX2VSparseUnavailableError,
)


def _sa_and_inputs(dim: int = 256, heads: int = 8, seq: int = 128):
    torch.manual_seed(0)
    sa = SelfAttention(dim, heads).eval()
    x = torch.randn(1, seq, dim)
    head_dim = dim // heads
    ang = torch.randn(seq, 1, head_dim // 2)
    freqs = torch.polar(torch.ones_like(ang), ang)
    return sa, x, freqs


# ---- parser ----


def test_parse_string_kind() -> None:
    assert parse_approximate_attention("sta").kind == "sta"
    assert parse_approximate_attention("VSA").kind == "vsa"


@pytest.mark.parametrize(
    ("selected", "expected_kind", "expected_operator"),
    [
        ("dynamic", "dynamic_sparse", "triton"),
        ("sparge", "sparge", "meansim_sage2"),
        ("nbhd", "nbhd", "magi"),
        ("sla", "lightx2v_sla_mask", "triton"),
        ("flex-block", "flexblock", "flex_block"),
        ("sage_sla", "lightx2v_spas_sage", "sage2"),
    ],
)
def test_parse_lightx2v_kinds_and_defaults(
    selected: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    config = parse_approximate_attention(selected)
    assert config.kind == expected_kind
    assert config.lightx2v_operator == expected_operator


@pytest.mark.parametrize(
    ("selected", "expected_topk"),
    [("fastvideo_sla", 0.1), ("fastvideo-sagesla", 0.5)],
)
def test_parse_fastvideo_learned_sla_is_unambiguous(
    selected: str,
    expected_topk: float,
) -> None:
    config = parse_approximate_attention(selected)
    assert config.kind == selected.replace("-", "_")
    assert config.fastvideo_topk_ratio == expected_topk
    assert config.fastvideo_feature_map == "softmax"


def test_parse_mapping() -> None:
    cfg = parse_approximate_attention({"kind": "vsa", "sparsity": 0.8, "window": [4, 4, 4], "dense_steps": 3})
    assert cfg.kind == "vsa" and cfg.sparsity == 0.8 and cfg.window == (4, 4, 4) and cfg.dense_steps == 3


def test_parse_lightx2v_mapping_preserves_provider_settings() -> None:
    config = parse_approximate_attention(
        {
            "kind": "nbhd",
            "sparsity": 0.7,
            "lightx2v_operator": "flashinfer",
            "nbhd_coefficient": [1.0, 0.25],
            "nbhd_min_width": 2.0,
            "attnmap_frame_num": 2,
            "lightx2v_per_block_mean": False,
        }
    )
    assert config.kind == "nbhd"
    assert config.sparsity == 0.7
    assert config.lightx2v_operator == "flashinfer"
    assert config.nbhd_coefficient == (1.0, 0.25)
    assert config.nbhd_min_width == 2.0
    assert config.attnmap_frame_num == 2


def test_parse_passthrough_config() -> None:
    cfg = ApproximateAttentionConfig(kind="sta")
    assert parse_approximate_attention(cfg) is cfg


def test_parse_rejects_bad_type() -> None:
    with pytest.raises(TypeError):
        parse_approximate_attention(123)


def test_config_validates() -> None:
    with pytest.raises(ValueError):
        ApproximateAttentionConfig(kind="bogus")
    with pytest.raises(ValueError):
        ApproximateAttentionConfig(sparsity=1.0)
    with pytest.raises(ValueError):
        ApproximateAttentionConfig(dense_steps=-1)
    with pytest.raises(ValueError, match="three positive"):
        ApproximateAttentionConfig(window=(3, 3, 0))
    with pytest.raises(ValueError, match="three positive"):
        ApproximateAttentionConfig(window=(3, 3, 2.5))
    with pytest.raises(ValueError, match="non-negative integer"):
        ApproximateAttentionConfig(dense_steps=1.5)
    with pytest.raises(ValueError, match="supported volumes"):
        ApproximateAttentionConfig(block_tile=(2, 2, 2))


def test_parser_rejects_unknown_or_misnamed_options() -> None:
    with pytest.raises(ValueError, match="block_tile"):
        parse_approximate_attention({"kind": "vsa", "block_size": 64})
    with pytest.raises(ValueError, match="unknown"):
        parse_approximate_attention({"kind": "vsa", "typo": True})


# ---- install + fallback ----


def test_install_wraps_self_attention() -> None:
    sa, _, _ = _sa_and_inputs()
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa"))
    assert isinstance(sa.get_processor(), ApproximateSelfAttentionProcessor)
    assert state.wrapped_blocks == 1
    assert state.effective_kernel.startswith("exact")
    assert "VSA-QAT gate weights unavailable" in state.effective_kernel
    assert any("gate_compress" in note for note in state.notes)


def test_reinstall_is_idempotent() -> None:
    sa, _, _ = _sa_and_inputs()
    install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa"))
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="sta"))
    proc = sa.get_processor()
    assert isinstance(proc, ApproximateSelfAttentionProcessor)
    # inner is the original default processor, not a nested approximate one.
    assert isinstance(proc._inner, SelfAttentionProcessor)
    assert state.wrapped_blocks == 1


def test_cpu_fallback_is_bitwise_dense() -> None:
    # No fastvideo_kernel on CPU -> exact fallback -> identical to dense output.
    sa, x, freqs = _sa_and_inputs()
    with torch.no_grad():
        dense = sa(x, freqs)
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa", sparsity=0.9))
    with torch.no_grad():
        approx = sa(x, freqs)
    assert torch.equal(dense, approx)
    report = approximate_attention_report(state)
    assert report["kernel_fallbacks"] >= 1
    assert report["effective_kernel"].startswith("exact")


def test_dense_boundary_schedule() -> None:
    sa, x, freqs = _sa_and_inputs()
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa", dense_steps=2), total_steps=10)
    # step 0,1 -> dense boundary; the processor must not even attempt the kernel.
    assert state.is_dense_step() is True  # step 0
    advance_approximate_step(state)  # -> step 1
    assert state.is_dense_step() is True
    advance_approximate_step(state)  # -> step 2 (sparse region)
    assert state.is_dense_step() is False
    state.step = 9  # last step index (total_steps-1) within dense tail
    assert state.is_dense_step() is True


def test_reset_clears_counters() -> None:
    sa, x, freqs = _sa_and_inputs()
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa"))
    with torch.no_grad():
        sa(x, freqs)
    assert state.dense_fallback_calls >= 1
    reset_approximate_attention(state)
    assert state.step == 0 and state.dense_fallback_calls == 0 and state.kernel_fallbacks == 0


def test_runtime_report_exposes_grid_and_fallback_notes() -> None:
    sa, x, freqs = _sa_and_inputs(seq=64)
    state = install_approximate_attention(sa, ApproximateAttentionConfig(kind="vsa"))
    with torch.no_grad():
        sa(x, freqs, _worldfoundry_sparse_grid=(4, 4, 4))
    report = approximate_attention_report(state)
    assert report["grid_size"] == (4, 4, 4)
    assert any("gate_compress" in note for note in report["notes"])


class _CaptureVSAOps:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def video_sparse_attn(
        self,
        q,
        k,
        v,
        *,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        self.calls.append(
            {
                "layout": "bhsd",
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "variable_block_sizes": variable_block_sizes.detach().clone(),
                "q_variable_block_sizes": q_variable_block_sizes.detach().clone(),
                "topk": topk,
                "block_size": block_size,
                "compress_attn_weight": compress_attn_weight.detach().clone(),
            }
        )
        return q


class _CaptureVSA256Ops(_CaptureVSAOps):
    def video_sparse_attn(self, *args, **kwargs):
        raise AssertionError("256-token tiles must use the available BSHD provider")

    def video_sparse_attn_bshd(
        self,
        q,
        k,
        v,
        *,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        self.calls.append(
            {
                "layout": "bshd",
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "variable_block_sizes": variable_block_sizes.detach().clone(),
                "q_variable_block_sizes": q_variable_block_sizes.detach().clone(),
                "topk": topk,
                "block_size": block_size,
                "compress_attn_weight": compress_attn_weight.detach().clone(),
            }
        )
        return q


class _CaptureSTAOps:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sliding_tile_attention(self, q, k, v, window, *, text_length, has_text, seq_shape):
        self.calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "window": window,
                "text_length": text_length,
                "has_text": has_text,
                "seq_shape": seq_shape,
            }
        )
        return q


def _install_fake_vsa(monkeypatch, *, grid, tile=(4, 4, 4), sparsity=0.5, ops=None):
    ops = _CaptureVSAOps() if ops is None else ops
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(approximate_module, "_sparse_runtime_ineligibility", lambda q, head_dim: None)
    attention = SelfAttention(64, 1).eval()
    attention.gate_compress = torch.nn.Identity()
    state = install_approximate_attention(
        attention,
        ApproximateAttentionConfig(kind="vsa", sparsity=sparsity, block_tile=tile),
    )
    processor = attention.get_processor()
    assert isinstance(processor, ApproximateSelfAttentionProcessor)
    sequence = grid[0] * grid[1] * grid[2]
    values = torch.arange(sequence * 64, dtype=torch.float32).reshape(1, sequence, 64)
    return attention, processor, state, values, ops


def test_vsa_uses_true_3d_tile_pad_gate_and_untile_contract(monkeypatch) -> None:
    grid = (2, 5, 4)  # partial H boundary: block sizes are 32 and 8, padded to 128.
    attention, processor, state, values, ops = _install_fake_vsa(monkeypatch, grid=grid)

    before = approximate_attention_report(state)
    assert before["runtime_effective"] is False
    assert before["sparse_calls"] == 0
    assert before["effective_kernel"].startswith("exact")

    output = processor._sparse_attention(
        attention,
        values,
        values,
        values + 1,
        values + 2,
        _worldfoundry_sparse_grid=grid,
    )

    assert output is not None
    torch.testing.assert_close(output, values)
    assert len(ops.calls) == 1
    call = ops.calls[0]
    assert call["layout"] == "bhsd"
    assert tuple(call["q"].shape) == (1, 1, 128, 64)
    assert call["variable_block_sizes"].tolist() == [32, 8]
    assert torch.equal(call["variable_block_sizes"], call["q_variable_block_sizes"])
    assert call["topk"] == 1
    assert call["block_size"] == (4, 4, 4)
    # Identity gate projection must be tiled and passed to the provider, not dropped.
    torch.testing.assert_close(call["compress_attn_weight"], call["q"])
    # Fixed-size boundary slots are explicitly zero, never uninitialized garbage.
    assert torch.count_nonzero(call["q"][:, :, 32:64]) == 0
    assert torch.count_nonzero(call["q"][:, :, 72:]) == 0

    report = approximate_attention_report(state)
    assert report["runtime_effective"] is True
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 1
    assert report["kernel_fallbacks"] == 0
    assert report["effective_kernel"] == "vsa"
    assert report["provider_path"] == "fastvideo_kernel.video_sparse_attn"


def test_vsa_256_prefers_provider_bshd_fastpath(monkeypatch) -> None:
    grid = (4, 8, 8)
    ops = _CaptureVSA256Ops()
    attention, processor, state, values, _ = _install_fake_vsa(
        monkeypatch,
        grid=grid,
        tile=(4, 8, 8),
        ops=ops,
    )

    output = processor._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
        _worldfoundry_sparse_grid=grid,
    )

    assert output is not None
    assert ops.calls[0]["layout"] == "bshd"
    assert tuple(ops.calls[0]["q"].shape) == (1, 256, 1, 64)
    assert approximate_attention_report(state)["provider_path"] == "fastvideo_kernel.video_sparse_attn_bshd"


def test_vsa_missing_grid_is_exact_and_never_attempts_provider(monkeypatch) -> None:
    grid = (2, 4, 4)
    attention, processor, state, values, ops = _install_fake_vsa(monkeypatch, grid=grid)

    output = processor._sparse_attention(attention, values, values, values, values)

    assert output is None
    assert not ops.calls
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["kernel_attempts"] == 0
    assert report["sparse_calls"] == 0
    assert report["kernel_fallbacks"] == 1
    assert report["effective_kernel"].startswith("exact")


def test_vsa_malformed_gate_fails_fast_before_provider(monkeypatch) -> None:
    class _BadGate(torch.nn.Module):
        def forward(self, x):
            return x[..., :-1]

    grid = (2, 4, 4)
    attention, processor, state, values, ops = _install_fake_vsa(monkeypatch, grid=grid)
    attention.gate_compress = _BadGate()

    with pytest.raises(RuntimeError, match="gate_compress returned shape"):
        processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )
    assert not ops.calls
    assert state.kernel_attempts == 0
    assert state.sparse_calls == 0


def test_all_dense_schedule_never_becomes_runtime_effective(monkeypatch) -> None:
    grid = (2, 4, 4)
    ops = _CaptureVSAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(approximate_module, "_sparse_runtime_ineligibility", lambda q, head_dim: None)
    attention = SelfAttention(64, 1).eval()
    attention.gate_compress = torch.nn.Identity()
    state = install_approximate_attention(
        attention,
        ApproximateAttentionConfig(kind="vsa", dense_steps=1),
        total_steps=1,
    )
    processor = attention.get_processor()
    values = torch.randn(1, 32, 64)

    assert (
        processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )
        is None
    )
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["scheduled_dense_calls"] == 1
    assert report["kernel_attempts"] == 0
    assert report["kernel_fallbacks"] == 0
    assert report["effective_kernel"].startswith("exact")


def test_planned_dense_forward_is_not_counted_as_provider_fallback(monkeypatch) -> None:
    grid = (2, 4, 4)
    ops = _CaptureVSAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(approximate_module, "_sparse_runtime_ineligibility", lambda q, head_dim: None)
    attention = SelfAttention(64, 1).eval()
    attention.gate_compress = torch.nn.Identity()
    state = install_approximate_attention(
        attention,
        ApproximateAttentionConfig(kind="vsa", dense_steps=1),
        total_steps=3,
    )
    values = torch.randn(1, 32, 64)
    angles = torch.randn(32, 1, 32)
    freqs = torch.polar(torch.ones_like(angles), angles)

    with torch.no_grad():
        output = attention(
            values,
            freqs,
            _worldfoundry_sparse_grid=grid,
        )

    assert output.shape == values.shape
    assert not ops.calls
    report = approximate_attention_report(state)
    assert report["scheduled_dense_calls"] == 1
    assert report["dense_fallback_calls"] == 0
    assert report["kernel_fallbacks"] == 0


def test_provider_failure_is_distinct_from_scheduled_dense(monkeypatch) -> None:
    class _FailingVSAOps(_CaptureVSAOps):
        def video_sparse_attn(self, *args, **kwargs):
            raise ValueError("only supports a different provider shape")

    grid = (2, 4, 4)
    ops = _FailingVSAOps()
    attention, processor, state, values, _ = _install_fake_vsa(monkeypatch, grid=grid, ops=ops)

    assert (
        processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )
        is None
    )
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 0
    assert report["scheduled_dense_calls"] == 0
    assert report["kernel_fallbacks"] == 1
    assert report["effective_kernel"].startswith("exact")


def test_sta_requires_exact_3d_plan_and_reports_real_provider_call(monkeypatch) -> None:
    grid = (2, 4, 4)
    ops = _CaptureSTAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(approximate_module, "_sparse_runtime_ineligibility", lambda q, head_dim: None)
    attention = SelfAttention(64, 1).eval()
    state = install_approximate_attention(attention, ApproximateAttentionConfig(kind="sta", window=(3, 2, 1)))
    processor = attention.get_processor()
    values = torch.randn(1, 32, 64)

    # Same code path, but no tuned 3D plan: no provider function was attempted.
    assert (
        processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )
        is None
    )
    assert not ops.calls
    assert state.kernel_attempts == 0
    assert state.kernel_fallbacks == 1

    reset_approximate_attention(state)
    monkeypatch.setitem(approximate_module._STA_GRID_SHAPES, grid, "test-grid")
    output = processor._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
        _worldfoundry_sparse_grid=grid,
    )
    assert output is not None
    torch.testing.assert_close(output, values)
    assert ops.calls[0]["seq_shape"] == "test-grid"
    assert ops.calls[0]["window"] == [(3, 2, 1)]
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is True
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 1
    assert report["scheduled_dense_calls"] == 0
    assert report["kernel_fallbacks"] == 0
    assert report["effective_kernel"] == "sta"
    assert report["provider_path"] == "fastvideo_kernel.sliding_tile_attention"


class _ApproxReceiptBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SelfAttention(64, 1).eval()
        self.self_attn.gate_compress = torch.nn.Identity()


class _ApproxReceiptStack(torch.nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_ApproxReceiptBlock() for _ in range(layers)]
        )


class _ReferenceLightX2VProvider:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def apply(self, q, k, v, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, **kwargs})
        if self.error is not None:
            raise self.error
        return q


class _FakeFastVideoSLAAdapter(torch.nn.Module):
    def __init__(self, config, *, layer_specs, checkpoint_state_dict) -> None:
        super().__init__()
        assert checkpoint_state_dict
        self.config = config
        self.providers = torch.nn.ModuleDict(
            {str(spec.layer_index): torch.nn.Identity() for spec in layer_specs}
        )
        self.calls: list[int] = []

    def forward(
        self,
        layer_index,
        q,
        k,
        v,
        *,
        current_timestep,
        on_provider_attempt=None,
    ):
        del k, v
        if on_provider_attempt is not None:
            on_provider_attempt()
        self.calls.append(layer_index)
        provider_name = (
            "SLAAttentionImpl"
            if self.config.kind == "fastvideo_sla"
            else "SageSLAAttentionImpl"
        )
        provider_path = (
            "fastvideo.attention.backends.sla." + provider_name
        )
        provider_family = (
            "fastvideo/sparse-linear-attention"
            if self.config.kind == "fastvideo_sla"
            else "fastvideo/sage-sparse-linear-attention"
        )
        contract = {
            "shape": list(q.shape),
            "device": str(q.device),
            "dtype": str(q.dtype),
            "contiguous": q.is_contiguous(),
        }
        return SimpleNamespace(
            output=q,
            receipt={
                "algorithm": self.config.kind,
                "provider_family": provider_family,
                "provider_path": provider_path,
                "reference_provider_path": provider_path,
                "provider_fingerprint": "1" * 64,
                "injected_test_provider": False,
                "reference_fastvideo_commit": PINNED_FASTVIDEO_COMMIT,
                "provider_source_commit": PINNED_FASTVIDEO_COMMIT,
                "provider_source_clean": True,
                "provider_source_fingerprint": "2" * 64,
                "provider_source_root": "/test/FastVideo",
                "provider_source_file": "/test/FastVideo/sla.py",
                "reference_parity_verified": True,
                "checkpoint_layout": "fastvideo_diffusers",
                "projection_source_keys": ["weight", "bias"],
                "projection_weight_fingerprint": "3" * 64,
                "all_projection_weights_fingerprint": "4" * 64,
                "layer_index": layer_index,
                "provider_calls": 1,
                "current_timestep": current_timestep,
                "topk_ratio": self.config.topk_ratio,
                "feature_map": self.config.feature_map,
                "q": contract,
                "k": contract,
                "v": contract,
                "output": contract,
                "runtime_effective": True,
            },
        )


def test_fastvideo_learned_sla_requires_projection_checkpoint() -> None:
    model = _ApproxReceiptStack(1)
    with pytest.raises(RuntimeError, match="learned proj_l"):
        install_approximate_attention(
            model,
            ApproximateAttentionConfig(kind="fastvideo_sla"),
        )


def test_fastvideo_learned_sla_full_request_receipt(monkeypatch) -> None:
    monkeypatch.setattr(
        approximate_module,
        "FastVideoSLAAdapter",
        _FakeFastVideoSLAAdapter,
    )
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _ApproxReceiptStack(2)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind="fastvideo_sla"),
        checkpoint_state_dict={"projection": torch.ones(1)},
    )
    values = torch.randn(1, 32, 64)
    for step in range(2):
        for branch in ("positive", "negative"):
            _run_request_layer_events(
                state,
                model,
                values,
                request_id="fastvideo-sla-request",
                branch=branch,
                step=step,
                total_steps=2,
            )

    report = approximate_attention_report(state, "fastvideo-sla-request")
    learned = report["fastvideo_sla"]
    assert learned == fastvideo_sla_attention_report(
        state,
        "fastvideo-sla-request",
    )
    assert report["coverage"]["complete"] is True
    assert report["kernel_attempts"] == report["sparse_calls"] == 8
    assert report["dense_fallback_calls"] == report["kernel_fallbacks"] == 0
    assert learned["algorithm"] == "fastvideo_sla"
    assert learned["canonical_provider_executed"] is True
    assert learned["provider_source_commit"] == PINNED_FASTVIDEO_COMMIT
    assert learned["provider_source_clean"] is True
    assert learned["reference_parity_verified"] is True
    assert learned["provider_calls"] == learned["receipt_count"] == 8


def _install_fake_lightx2v(
    monkeypatch,
    *,
    kind="dynamic_sparse",
    layers=2,
    provider=None,
    config_overrides=None,
):
    provider = _ReferenceLightX2VProvider() if provider is None else provider
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    monkeypatch.setattr(
        approximate_module,
        "LightX2VSparseAdapter",
        lambda config: LightX2VSparseAdapter(
            config,
            allow_non_cuda_for_tests=True,
        ),
    )
    monkeypatch.setattr(
        lightx2v_module,
        "_build_provider",
        lambda config, frame_num: provider,
    )
    monkeypatch.setattr(
        lightx2v_module,
        "_provider_source_identity",
        lambda built_provider: {
            "provider_source_commit": PINNED_LIGHTX2V_COMMIT,
            "provider_source_clean": True,
            "provider_source_fingerprint": "test-reference-fingerprint",
            "provider_source_root": "/test/LightX2V",
        },
    )
    model = _ApproxReceiptStack(layers)
    overrides = {} if config_overrides is None else dict(config_overrides)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind=kind, sparsity=0.75, **overrides),
    )
    return model, state, torch.randn(1, 32, 64), provider


def test_load_preparation_materializes_without_runtime_credit(monkeypatch) -> None:
    provider = _ReferenceLightX2VProvider()
    monkeypatch.setattr(
        approximate_module,
        "LightX2VSparseAdapter",
        lambda config: LightX2VSparseAdapter(
            config,
            provider=provider,
            provider_path="lightx2v.fake.DynamicSparse.apply",
            allow_non_cuda_for_tests=True,
        ),
    )
    model = _ApproxReceiptStack(2)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind="dynamic_sparse"),
    )

    prepare_lightx2v_providers(model, state, fallback_device="cuda")

    assert state.lightx2v_load_preparation == {
        "attempted": True,
        "preflight_blocks": 2,
        "materialized_blocks": 2,
        "deferred_blocks": 0,
        "devices": ["cuda"],
    }
    assert provider.calls == []
    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["kernel_attempts"] == report["sparse_calls"] == 0
    assert report["lightx2v"]["load_preparation"] == (
        state.lightx2v_load_preparation
    )


def test_lightx2v_install_does_not_depend_on_fastvideo_kernel(monkeypatch) -> None:
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: None)
    attention = SelfAttention(64, 1).eval()

    state = install_approximate_attention(
        attention,
        ApproximateAttentionConfig(kind="dynamic_sparse"),
    )

    assert state.wrapped_blocks == 1
    assert state.install_reason is None
    assert not any("fastvideo_kernel" in note for note in state.notes)


def test_lightx2v_request_receipt_proves_provider_commit_and_full_coverage(
    monkeypatch,
) -> None:
    model, state, values, provider = _install_fake_lightx2v(monkeypatch)
    for step in range(2):
        for branch in ("positive", "negative"):
            _run_request_layer_events(
                state,
                model,
                values,
                request_id="lightx2v-request",
                branch=branch,
                step=step,
                total_steps=2,
            )

    report = approximate_attention_report(state, "lightx2v-request")
    assert report["runtime_effective"] is True
    assert report["coverage"]["complete"] is True
    assert report["expected_calls"] == report["event_count"] == 8
    assert report["kernel_attempts"] == report["sparse_calls"] == 8
    assert report["dense_fallback_calls"] == report["kernel_fallbacks"] == 0
    assert len(provider.calls) == 8
    lightx2v = report["lightx2v"]
    assert lightx2v == lightx2v_attention_report(state, "lightx2v-request")
    assert lightx2v["algorithm"] == "dynamic_sparse"
    assert lightx2v["provider_family"] == "lightx2v/dynamic-sparse"
    assert lightx2v["provider_families"] == ["lightx2v/dynamic-sparse"]
    assert lightx2v["operator"] == "triton"
    assert lightx2v["canonical_provider_executed"] is True
    assert lightx2v["reference_lightx2v_commit"] == PINNED_LIGHTX2V_COMMIT
    assert lightx2v["provider_source_commit"] == PINNED_LIGHTX2V_COMMIT
    assert lightx2v["provider_commit_complete"] is True
    assert lightx2v["provider_source_clean"] is True
    assert lightx2v["reference_parity_verified"] is True
    assert lightx2v["provider_calls"] == lightx2v["receipt_count"] == 8
    assert all(
        event["provider_source_commit"] == PINNED_LIGHTX2V_COMMIT
        and event["reference_parity_verified"] is True
        and event["runtime_effective"] is True
        for event in report["events"]
    )

    finalize_approximate_attention_request(state, "lightx2v-request")
    frozen = approximate_attention_report(state, "lightx2v-request")
    assert frozen["finalized"] is True
    assert frozen["completed"] is True
    assert frozen["lightx2v"]["reference_parity_verified"] is True


@pytest.mark.parametrize(
    ("kind", "config_overrides", "steps", "expected_dense_calls"),
    (
        ("draft_attn", {}, 1, 2),
        (
            "rainfusion_attn",
            {"lightx2v_skip_timesteps": 1},
            2,
            4,
        ),
    ),
)
def test_lightx2v_processor_reconciles_provider_dense_events(
    monkeypatch,
    kind,
    config_overrides,
    steps,
    expected_dense_calls,
) -> None:
    dense_path = "lightx2v.test.ReferenceDenseAttention.apply"

    def effective_provider_path(self, _provider, execution):
        if execution == "provider_dense":
            return dense_path
        return lightx2v_module.lightx2v_kernel_symbol(self.config)

    monkeypatch.setattr(
        LightX2VSparseAdapter,
        "_effective_provider_path",
        effective_provider_path,
    )
    model, state, values, _ = _install_fake_lightx2v(
        monkeypatch,
        kind=kind,
        layers=2,
        config_overrides=config_overrides,
    )
    request_id = f"{kind}-provider-dense-request"
    for step in range(steps):
        for branch in ("positive", "negative"):
            _run_request_layer_events(
                state,
                model,
                values,
                request_id=request_id,
                branch=branch,
                step=step,
                total_steps=steps,
            )

    report = approximate_attention_report(state, request_id)
    sparse_path = lightx2v_module.lightx2v_kernel_symbol(
        lightx2v_module.LightX2VSparseConfig(kind=kind)
    )
    assert report["coverage"]["complete"] is True
    assert report["provider_dense_calls"] == expected_dense_calls
    assert report["kernel_attempts"] == (
        report["sparse_calls"] + report["provider_dense_calls"]
    )
    assert report["event_count"] == (
        report["sparse_calls"] + report["provider_dense_calls"]
    )
    assert report["provider_path"] == sparse_path
    assert report["provider_paths"] == [sparse_path]
    assert report["provider_dense_path"] == dense_path
    assert report["provider_dense_paths"] == [dense_path]
    provider_dense_events = [
        event
        for event in report["events"]
        if event["execution"] == "provider_dense"
    ]
    sparse_events = [
        event for event in report["events"] if event["execution"] == "sparse"
    ]
    assert len(provider_dense_events) == expected_dense_calls
    assert len(sparse_events) == report["sparse_calls"]
    assert all(
        event["provider_path"] == dense_path
        and event["canonical_sparse_provider_path"] == sparse_path
        and event["provider_attempted"] is True
        and event["sparse_kernel_executed"] is False
        and event["provider_dense_executed"] is True
        for event in provider_dense_events
    )
    assert all(
        event["provider_path"] == sparse_path
        and event["canonical_sparse_provider_path"] == sparse_path
        and event["sparse_kernel_executed"] is True
        and event["provider_dense_executed"] is False
        for event in sparse_events
    )
    lightx2v = report["lightx2v"]
    assert lightx2v["sparse_event_count"] == len(sparse_events)
    assert lightx2v["provider_dense_event_count"] == expected_dense_calls
    assert lightx2v["provider_dense_events"] == provider_dense_events
    assert lightx2v["provider_calls"] == report["kernel_attempts"]


def test_lightx2v_provider_error_fails_closed_and_finalizes_request(
    monkeypatch,
) -> None:
    provider = _ReferenceLightX2VProvider(
        error=LightX2VSparseUnavailableError("provider unavailable")
    )
    model, state, values, _ = _install_fake_lightx2v(
        monkeypatch,
        layers=1,
        provider=provider,
    )
    advance_approximate_step(
        state,
        step=0,
        total_steps=1,
        request_id="lightx2v-error",
        branch="positive",
    )
    processor = model.blocks[0].self_attn.get_processor()
    with pytest.raises(LightX2VSparseUnavailableError, match="provider unavailable") as caught:
        processor._sparse_attention(
            model.blocks[0].self_attn,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=(2, 4, 4),
        )

    report = approximate_attention_report(state, "lightx2v-error")
    assert report["runtime_effective"] is False
    assert report["kernel_attempts"] == 1
    assert report["sparse_calls"] == 0
    assert report["dense_fallback_calls"] == report["kernel_fallbacks"] == 0
    assert report["events"][0]["execution"] == "error"
    assert report["events"][0]["provider_attempted"] is True
    finalize_approximate_attention_request(
        state,
        "lightx2v-error",
        error=caught.value,
    )
    frozen = approximate_attention_report(state, "lightx2v-error")
    assert frozen["completed"] is False
    assert frozen["release_reason"] == "error"
    assert frozen["error_type"] == "LightX2VSparseUnavailableError"
    assert approximate_attention_lifecycle_report(state)["live_requests"] == 0


def test_lightx2v_cpu_contract_is_an_error_not_a_dense_fallback() -> None:
    attention = SelfAttention(64, 1).eval()
    state = install_approximate_attention(
        attention,
        ApproximateAttentionConfig(kind="sla"),
    )
    processor = attention.get_processor()
    values = torch.randn(1, 32, 64)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=(2, 4, 4),
        )

    report = approximate_attention_report(state)
    assert report["runtime_effective"] is False
    assert report["kernel_attempts"] == report["sparse_calls"] == 0
    assert report["kernel_fallbacks"] == report["dense_fallback_calls"] == 0
    assert report["events"][0]["execution"] == "error"


def _run_request_layer_events(
    state,
    model,
    values,
    *,
    request_id,
    branch,
    step,
    total_steps,
    grid=(2, 4, 4),
):
    advance_approximate_step(
        state,
        step=step,
        total_steps=total_steps,
        request_id=request_id,
        branch=branch,
    )
    for block in model.blocks:
        processor = block.self_attn.get_processor()
        output = processor._sparse_attention(
            block.self_attn,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        )
        assert output is not None


def test_request_receipts_isolate_interleaved_requests_and_finalize(monkeypatch) -> None:
    ops = _CaptureVSAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _ApproxReceiptStack(2)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind="vsa", sparsity=0.5),
    )
    values = torch.randn(1, 32, 64)

    # A0/B0/A1/B1 deliberately interleave and B starts on the negative CFG
    # branch. Ownership must not depend on positive-first or global step state.
    for request_id, branch, step in (
        ("request-a", "positive", 0),
        ("request-b", "negative", 0),
        ("request-a", "positive", 1),
        ("request-b", "negative", 1),
    ):
        _run_request_layer_events(
            state,
            model,
            values,
            request_id=request_id,
            branch=branch,
            step=step,
            total_steps=2,
        )

    report_a = approximate_attention_report(state, "request-a")
    report_b = approximate_attention_report(state, "request-b")
    assert report_a["request_epoch"] != report_b["request_epoch"]
    assert report_a["coverage"]["complete"] is True
    assert report_b["coverage"]["complete"] is True
    assert report_a["expected_layers"] == [0, 1]
    assert report_a["expected_calls"] == report_a["event_count"] == 4
    assert {event["request_id"] for event in report_a["events"]} == {
        "request-a"
    }
    assert {event["branch"] for event in report_b["events"]} == {"negative"}
    assert all(
        event["provider_path"] == "fastvideo_kernel.video_sparse_attn"
        and event["input_shape"] == event["output_shape"]
        and event["input_device"] == event["output_device"]
        and event["input_dtype"] == event["output_dtype"]
        for event in report_a["events"]
    )

    finalize_approximate_attention_request(state, "request-a")
    frozen_a = approximate_attention_report(state, "request-a")
    assert frozen_a["finalized"] is True
    assert frozen_a["completed"] is True
    assert frozen_a["release_reason"] == "completed"
    assert approximate_attention_lifecycle_report(state) == {
        "live_requests": 1,
        "receipt_snapshots": 1,
        "max_receipt_snapshots": 32,
    }
    finalize_approximate_attention_request(state, "request-b")
    assert approximate_attention_lifecycle_report(state)["live_requests"] == 0


def test_finalize_releases_invocation_ledger_and_selects_frozen_receipt(
    monkeypatch,
) -> None:
    grid = (2, 4, 4)
    attention, processor, state, values, _ = _install_fake_vsa(
        monkeypatch,
        grid=grid,
    )
    advance_approximate_step(
        state,
        step=0,
        total_steps=1,
        request_id="released-request",
        branch="positive",
    )
    assert processor._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
        _worldfoundry_sparse_grid=grid,
    ) is not None
    invocation = state.current_invocation()
    assert invocation is not None
    request_state = invocation.request_state
    assert request_state.events

    finalize_approximate_attention_request(state, "released-request")

    assert state.current_invocation() is None
    assert "released-request" not in state._requests
    assert state._last_finalized_request_id.get() == "released-request"
    assert isinstance(state._receipt_snapshots["released-request"], str)
    assert all(
        value is not request_state
        for value in copy_context().values()
    )
    frozen = approximate_attention_report(state)
    assert frozen["request_id"] == "released-request"
    assert frozen["finalized"] is True
    assert frozen["completed"] is True
    assert frozen["event_count"] == 1


def test_request_context_isolation_survives_parallel_forwards(monkeypatch) -> None:
    ops = _CaptureVSAOps()
    monkeypatch.setattr(approximate_module, "_load_sparse_ops", lambda: ops)
    monkeypatch.setattr(
        approximate_module,
        "_sparse_runtime_ineligibility",
        lambda q, head_dim: None,
    )
    model = _ApproxReceiptStack(2)
    state = install_approximate_attention(
        model,
        ApproximateAttentionConfig(kind="vsa"),
    )
    values = torch.randn(1, 32, 64)
    barrier = Barrier(2)

    def run_request(request_id, branch):
        for step in range(2):
            advance_approximate_step(
                state,
                step=step,
                total_steps=2,
                request_id=request_id,
                branch=branch,
            )
            # Both threads have selected different context-local owners before
            # either shared processor records a layer event.
            barrier.wait()
            for block in model.blocks:
                processor = block.self_attn.get_processor()
                assert processor._sparse_attention(
                    block.self_attn,
                    values,
                    values,
                    values,
                    values,
                    _worldfoundry_sparse_grid=(2, 4, 4),
                ) is not None
            barrier.wait()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(run_request, "parallel-a", "positive"),
            pool.submit(run_request, "parallel-b", "negative"),
        )
        for future in futures:
            future.result()

    for request_id, branch in (
        ("parallel-a", "positive"),
        ("parallel-b", "negative"),
    ):
        report = approximate_attention_report(state, request_id)
        assert report["coverage"]["complete"] is True
        assert {event["request_id"] for event in report["events"]} == {
            request_id
        }
        assert {event["branch"] for event in report["events"]} == {branch}


def test_failed_request_does_not_pollute_later_sparse_success(monkeypatch) -> None:
    grid = (2, 4, 4)
    attention, processor, state, values, _ = _install_fake_vsa(
        monkeypatch,
        grid=grid,
    )

    advance_approximate_step(
        state,
        step=0,
        total_steps=1,
        request_id="fallback-request",
        branch="positive",
    )
    assert processor._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
    ) is None
    fallback = approximate_attention_report(state, "fallback-request")
    assert fallback["kernel_fallbacks"] == 1
    assert fallback["runtime_effective"] is False
    finalize_approximate_attention_request(state, "fallback-request")

    advance_approximate_step(
        state,
        step=0,
        total_steps=1,
        request_id="successful-request",
        branch="positive",
    )
    assert processor._sparse_attention(
        attention,
        values,
        values,
        values,
        values,
        _worldfoundry_sparse_grid=grid,
    ) is not None
    success = approximate_attention_report(state, "successful-request")
    assert success["runtime_effective"] is True
    assert success["effective_kernel"] == "vsa"
    assert success["kernel_fallbacks"] == 0
    assert success["notes"] == []


def test_routed_expert_receipt_accepts_unique_global_step_subset(monkeypatch) -> None:
    grid = (2, 4, 4)
    attention, processor, state, values, _ = _install_fake_vsa(
        monkeypatch,
        grid=grid,
    )
    for step in (4, 7):
        advance_approximate_step(
            state,
            step=step,
            total_steps=10,
            request_id="dual-expert-request",
            branch="positive",
            routed_steps=True,
        )
        assert processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        ) is not None

    report = approximate_attention_report(state, "dual-expert-request")
    assert report["routed_steps"] is True
    assert report["expected_calls"] == report["event_count"] == 2
    assert report["branches"]["positive"]["expected_steps"] == [4, 7]
    assert report["coverage"]["complete"] is True
    with pytest.raises(ValueError, match="unique and increasing"):
        advance_approximate_step(
            state,
            step=7,
            total_steps=10,
            request_id="dual-expert-request",
            branch="positive",
            routed_steps=True,
        )


def test_request_receipt_snapshots_are_bounded(monkeypatch) -> None:
    grid = (2, 4, 4)
    attention, processor, state, values, _ = _install_fake_vsa(
        monkeypatch,
        grid=grid,
    )
    for index in range(35):
        request_id = f"request-{index}"
        advance_approximate_step(
            state,
            step=0,
            total_steps=1,
            request_id=request_id,
            branch="positive",
        )
        assert processor._sparse_attention(
            attention,
            values,
            values,
            values,
            values,
            _worldfoundry_sparse_grid=grid,
        ) is not None
        finalize_approximate_attention_request(state, request_id)

    assert approximate_attention_lifecycle_report(state) == {
        "live_requests": 0,
        "receipt_snapshots": 32,
        "max_receipt_snapshots": 32,
    }
    assert "request-0" not in state._receipt_snapshots
    assert approximate_attention_report(state, "request-34")["finalized"] is True


def test_processor_uses_one_fused_qkv_projection_and_keeps_sparse_dispatch(
    monkeypatch,
) -> None:
    class _CountingQKV(torch.nn.Linear):
        def __init__(self, dim: int) -> None:
            super().__init__(dim, 3 * dim)
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return super().forward(value)

    grid = (2, 4, 4)
    attention, _, state, values, ops = _install_fake_vsa(
        monkeypatch,
        grid=grid,
    )
    processor = attention.get_processor()
    assert processor.supports_fused_qkv is True
    fused = _CountingQKV(64)
    with torch.no_grad():
        fused.weight.copy_(
            torch.cat(
                (attention.q.weight, attention.k.weight, attention.v.weight),
                dim=0,
            )
        )
        fused.bias.copy_(
            torch.cat(
                (attention.q.bias, attention.k.bias, attention.v.bias),
                dim=0,
            )
        )
    attention.qkv = fused
    del attention.q, attention.k, attention.v
    angles = torch.randn(values.shape[1], 1, 32)
    freqs = torch.polar(torch.ones_like(angles), angles)

    with torch.no_grad():
        output = attention(
            values,
            freqs,
            _worldfoundry_sparse_grid=grid,
        )

    assert output.shape == values.shape
    assert fused.calls == 1
    assert len(ops.calls) == 1
    assert approximate_attention_report(state)["sparse_calls"] == 1


def test_lightx2v_processor_composes_with_one_fused_qkv_projection(
    monkeypatch,
) -> None:
    class _CountingQKV(torch.nn.Linear):
        def __init__(self, dim: int) -> None:
            super().__init__(dim, 3 * dim)
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return super().forward(value)

    model, state, values, provider = _install_fake_lightx2v(
        monkeypatch,
        kind="sla",
        layers=1,
    )
    attention = model.blocks[0].self_attn
    fused = _CountingQKV(64)
    with torch.no_grad():
        fused.weight.copy_(
            torch.cat(
                (attention.q.weight, attention.k.weight, attention.v.weight),
                dim=0,
            )
        )
        fused.bias.copy_(
            torch.cat(
                (attention.q.bias, attention.k.bias, attention.v.bias),
                dim=0,
            )
        )
    attention.qkv = fused
    del attention.q, attention.k, attention.v
    angles = torch.randn(values.shape[1], 1, 32)
    freqs = torch.polar(torch.ones_like(angles), angles)

    with torch.no_grad():
        output = attention(
            values,
            freqs,
            _worldfoundry_sparse_grid=(2, 4, 4),
        )

    assert output.shape == values.shape
    assert fused.calls == 1
    assert len(provider.calls) == 1
    report = approximate_attention_report(state)
    assert report["sparse_calls"] == 1
    assert report["lightx2v"]["canonical_provider_executed"] is True



class _CaptureSelfProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, attention, x, freqs, **kwargs):
        del attention, freqs
        self.kwargs = dict(kwargs)
        return torch.zeros_like(x)


class _CaptureCrossProcessor:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def __call__(self, attention, x, context, **kwargs):
        del attention, context
        self.kwargs = dict(kwargs)
        return torch.zeros_like(x)


def test_dit_block_splits_internal_self_attention_kwargs_from_cross_kwargs() -> None:
    block = DiTBlock(False, dim=32, num_heads=4, ffn_dim=64).eval()
    self_capture = _CaptureSelfProcessor()
    cross_capture = _CaptureCrossProcessor()
    block.self_attn.set_processor(self_capture)
    block.cross_attn.set_processor(cross_capture)
    hidden = torch.randn(1, 8, 32)
    context = torch.randn(1, 3, 32)
    t_mod = torch.randn(1, 6, 32)
    rope_table = torch.randn(4, 8)
    memory_context = object()

    with torch.no_grad():
        block(
            hidden,
            context,
            t_mod,
            torch.randn(8, 1, 4, dtype=torch.complex64),
            _worldfoundry_sparse_grid=(2, 2, 2),
            _worldfoundry_rope_grid=(2, 2, 2),
            _worldfoundry_rope_table=rope_table,
            memory_context=memory_context,
            cross_scale=0.5,
        )

    assert self_capture.kwargs == {
        "_worldfoundry_sparse_grid": (2, 2, 2),
        "_worldfoundry_rope_grid": (2, 2, 2),
        "_worldfoundry_rope_table": rope_table,
    }
    assert cross_capture.kwargs == {"memory_context": memory_context, "cross_scale": 0.5}


def test_tiny_wan_routes_current_grid_and_fused_rope_to_self_attention() -> None:
    model = WanModel(
        dim=96,
        in_dim=4,
        ffn_dim=192,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        require_vae_embedding=False,
    ).eval()
    capture = _CaptureSelfProcessor()
    model.blocks[0].self_attn.set_processor(capture)
    model._worldfoundry_approximate_attention = SimpleNamespace()
    model._worldfoundry_fused_rope = True
    model._worldfoundry_rope_precision = "fp32"

    with torch.no_grad():
        output = model(
            x=torch.randn(1, 4, 2, 4, 4),
            timestep=torch.tensor([10.0]),
            context=torch.randn(1, 5, 32),
        )

    assert output.shape == (1, 4, 2, 4, 4)
    assert capture.kwargs["_worldfoundry_sparse_grid"] == (2, 2, 2)
    assert capture.kwargs["_worldfoundry_rope_grid"] == (2, 2, 2)
    assert isinstance(capture.kwargs["_worldfoundry_rope_table"], torch.Tensor)


def test_sparse_kernel_oom_and_unknown_errors_are_not_hidden() -> None:
    assert _recoverable_sparse_kernel_error(RuntimeError("kernel only supports head dim 128"))
    assert not _recoverable_sparse_kernel_error(RuntimeError("CUDA out of memory"))
    assert not _recoverable_sparse_kernel_error(RuntimeError("invalid tensor stride"))
