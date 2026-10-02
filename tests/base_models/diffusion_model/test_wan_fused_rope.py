"""CPU contracts for Wan's optional fused QK RMSNorm + 3D RoPE path."""

from __future__ import annotations

import copy

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    SelfAttention,
    WanModel,
    apply_wan_qk_norm_rope,
)
from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
    ApproximateAttentionConfig,
    install_approximate_attention,
)
from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
    FusedRoPERuntimeState,
    fused_rope_runtime_report,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
    fuse_qkv_projections,
)
from worldfoundry.core.attention import complex_rotary_frequencies_3d
from worldfoundry.core.attention.complex_rope import apply_complex_rotary_embedding
from worldfoundry.core.nn import RMSNorm


def _rope_inputs(
    *,
    dim: int = 96,
    heads: int = 4,
    grid: tuple[int, int, int] = (2, 3, 4),
) -> tuple[
    SelfAttention,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
]:
    torch.manual_seed(0)
    attention = SelfAttention(dim, heads).eval()
    sequence = grid[0] * grid[1] * grid[2]
    x = torch.randn(1, sequence, dim)
    head_dim = dim // heads
    base = complex_rotary_frequencies_3d(head_dim)
    assembled = torch.cat(
        (
            base[0][: grid[0]].view(grid[0], 1, 1, -1).expand(*grid, -1),
            base[1][: grid[1]].view(1, grid[1], 1, -1).expand(*grid, -1),
            base[2][: grid[2]].view(1, 1, grid[2], -1).expand(*grid, -1),
        ),
        dim=-1,
    ).reshape(sequence, 1, -1)
    fused_table = torch.cat(base, dim=-1)
    return attention, x, assembled, fused_table, grid


@pytest.mark.parametrize(
    ("precision", "rtol", "atol"),
    ((torch.complex128, 0.0, 0.0), (torch.complex64, 2e-6, 2e-6)),
)
def test_fused_qk_norm_rope_matches_complex_reference(
    precision: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    attention, x, freqs, fused_table, grid = _rope_inputs()
    q = attention.q(x)
    k = attention.k(x)

    expected_q, expected_k = apply_wan_qk_norm_rope(attention, q, k, freqs)
    actual_q, actual_k = apply_wan_qk_norm_rope(
        attention,
        q,
        k,
        freqs,
        fused_table=fused_table.to(precision),
        fused_grid=grid,
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


def test_fused_rope_composes_with_qkv_fusion() -> None:
    attention, x, freqs, fused_table, grid = _rope_inputs()
    fused_attention = copy.deepcopy(attention)
    assert fuse_qkv_projections(fused_attention) == 1

    kwargs = {
        "_worldfoundry_rope_table": fused_table,
        "_worldfoundry_rope_grid": grid,
    }
    with torch.no_grad():
        expected = attention(x, freqs, **kwargs)
        actual = fused_attention(x, freqs, **kwargs)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_fused_rope_composes_with_approximate_exact_fallback(monkeypatch) -> None:
    attention, x, freqs, fused_table, grid = _rope_inputs()
    approximate = copy.deepcopy(attention)
    monkeypatch.setattr(
        "worldfoundry.base_models.diffusion_model.optimizations.approximate_attention._load_sparse_ops",
        lambda: None,
    )
    state = install_approximate_attention(
        approximate,
        ApproximateAttentionConfig(kind="vsa", sparsity=0.9),
    )

    kwargs = {
        "_worldfoundry_rope_table": fused_table,
        "_worldfoundry_rope_grid": grid,
    }
    with torch.no_grad():
        expected = attention(x, freqs, **kwargs)
        actual = approximate(x, freqs, **kwargs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert state.dense_fallback_calls == 1
    assert state.kernel_fallbacks == 1


def test_fused_rope_requires_table_and_grid_together(monkeypatch) -> None:
    """A half-configured call must not silently select a different path."""

    attention, x, freqs, fused_table, grid = _rope_inputs()
    q = attention.q(x)
    k = attention.k(x)

    with pytest.raises(ValueError, match="together"):
        apply_wan_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=fused_table,
            fused_grid=None,
        )
    with pytest.raises(ValueError, match="together"):
        apply_wan_qk_norm_rope(
            attention,
            q,
            k,
            freqs,
            fused_table=None,
            fused_grid=grid,
        )


def test_fused_rope_eager_call_records_concrete_dispatch(monkeypatch) -> None:
    attention, x, freqs, fused_table, grid = _rope_inputs()
    state = FusedRoPERuntimeState(installed_blocks=1)
    attention._worldfoundry_fused_rope_runtime = state
    monkeypatch.delenv("WORLDFOUNDRY_KERNEL_BACKEND", raising=False)
    q = attention.q(x)
    k = attention.k(x)

    apply_wan_qk_norm_rope(
        attention,
        q,
        k,
        freqs,
        fused_table=fused_table,
        fused_grid=grid,
    )

    report = fused_rope_runtime_report(state)
    assert report["eager_calls"] == 1
    assert report["provider_calls"] + report["torch_fallback_calls"] == 1
    assert report["malformed_receipts"] == 0


def test_fused_rope_compile_trace_is_distinct_from_eager_receipt(
    monkeypatch,
) -> None:
    attention, x, freqs, fused_table, grid = _rope_inputs()
    state = FusedRoPERuntimeState(installed_blocks=1)
    attention._worldfoundry_fused_rope_runtime = state
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    apply_wan_qk_norm_rope(
        attention,
        attention.q(x),
        attention.k(x),
        freqs,
        fused_table=fused_table,
        fused_grid=grid,
    )

    report = fused_rope_runtime_report(state)
    assert report["compiled_graph_traces"] == 1
    assert report["eager_calls"] == 0


def test_wan_fused_rope_cache_is_precision_isolated_and_fp32_is_real() -> None:
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
        num_layers=0,
        has_image_input=False,
        require_vae_embedding=False,
    )

    fp32_table = model.fused_rope_table(device=torch.device("cpu"), precision="fp32")
    fp64_table = model.fused_rope_table(device=torch.device("cpu"), precision="fp64")

    assert fp32_table.dtype == torch.float32
    assert fp32_table.shape[-1] == 2
    assert fp64_table.dtype == torch.complex128
    assert fp32_table is model.fused_rope_table(
        device=torch.device("cpu"), precision="fp32"
    )
    assert fp64_table is model.fused_rope_table(
        device=torch.device("cpu"), precision="fp64"
    )


@pytest.mark.parametrize(
    ("compute_dtype", "complex_dtype"),
    ((torch.float32, torch.complex64), (torch.float64, torch.complex128)),
)
def test_complex_rope_uses_requested_intermediate_precision(
    monkeypatch,
    compute_dtype: torch.dtype,
    complex_dtype: torch.dtype,
) -> None:
    value = torch.randn(1, 4, 24)
    frequencies = torch.ones(4, 1, 3, dtype=torch.complex128)
    observed: list[torch.dtype] = []
    original = torch.view_as_complex

    def recording_view_as_complex(input: torch.Tensor) -> torch.Tensor:
        observed.append(input.dtype)
        result = original(input)
        assert result.dtype == complex_dtype
        return result

    monkeypatch.setattr(torch, "view_as_complex", recording_view_as_complex)
    output = apply_complex_rotary_embedding(
        value,
        frequencies,
        num_heads=4,
        compute_dtype=compute_dtype,
    )

    assert output.dtype == value.dtype
    assert observed == [compute_dtype]


def test_rms_norm_input_mode_uses_input_dtype_reduction(monkeypatch) -> None:
    value = torch.randn(2, 8, dtype=torch.bfloat16)
    observed: list[torch.dtype] = []
    original = torch.rsqrt

    def recording_rsqrt(input: torch.Tensor) -> torch.Tensor:
        observed.append(input.dtype)
        return original(input)

    monkeypatch.setattr(torch, "rsqrt", recording_rsqrt)
    norm = RMSNorm(8, compute_mode="input", dtype=torch.bfloat16)
    output = norm(value)

    assert observed == [torch.bfloat16]
    assert output.dtype == torch.bfloat16
    expected = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + norm.eps
    )
    torch.testing.assert_close(output, expected)

    observed.clear()
    norm = RMSNorm(8, compute_mode="fp32", dtype=torch.bfloat16)
    output = norm(value)
    assert observed == [torch.float32]
    assert output.dtype == torch.bfloat16
