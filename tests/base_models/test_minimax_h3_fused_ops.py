"""Stage 8 tests: fused Triton AdaLN kernels match the PyTorch fallback (bf16)."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import ops

_CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _CUDA, reason="fused Triton kernels require CUDA")


def _fallback_scale_shift(x, shift, scale, idx):
    return (x * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)).to(x.dtype)


def _fallback_gate(x, gate, other, idx):
    return (x + gate.index_select(0, idx) * other).to(x.dtype)


def test_probe_true_for_bf16_cuda_contiguous() -> None:
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    assert ops.can_use_fused_indexed_modulation(x, s, s) is True
    # fp32 or CPU disqualifies.
    assert ops.can_use_fused_indexed_modulation(x.float(), s.float(), s.float()) is False


def test_fused_scale_shift_matches_fallback() -> None:
    torch.manual_seed(0)
    rows, hidden = 32, 256
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(3, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(3, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(0, 3, (rows,), device="cuda")

    fused = ops.indexed_scale_shift(x, shift, scale, idx, dtype=torch.bfloat16)
    ref = _fallback_scale_shift(x, shift, scale, idx)
    # bf16 rounding: kernel matches the reference boundary to within one ulp.
    assert torch.allclose(fused.float(), ref.float(), atol=2e-2, rtol=2e-2)
    assert fused.dtype == torch.bfloat16


def test_fused_gate_matches_fallback() -> None:
    torch.manual_seed(0)
    rows, hidden = 32, 256
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(3, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    other = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 3, (rows,), device="cuda")

    fused = ops.indexed_gate(x, gate, other, idx, dtype=torch.bfloat16)
    ref = _fallback_gate(x, gate, other, idx)
    assert torch.allclose(fused.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_fused_does_not_mutate_input() -> None:
    rows, hidden = 16, 128
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    x_before = x.clone()
    shift = torch.zeros(2, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.zeros(2, hidden, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(rows, dtype=torch.long, device="cuda")
    ops.indexed_scale_shift(x, shift, scale, idx, dtype=torch.bfloat16)
    assert torch.equal(x, x_before)  # functional contract: input untouched


def _make_rope_cache(seq_len: int, rot_dim: int = 96, inv_freq_len: int = 16) -> torch.Tensor:
    inv_freq = torch.rand(inv_freq_len, device="cuda")
    pos = torch.rand(seq_len, 3, device="cuda")
    per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
    freqs = torch.cat((half, half), dim=-1)  # [S, 96]
    h = freqs.shape[-1] // 2
    return torch.cat((torch.cos(freqs[:, :h]), torch.sin(freqs[:, :h])), dim=-1).to(torch.bfloat16)


def test_probe_qknorm_rope_eligibility() -> None:
    q = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16)
    cache = _make_rope_cache(5)
    assert ops.can_use_fused_qknorm_rope(q, q.clone(), cache) is True
    # unsupported head_dim
    q_bad = torch.randn(5, 2, 100, device="cuda", dtype=torch.bfloat16)
    assert ops.can_use_fused_qknorm_rope(q_bad, q_bad.clone(), cache) is False


def test_fused_qknorm_rope_matches_reference() -> None:
    torch.manual_seed(0)
    seq_len, num_heads, head_dim = 5, 2, 128
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).cuda()
    k_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).cuda()
    cache = _make_rope_cache(seq_len)
    positions = torch.arange(seq_len, device="cuda")

    fused_q, fused_k = ops.qk_norm_rope(q, k, q_norm, k_norm, cache, positions)
    # Reference: force the fallback by disabling the fused backend.
    ref_q, ref_k = ops.apply_qk_norm(q, k, q_norm, k_norm, head_dim)
    ref_q, ref_k = ops.apply_rope_qk(ref_q, ref_k, cache, positions)

    assert fused_q.shape == q.shape and fused_k.shape == k.shape
    assert torch.allclose(fused_q.float(), ref_q.float(), atol=3e-2, rtol=3e-2)
    assert torch.allclose(fused_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    # Passthrough tail (dims beyond rot_dim) equals the normalized value.
    rot_dim = cache.shape[-1]
    assert torch.allclose(fused_q[..., rot_dim:].float(), ref_q[..., rot_dim:].float(), atol=3e-2)
