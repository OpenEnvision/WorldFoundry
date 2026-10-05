"""Stage 0 sanity tests for MiniMax H3 portable fallback operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import ops


def test_fused_probes_reject_cpu_and_non_bf16() -> None:
    # Fused Triton kernels require CUDA bf16; CPU / empty always falls back.
    assert ops.can_use_fused_indexed_modulation() is False
    x = torch.randn(4, 128)  # cpu fp32
    assert ops.can_use_fused_indexed_modulation(x, x, x) is False
    q = torch.randn(4, 2, 128)  # cpu
    cache = torch.randn(4, 96)
    assert ops.can_use_fused_qknorm_rope(q, q, cache) is False


def test_silu_mul_matches_reference() -> None:
    hidden = torch.randn(4, 16)
    gate, up = hidden.chunk(2, dim=-1)
    assert torch.allclose(ops.silu_mul(hidden), F.silu(gate) * up, atol=1e-5)


def test_indexed_scale_shift_matches_reference() -> None:
    x = torch.randn(6, 8)
    scale = torch.randn(3, 8)
    shift = torch.randn(3, 8)
    idx = torch.tensor([0, 1, 2, 0, 1, 2])
    expected = x * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
    assert torch.allclose(ops.indexed_scale_shift(x, shift, scale, idx, dtype=x.dtype), expected)


def test_indexed_gate_matches_reference() -> None:
    x = torch.randn(6, 8)
    gate = torch.randn(3, 8)
    other = torch.randn(6, 8)
    idx = torch.tensor([0, 1, 2, 0, 1, 2])
    expected = x + gate.index_select(0, idx) * other
    assert torch.allclose(ops.indexed_gate(x, gate, other, idx, dtype=x.dtype), expected)


def test_apply_qk_norm_preserves_shape_and_normalizes() -> None:
    q_norm = ops.make_rms_norm(8, eps=1e-5, dtype=torch.float32)
    k_norm = ops.make_rms_norm(8, eps=1e-5, dtype=torch.float32)
    q = torch.randn(5, 2, 8)
    k = torch.randn(5, 2, 8)
    q_out, k_out = ops.apply_qk_norm(q, k, q_norm, k_norm, head_dim=8)
    assert q_out.shape == q.shape and k_out.shape == k.shape
    assert torch.allclose(q_out, q_norm(q))


def _build_freqs(seq_len: int, inv_freq_len: int = 16) -> torch.Tensor:
    inv_freq = torch.rand(inv_freq_len)
    pos = torch.rand(seq_len, 3)
    per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    return torch.cat((half, half), dim=-1)


def test_apply_rope_qk_rotates_prefix_and_passes_tail() -> None:
    seq_len, head_dim = 5, 128
    freqs = _build_freqs(seq_len)  # [S, 96]
    cache = ops.rope_cos_sin_cache(freqs, dtype=torch.float32)
    q = torch.randn(seq_len, 2, head_dim)
    k = torch.randn(seq_len, 2, head_dim)
    q_out, k_out = ops.apply_rope_qk(q, k, cache, torch.arange(seq_len))
    rot_dim = cache.shape[-1]
    assert q_out.shape == q.shape and k_out.shape == k.shape
    # Head dims beyond rot_dim are passed through unchanged.
    assert torch.allclose(q_out[..., rot_dim:], q[..., rot_dim:])
    assert torch.allclose(k_out[..., rot_dim:], k[..., rot_dim:])
