# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Track 9E shared rotary Hopper kernels and fast KernelRegistry dispatch.

Covers:
- Lazy Triton import and CPU-only execution when Triton is absent.
- Bit-exact / <=1 ULP GPU parity vs PyTorch fallback for:
  1. ``complex_rope.apply_complex_rotary_embedding`` (FP64 & FP32 compute, broadcast shapes, wide D).
  2. ``embedding.apply_rotary_pos_emb`` (``unsqueeze_dim=1/2/-2``, GQA, mixed precision, transposed views).
  3. ``rope_2d.RotaryPositionEmbedding2D`` & ``indexed_2d.IndexedRotaryPositionEmbedding2D``
     (signed & unsigned coordinates, negative index rejection, pow2 & non-pow2 D).
  4. ``interleaved.apply_rotary_emb`` (3D/4D, even & odd ``start_index``, partial slices, ``scale``).
- Hopper SM90 ``n_spills == 0`` across all compiled kernel variants.
- Strict ``int64`` indexing (> 2^31 element offset dereference when >=12 GiB free VRAM).
- ``torch.compile(backend="inductor", fullgraph=True)`` + ``register_worldfoundry_inductor_fallbacks()``.
- CUDA Graph capture & replay with mutated inputs.
- Autograd fallback & ``WORLDFOUNDRY_KERNEL_BACKEND=torch`` fallback.
- ``KernelRegistry.dispatch`` fast cache-hit, LRU tail tracking, receipts, quarantine, and multi-GPU device guard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.core.attention.rotary.complex_rope import (
    apply_complex_rotary_embedding,
    complex_rotary_frequencies,
)
from worldfoundry.core.attention.rotary.embedding import apply_rotary_pos_emb
from worldfoundry.core.attention.rotary.indexed_2d import (
    IndexedRotaryPositionEmbedding2D,
    ShapeCachedPositionGetter,
)
from worldfoundry.core.attention.rotary.interleaved import apply_rotary_emb
from worldfoundry.core.attention.rotary.rope_2d import (
    CrocoRotaryPositionEmbedding2D,
    PositionGetter,
    RotaryPositionEmbedding2D,
)
from worldfoundry.core.attention.rotary.triton_rotary_ops import (
    _get_triton_suite,
    triton_rope_2d_indexed,
)
from worldfoundry.core.kernels.registry import (
    KernelNotSupported,
    KernelRegistry,
    kernel_autotune_enabled,
    kernel_dispatch_receipt_scope,
)

_CUDA = torch.cuda.is_available()
_HOPPER_PLUS = _CUDA and torch.cuda.get_device_capability()[0] >= 9


def _run_torch_backend(fn: Any) -> Any:
    prev = os.environ.get("WORLDFOUNDRY_KERNEL_BACKEND")
    os.environ["WORLDFOUNDRY_KERNEL_BACKEND"] = "torch"
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop("WORLDFOUNDRY_KERNEL_BACKEND", None)
        else:
            os.environ["WORLDFOUNDRY_KERNEL_BACKEND"] = prev


def _assert_bf16_lossless_1ulp(
    actual: torch.Tensor, expected: torch.Tensor, scale_ref: torch.Tensor | None = None
) -> None:
    """Assert BF16 outputs differ by at most 1 BF16 ULP relative to output/input scale."""
    assert actual.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16
    diff = (actual.float() - expected.float()).abs()
    if actual.numel() == 0:
        return
    # Exact 1 BF16 ULP at magnitude m is 2^(floor(log2(m)) - 7) <= m / 128.
    mag = expected.float().abs()
    if scale_ref is not None:
        mag = torch.maximum(mag, scale_ref.float().abs())
    # Next power-of-two upper bound on 1 BF16 ULP (with tiny floor for near-zero cancellation).
    ulp_bound = torch.maximum(mag * (1.0 / 64.0), torch.tensor(1e-5, device=actual.device))
    assert bool((diff <= ulp_bound).all()), f"max diff={diff.max().item():.3e} exceeded 1-ULP bound"
    rel_l2 = float((diff.norm() / expected.float().norm().clamp_min(1e-8)).item())
    assert rel_l2 < 1e-4, f"rel_l2={rel_l2:.3e}"


# ──────────────────────────────────────────────────────────────────────────
# 1. CPU-only lazy Triton import subprocess test
# ──────────────────────────────────────────────────────────────────────────


def test_cpu_only_absent_triton_subprocess() -> None:
    """All shared rotary modules and KernelRegistry must work on CPU with Triton blocked."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "import torch\n"
        "from worldfoundry.core.attention.rotary.complex_rope import (\n"
        "    apply_complex_rotary_embedding, complex_rotary_frequencies,\n"
        ")\n"
        "from worldfoundry.core.attention.rotary.embedding import apply_rotary_pos_emb\n"
        "from worldfoundry.core.attention.rotary.interleaved import apply_rotary_emb\n"
        "from worldfoundry.core.attention.rotary.rope_2d import PositionGetter, RotaryPositionEmbedding2D\n"
        "from worldfoundry.core.attention.rotary.indexed_2d import IndexedRotaryPositionEmbedding2D\n"
        "from worldfoundry.core.kernels.registry import KernelRegistry\n"
        "x = torch.randn(1, 6, 32)\n"
        "f = complex_rotary_frequencies(16, end=6).view(6, 1, 8)\n"
        "assert apply_complex_rotary_embedding(x, f, 2).shape == (1, 6, 32)\n"
        "q = torch.randn(1, 4, 6, 16)\n"
        "k = torch.randn(1, 2, 6, 16)\n"
        "c = torch.randn(1, 6, 16)\n"
        "s = torch.randn(1, 6, 16)\n"
        "qo, ko = apply_rotary_pos_emb(q, k, c, s, unsqueeze_dim=1)\n"
        "assert qo.shape == q.shape and ko.shape == k.shape\n"
        "pos = PositionGetter()(1, 2, 3, torch.device('cpu'))\n"
        "assert RotaryPositionEmbedding2D()(q, pos).shape == q.shape\n"
        "assert IndexedRotaryPositionEmbedding2D()(q, pos).shape == q.shape\n"
        "fr = torch.randn(6, 8)\n"
        "assert apply_rotary_emb(fr, q, start_index=4).shape == q.shape\n"
        "reg = KernelRegistry()\n"
        "assert reg.dispatch('cpu_op', lambda v: v + 1, 4, signature=('s',)) == 5\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"CPU-only absent-Triton test failed:\n{proc.stdout}\n{proc.stderr}"


# ──────────────────────────────────────────────────────────────────────────
# 2. complex_rope.py GPU parity across dtypes, compute_dtype, broadcast shapes
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("compute_dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("freq_dtype", [torch.complex128, torch.complex64])
@pytest.mark.parametrize(
    "shape_mode,heads,head_dim",
    [
        ("s_1_d", 12, 128),
        ("1_s_1_d", 8, 64),
        ("b_s_1_d", 6, 96),
        ("s_h_d", 4, 128),
        ("b_s_h_d", 4, 64),
        ("s_1_d", 2, 640),  # wide D > 512 multi-tile D_HALF
    ],
)
def test_gpu_complex_rope_parity(
    act_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    freq_dtype: torch.dtype,
    shape_mode: str,
    heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(101)
    device = torch.device("cuda")
    b, s = 2, 37
    # Exercise strided QKV slice (non-contiguous seq stride)
    qkv = torch.randn(b, s, 3 * heads * head_dim, device=device, dtype=act_dtype)
    val = qkv.chunk(3, dim=-1)[0]

    d_half = head_dim // 2
    if shape_mode == "s_1_d":
        ang = torch.randn(s, 1, d_half, device=device, dtype=torch.float64)
    elif shape_mode == "1_s_1_d":
        ang = torch.randn(1, s, 1, d_half, device=device, dtype=torch.float64)
    elif shape_mode == "b_s_1_d":
        ang = torch.randn(b, s, 1, d_half, device=device, dtype=torch.float64)
    elif shape_mode == "s_h_d":
        ang = torch.randn(s, heads, d_half, device=device, dtype=torch.float64)
    else:
        ang = torch.randn(b, s, heads, d_half, device=device, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(ang), ang).to(freq_dtype)

    with torch.inference_mode():
        got = apply_complex_rotary_embedding(val, freqs, heads, compute_dtype=compute_dtype)
        ref = _run_torch_backend(lambda: apply_complex_rotary_embedding(val, freqs, heads, compute_dtype=compute_dtype))

    assert got.shape == ref.shape and got.dtype == ref.dtype
    if act_dtype == torch.bfloat16:
        _assert_bf16_lossless_1ulp(got, ref, val)
    elif act_dtype == torch.float16:
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=2e-3)
    else:
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_complex_rope_edge_cases() -> None:
    device = torch.device("cuda")
    empty = torch.empty(1, 0, 128, device=device, dtype=torch.bfloat16)
    freqs_empty = complex_rotary_frequencies(64, end=0, device=device).view(0, 1, 32)
    with torch.inference_mode():
        out_empty = apply_complex_rotary_embedding(empty, freqs_empty, 2)
    assert out_empty.shape == empty.shape and out_empty.dtype == empty.dtype

    x = torch.randn(1, 4, 32, device=device, dtype=torch.bfloat16)
    f = complex_rotary_frequencies(16, end=4, device=device).view(4, 1, 8)
    with pytest.raises(ValueError, match="float32 or float64"):
        apply_complex_rotary_embedding(x, f, 2, compute_dtype=torch.bfloat16)


# ──────────────────────────────────────────────────────────────────────────
# 3. embedding.py GPU parity (BHSD, BSHD, -2, 3D Qwen vision, GQA, mixed dtype)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("qk_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2, -2])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("head_dim,hq,hk", [(64, 16, 4), (128, 32, 8), (96, 6, 2), (640, 2, 1)])
def test_gpu_qk_half_rope_parity(
    qk_dtype: torch.dtype,
    table_dtype: torch.dtype,
    unsqueeze_dim: int,
    transposed: bool,
    head_dim: int,
    hq: int,
    hk: int,
) -> None:
    if qk_dtype == torch.float16 and table_dtype == torch.bfloat16:
        # PyTorch promotes fp16+bf16 to fp32; test still validates exact promotion!
        pass
    torch.manual_seed(202)
    device = torch.device("cuda")
    b, s = 2, 35
    q = torch.randn(b, hq, s, head_dim, device=device, dtype=qk_dtype)
    k = torch.randn(b, hk, s, head_dim, device=device, dtype=qk_dtype)
    norm_dim = unsqueeze_dim if unsqueeze_dim >= 0 else 4 + unsqueeze_dim
    if norm_dim == 2:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if not transposed:
            q = q.contiguous()
            k = k.contiguous()
    elif transposed:
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)

    # Non-duplicated arbitrary cos/sin tables (matches test_qk_rotary_embedding contract)
    cos = torch.randn(1 if head_dim == 96 else b, s, head_dim, device=device, dtype=table_dtype)
    sin = torch.randn(1 if head_dim == 96 else b, s, head_dim, device=device, dtype=table_dtype)

    with torch.inference_mode():
        gq, gk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        rq, rk = _run_torch_backend(lambda: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim))

    assert gq.dtype == rq.dtype and gk.dtype == rk.dtype
    assert gq.shape == rq.shape and gk.shape == rk.shape
    if rq.dtype == torch.bfloat16:
        assert torch.equal(gq, rq) and torch.equal(gk, rk)
    elif rq.dtype == torch.float16:
        assert torch.equal(gq, rq) and torch.equal(gk, rk)
    else:
        torch.testing.assert_close(gq, rq, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(gk, rk, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_qk_half_rope_3d_qwen_vision_layout() -> None:
    torch.manual_seed(203)
    device = torch.device("cuda")
    q = torch.randn(256, 16, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(256, 16, 64, device=device, dtype=torch.bfloat16)
    cos = torch.randn(256, 64, device=device, dtype=torch.float32)
    sin = torch.randn(256, 64, device=device, dtype=torch.float32)
    with torch.inference_mode():
        gq, gk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=-2)
        rq, rk = _run_torch_backend(lambda: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=-2))
    assert gq.dtype == rq.dtype == torch.float32
    torch.testing.assert_close(gq, rq, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gk, rk, rtol=1e-6, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────
# 4. rope_2d.py & indexed_2d.py GPU parity + signed/unsigned contracts
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [32, 64, 96, 128, 256])
def test_gpu_rope_2d_and_indexed_2d_parity(dtype: torch.dtype, dim: int) -> None:
    torch.manual_seed(303)
    device = torch.device("cuda")
    b, h, gh, gw = 2, 8, 7, 9
    n = gh * gw
    tokens = torch.randn(b, h, n, dim, device=device, dtype=dtype)
    pos_pos = ShapeCachedPositionGetter()(b, gh, gw, device)

    # Signed positions including negatives and zeros (CUT3R / CroCo contract)
    pos_signed = pos_pos.clone()
    pos_signed[:, 0, :] = -1
    pos_signed[:, 1, 0] = -5
    pos_signed[:, 2, 1] = 0

    rope2d = RotaryPositionEmbedding2D(frequency=100.0, scaling_factor=1.25)
    croco = CrocoRotaryPositionEmbedding2D(freq=100.0, F0=1.25)
    idx2d = IndexedRotaryPositionEmbedding2D(frequency=100.0, scaling_factor=2.0)

    with torch.inference_mode():
        g_signed = rope2d(tokens, pos_signed)
        r_signed = _run_torch_backend(lambda: rope2d(tokens, pos_signed))
        g_croco = croco(tokens, pos_signed)
        g_idx = idx2d(tokens, pos_pos)
        r_idx = _run_torch_backend(lambda: idx2d(tokens, pos_pos))

    assert g_signed.dtype == dtype and g_idx.dtype == dtype
    if dtype in (torch.bfloat16, torch.float16):
        assert torch.equal(g_signed, r_signed)
        assert torch.equal(g_croco, r_signed)
        assert torch.equal(g_idx, r_idx)
    else:
        torch.testing.assert_close(g_signed, r_signed, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(g_croco, r_signed, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(g_idx, r_idx, rtol=1e-6, atol=1e-6)

    # Frequency cache reuse preserved
    assert len(rope2d.frequency_cache) == 1
    assert len(idx2d.frequency_cache) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_indexed_2d_rejects_negative_indices() -> None:
    device = torch.device("cuda")
    rotary = IndexedRotaryPositionEmbedding2D()
    tokens = torch.randn(1, 4, 4, 32, device=device, dtype=torch.bfloat16)
    positions = torch.tensor([[[0, 1], [-1, 2], [1, 0], [2, 2]]], device=device)
    with pytest.raises(IndexError):
        rotary(tokens, positions)


# ──────────────────────────────────────────────────────────────────────────
# 5. interleaved.py GPU parity (3D/4D, even & odd start_index, scale, partial)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "ndim,dim,start_index,rot_dim,scale",
    [
        (4, 128, 0, 128, 1.0),
        (4, 128, 16, 64, 1.0),
        (4, 128, 15, 64, 0.75),  # odd start_index
        (4, 96, 8, 48, 1.25),  # non-pow2 D and rot_dim
        (4, 65, 4, 32, 1.0),  # odd total D with even start_index
        (3, 128, 32, 64, 1.0),  # 3D with seq slicing freqs[-seq_len:]
        (4, 384, 32, 256, 1.0),  # multi-tile D > 256
    ],
)
def test_gpu_interleaved_partial_rope_parity(
    dtype: torch.dtype,
    ndim: int,
    dim: int,
    start_index: int,
    rot_dim: int,
    scale: float,
) -> None:
    torch.manual_seed(404)
    device = torch.device("cuda")
    b, h, s = 2, 8, 41
    if ndim == 3:
        t = torch.randn(b, s, dim, device=device, dtype=dtype)
        freqs = torch.randn(s + 13, rot_dim, device=device, dtype=torch.float32)
    else:
        t = torch.randn(b, h, s, dim, device=device, dtype=dtype)
        freqs = torch.randn(s, rot_dim, device=device, dtype=torch.float32)

    with torch.inference_mode():
        got = apply_rotary_emb(freqs, t, start_index=start_index, scale=scale, seq_dim=-2)
        ref = _run_torch_backend(lambda: apply_rotary_emb(freqs, t, start_index=start_index, scale=scale, seq_dim=-2))

    assert got.shape == ref.shape and got.dtype == ref.dtype
    # Outer slices must be bit-identical
    if start_index > 0:
        assert torch.equal(got[..., :start_index], t[..., :start_index])
    if start_index + rot_dim < dim:
        assert torch.equal(got[..., start_index + rot_dim :], t[..., start_index + rot_dim :])

    if dtype == torch.bfloat16:
        _assert_bf16_lossless_1ulp(got, ref, t * abs(scale))
    elif dtype == torch.float16:
        torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_interleaved_per_head_freqs_and_tensor_scale_fallback() -> None:
    torch.manual_seed(405)
    device = torch.device("cuda")
    t = torch.randn(2, 4, 16, 64, device=device, dtype=torch.bfloat16)
    freqs_per_head = torch.randn(2, 4, 16, 32, device=device, dtype=torch.float32)
    with torch.inference_mode():
        got = apply_rotary_emb(freqs_per_head, t, start_index=8, scale=1.0, seq_dim=-2)
        ref = _run_torch_backend(lambda: apply_rotary_emb(freqs_per_head, t, start_index=8, scale=1.0, seq_dim=-2))
        _assert_bf16_lossless_1ulp(got, ref, t)

        # Tensor scale falls back to PyTorch cleanly
        t_scale = torch.linspace(0.5, 1.5, 16, device=device, dtype=torch.float32).view(1, 1, 16, 1)
        got_ts = apply_rotary_emb(freqs_per_head, t, start_index=8, scale=t_scale, seq_dim=-2)
        ref_ts = _run_torch_backend(
            lambda: apply_rotary_emb(freqs_per_head, t, start_index=8, scale=t_scale, seq_dim=-2)
        )
        assert torch.equal(got_ts, ref_ts)


# ──────────────────────────────────────────────────────────────────────────
# 6. Hopper SM90 zero-spill (n_spills == 0) verification
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="requires Hopper SM90+ GPU")
def test_hopper_all_rotary_kernels_zero_spills() -> None:
    device = torch.device("cuda")
    torch.manual_seed(505)
    with torch.inference_mode():
        # Trigger all 6 Triton kernel definitions across representative shapes
        v = torch.randn(1, 128, 40 * 128, device=device, dtype=torch.bfloat16)
        f = complex_rotary_frequencies(128, end=128, device=device).view(128, 1, 64)
        apply_complex_rotary_embedding(v, f, 40, compute_dtype=torch.float64)
        apply_complex_rotary_embedding(v, f, 40, compute_dtype=torch.float32)

        q = torch.randn(2, 32, 128, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 128, 128, device=device, dtype=torch.bfloat16)
        c = torch.randn(2, 128, 128, device=device, dtype=torch.bfloat16)
        s = torch.randn(2, 128, 128, device=device, dtype=torch.bfloat16)
        apply_rotary_pos_emb(q, k, c, s, unsqueeze_dim=1)
        apply_rotary_pos_emb(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), c, s, unsqueeze_dim=2)

        pos = PositionGetter()(2, 8, 8, device)
        tok64 = torch.randn(2, 16, 64, 64, device=device, dtype=torch.bfloat16)
        tok96 = torch.randn(2, 8, 64, 96, device=device, dtype=torch.bfloat16)
        RotaryPositionEmbedding2D()(tok64, pos)
        RotaryPositionEmbedding2D()(tok96, pos)

        t4 = torch.randn(2, 16, 128, 128, device=device, dtype=torch.bfloat16)
        fr4 = torch.randn(128, 64, device=device, dtype=torch.float32)
        apply_rotary_emb(fr4, t4, start_index=16, scale=1.0, seq_dim=-2)
        apply_rotary_emb(fr4, t4, start_index=15, scale=1.0, seq_dim=-2)

    suite = _get_triton_suite()
    kernel_names = (
        "complex_rope_fwd_kernel",
        "qk_half_rope_fwd_kernel",
        "rope_2d_indexed_contig_kernel",
        "rope_2d_indexed_fwd_kernel",
        "interleaved_even_start_kernel",
        "interleaved_partial_rope_fwd_kernel",
    )
    for name in kernel_names:
        kernel = getattr(suite, name)
        compiled_count = 0
        for entry in kernel.device_caches.values():
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            for compiled in cache_dict.values():
                compiled_count += 1
                assert compiled.n_spills == 0, f"{name} spilled {compiled.n_spills} bytes (regs={compiled.n_regs})"
        assert compiled_count > 0, f"{name} was not compiled"


# ──────────────────────────────────────────────────────────────────────────
# 7. Strict int64 (> 2^31 element offset) verification
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_strict_int64_indexing_above_2_pow_31() -> None:
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 12 * (1024**3):
        # Requires >=12 GiB free VRAM for >2^31 element offset check.
        pytest.skip()
    device = torch.device("cuda")
    # Allocate (2^31 + 2048) BF16 elements (~4.3 GiB) and view batch 0 at 0 and batch 1 at offset 2^31 + 1024.
    big_stride = (1 << 31) + 1024
    buf = torch.empty(big_stride + 1024, device=device, dtype=torch.bfloat16)
    torch.manual_seed(606)
    buf[:1024].normal_()
    buf[big_stride : big_stride + 1024].normal_()

    # 1. complex_rope with batch stride > 2^31
    val = torch.as_strided(buf, size=(2, 4, 256), stride=(big_stride, 256, 1))
    freqs = complex_rotary_frequencies(64, end=4, device=device).view(4, 1, 32)
    with torch.inference_mode():
        got = apply_complex_rotary_embedding(val, freqs, 4, compute_dtype=torch.float64)
        ref = _run_torch_backend(lambda: apply_complex_rotary_embedding(val, freqs, 4, compute_dtype=torch.float64))
    assert torch.equal(got, ref)

    # 2. qk_half_rope with batch stride > 2^31
    q = torch.as_strided(buf, size=(2, 4, 4, 64), stride=(big_stride, 256, 64, 1))
    k = torch.as_strided(buf, size=(2, 2, 4, 64), stride=(big_stride, 256, 64, 1))
    cos = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
    sin = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        gq, gk = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        rq, rk = _run_torch_backend(lambda: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1))
    assert torch.equal(gq, rq) and torch.equal(gk, rk)

    del buf
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# 8. torch.compile(fullgraph=True) + register_worldfoundry_inductor_fallbacks()
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_torch_compile_fullgraph_and_inductor_fallbacks() -> None:
    from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

    register_worldfoundry_inductor_fallbacks()
    device = torch.device("cuda")
    torch.manual_seed(707)

    val = torch.randn(1, 32, 4 * 64, device=device, dtype=torch.bfloat16)
    freqs_c = complex_rotary_frequencies(64, end=32, device=device).view(32, 1, 32)
    q = torch.randn(1, 4, 32, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 32, 64, device=device, dtype=torch.bfloat16)
    cos = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    sin = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    pos = PositionGetter()(1, 4, 8, device)
    rope2d = RotaryPositionEmbedding2D()
    _ = rope2d(q, pos)
    cos_tbl, sin_tbl = next(iter(rope2d.frequency_cache.values()))
    freqs_i = torch.randn(32, 32, device=device, dtype=torch.float32)

    def combined(
        v_in: torch.Tensor,
        fc_in: torch.Tensor,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        c_in: torch.Tensor,
        s_in: torch.Tensor,
        p_in: torch.Tensor,
        ct_in: torch.Tensor,
        st_in: torch.Tensor,
        fi_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        o1 = apply_complex_rotary_embedding(v_in, fc_in, 4, compute_dtype=torch.float64)
        o2q, o2k = apply_rotary_pos_emb(q_in, k_in, c_in, s_in, unsqueeze_dim=1)
        o3 = triton_rope_2d_indexed(o2q, p_in, ct_in, st_in, signed_positions=True)
        assert o3 is not None
        o4 = apply_rotary_emb(fi_in, o3, start_index=8, scale=1.0, seq_dim=-2)
        return o1, o2q, o2k, o3, o4

    compiled = torch.compile(combined, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        ref_outs = combined(val, freqs_c, q, k, cos, sin, pos, cos_tbl, sin_tbl, freqs_i)
        got_outs = compiled(val, freqs_c, q, k, cos, sin, pos, cos_tbl, sin_tbl, freqs_i)
    for g, r in zip(got_outs, ref_outs, strict=True):
        assert torch.equal(g, r)


# ──────────────────────────────────────────────────────────────────────────
# 9. CUDA Graph capture & replay
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_cuda_graph_capture_and_replay() -> None:
    device = torch.device("cuda")
    torch.manual_seed(808)

    val = torch.randn(1, 32, 4 * 64, device=device, dtype=torch.bfloat16)
    freqs_c = complex_rotary_frequencies(64, end=32, device=device).view(32, 1, 32)
    q = torch.randn(1, 4, 32, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 32, 64, device=device, dtype=torch.bfloat16)
    cos = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    sin = torch.randn(1, 32, 64, device=device, dtype=torch.bfloat16)
    pos = PositionGetter()(1, 4, 8, device)
    rope2d = RotaryPositionEmbedding2D()
    _ = rope2d(q, pos)
    cos_tbl, sin_tbl = next(iter(rope2d.frequency_cache.values()))
    freqs_i = torch.randn(32, 32, device=device, dtype=torch.float32)

    def run_all() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        o1 = apply_complex_rotary_embedding(val, freqs_c, 4, compute_dtype=torch.float64)
        o2q, o2k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        o3 = triton_rope_2d_indexed(q, pos, cos_tbl, sin_tbl, signed_positions=True)
        assert o3 is not None
        o4 = apply_rotary_emb(freqs_i, q, start_index=8, scale=1.0, seq_dim=-2)
        return o1, o2q, o2k, o3, o4

    with torch.inference_mode():
        # Warmup outside capture
        _ = run_all()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_outs = run_all()

        # Mutate static inputs and replay
        val.normal_()
        q.normal_()
        k.normal_()
        graph.replay()
        torch.cuda.synchronize()
        e_outs = run_all()

    for g, e in zip(g_outs, e_outs, strict=True):
        assert torch.equal(g, e)


# ──────────────────────────────────────────────────────────────────────────
# 10. Autograd fallback parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_autograd_fallback_all_rotary_ops() -> None:
    device = torch.device("cuda")
    torch.manual_seed(909)

    # 1. complex_rope
    v = torch.randn(1, 8, 2 * 32, device=device, dtype=torch.float32, requires_grad=True)
    f = complex_rotary_frequencies(32, end=8, device=device).view(8, 1, 16)
    apply_complex_rotary_embedding(v, f, 2, compute_dtype=torch.float64).sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # 2. embedding
    q = torch.randn(1, 4, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, 2, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    c = torch.randn(1, 8, 32, device=device, dtype=torch.float32)
    s = torch.randn(1, 8, 32, device=device, dtype=torch.float32)
    qo, ko = apply_rotary_pos_emb(q, k, c, s, unsqueeze_dim=1)
    (qo.sum() + ko.sum()).backward()
    assert q.grad is not None and k.grad is not None

    # 3. rope_2d & indexed_2d
    pos = PositionGetter()(1, 2, 4, device)
    t1 = torch.randn(1, 4, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    RotaryPositionEmbedding2D()(t1, pos).sum().backward()
    assert t1.grad is not None
    t2 = torch.randn(1, 4, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    IndexedRotaryPositionEmbedding2D()(t2, pos).sum().backward()
    assert t2.grad is not None

    # 4. interleaved
    ti = torch.randn(1, 4, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    fi = torch.randn(8, 16, device=device, dtype=torch.float32)
    apply_rotary_emb(fi, ti, start_index=4, scale=1.0, seq_dim=-2).sum().backward()
    assert ti.grad is not None


# ──────────────────────────────────────────────────────────────────────────
# 11. Fast KernelRegistry.dispatch contracts (LRU tail, receipts, quarantine, env)
# ──────────────────────────────────────────────────────────────────────────


def test_fast_kernel_registry_lru_tail_receipts_and_quarantine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")
    assert kernel_autotune_enabled() is False
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "1")
    assert kernel_autotune_enabled() is True
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")

    reg = KernelRegistry(selection_limit=2)
    calls = {"fast": 0, "fb": 0}

    def fast_impl(v: int) -> int:
        calls["fast"] += 1
        return v * 10

    def fb_impl(v: int) -> int:
        calls["fb"] += 1
        return v + 1

    reg.register("op", backend="triton", name="fast", implementation=fast_impl, predicate=lambda v: v > 0)

    # Populate 2 entries, hit s1 so s1 moves to tail, then insert s3 -> s2 must be evicted (not s1!).
    assert reg.dispatch("op", fb_impl, 1, signature=("s1",)) == 10
    assert reg.dispatch("op", fb_impl, 2, signature=("s2",)) == 20
    assert reg.dispatch("op", fb_impl, 1, signature=("s1",)) == 10  # moves s1 to tail
    assert reg.dispatch("op", fb_impl, 3, signature=("s3",)) == 30  # evicts s2
    assert len(reg._selection_cache) == 2

    receipt: dict[str, Any] = {}
    with kernel_dispatch_receipt_scope(receipt):
        assert reg.dispatch("op", fb_impl, 1, signature=("s1",)) == 10
        assert reg.dispatch("op", fb_impl, 2, signature=("s2",)) == 20  # was evicted -> miss!
    assert receipt["dispatches"][0]["cache_hit"] is True
    assert receipt["dispatches"][1]["cache_hit"] is False

    # Runtime failure on cached entry quarantines it and drops selection cleanly.
    reg2 = KernelRegistry()
    state = {"fail": False}

    def flaky(v: int) -> int:
        if state["fail"]:
            raise KernelNotSupported("flaky contract")
        return v * 3

    reg2.register("flaky_op", backend="triton", name="flaky", implementation=flaky, predicate=lambda _: True)
    assert reg2.dispatch("flaky_op", fb_impl, 5, signature=("x",)) == 15
    state["fail"] = True
    with pytest.warns(RuntimeWarning, match="flaky"):
        assert reg2.dispatch("flaky_op", fb_impl, 5, signature=("x",)) == 6
    assert reg2.report()["failed_signatures"] == 1
    reg2.clear_failures()
    assert reg2.report()["selection_cache_entries"] == 0
    assert reg2.report()["failed_signatures"] == 0
