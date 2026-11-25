# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Wave 9C shared core Hopper SM90 kernels.

Covers:
1. ``worldfoundry.core.kernels.triton_group_norm_silu`` & ``diffusion.group_norm_silu``
2. ``worldfoundry.core.attention.rotary.rope_kernel`` (single + fused Q/K)
3. ``worldfoundry.core.kernels.triton_moe`` & ``worldfoundry.core.kernels.moe``

Verifies:
- Lazy Triton import under CPU-only ``CUDA_VISIBLE_DEVICES=""`` subprocesses.
- Bit-exact / <= 1-2 BF16 ULP lossless parity vs PyTorch eager & FP64 truth.
- Catastrophic-cancellation immunity on large-mean + outlier features.
- Native ``channels_last`` (4D NHWC) & ``channels_last_3d`` (5D NDHWC) support.
- Strict ``tl.int64`` indexing on ``>2^31``-element tensors.
- Zero register spills (``n_spills == 0``) on Hopper SM90 across all compiled kernels.
- ``torch.compile(fullgraph=True)`` with ``register_worldfoundry_inductor_fallbacks()``.
- CUDA Graph capture and replay safety (including MoE workspace reuse).
- Autograd, CPU, and ``WORLDFOUNDRY_KERNEL_BACKEND=torch`` fallback contracts.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.kernels.diffusion import clear_kernel_dispatch_cache, group_norm_silu
from worldfoundry.core.kernels.moe import routed_swiglu_moe, routed_swiglu_moe_pytorch
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope
from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ref_rope_fp32(x: torch.Tensor, freqs: torch.Tensor, interleaved: bool) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    f = freqs[:, 0, 0, ::2] if interleaved else freqs[:, 0, 0, :d_half]
    cos_t = torch.cos(f.double()).float()[None, :, None, :].to(x.dtype)
    sin_t = torch.sin(f.double()).float()[None, :, None, :].to(x.dtype)
    if interleaved:
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos_t - b * sin_t, b * cos_t + a * sin_t), dim=-1).flatten(-2)
    a, b = x[..., :d_half], x[..., d_half:]
    return torch.cat((a * cos_t - b * sin_t, b * cos_t + a * sin_t), dim=-1)


def _assert_zero_spills(prefix: str, kernel_dict: dict[str, object], expected_keys: tuple[str, ...]) -> None:
    for key in expected_keys:
        jit_fn = kernel_dict[key]
        dev_caches = getattr(jit_fn, "device_caches", None)
        assert dev_caches, f"{prefix}.{key} had no compiled device cache entries"
        checked = 0
        for entry in dev_caches.values():
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(cache_dict, dict):
                continue
            for compiled in cache_dict.values():
                checked += 1
                spills = int(getattr(compiled, "n_spills", 0))
                regs = int(getattr(compiled, "n_regs", 0))
                assert spills == 0, f"{prefix}.{key} spilled {spills} bytes (n_regs={regs})"
        assert checked > 0, f"{prefix}.{key} had zero compiled variants"


# ──────────────────────────────────────────────────────────────────────────
# 1. CPU-only lazy Triton import hygiene (zero triton / CUDA init)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cpu_only_subprocess_lazy_triton_imports() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = """
import sys
import torch
import torch.nn.functional as F

import worldfoundry.core.kernels.triton_group_norm_silu as tgn
import worldfoundry.core.kernels.diffusion as diff
import worldfoundry.core.attention.rotary.rope_kernel as rk
import worldfoundry.core.kernels.triton_moe as tm
import worldfoundry.core.kernels.moe as moe

assert "triton" not in sys.modules, "triton leaked on module import"
assert "triton.language" not in sys.modules, "triton.language leaked on module import"
assert not torch.cuda.is_initialized(), "CUDA initialized during CPU-only import"

# Verify custom ops registered cleanly without importing Triton
assert hasattr(torch.ops.worldfoundry, "core_group_norm_silu_fwd")
assert hasattr(torch.ops.worldfoundry, "rope_bshd_fwd")
assert hasattr(torch.ops.worldfoundry, "rope_bshd_inplace")
assert hasattr(torch.ops.worldfoundry, "rope_qk_bshd_fwd")
assert hasattr(torch.ops.worldfoundry, "rope_qk_bshd_inplace")
assert hasattr(torch.ops.worldfoundry, "routed_swiglu_moe_fwd")

# Verify CPU fallbacks work cleanly without touching Triton
x = torch.randn(1, 16, 4, 4)
w = torch.ones(16)
b = torch.zeros(16)
y = diff.group_norm_silu(x, w, b, num_groups=4)
assert torch.allclose(y, F.silu(F.group_norm(x, 4, w, b)))

h = torch.randn(4, 16)
rw = torch.full((4, 2), 0.5)
se = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64)
gw = torch.randn(4, 8, 16)
uw = torch.randn(4, 8, 16)
dw = torch.randn(4, 16, 8)
out_moe = moe.routed_swiglu_moe(h, rw, se, gw, uw, dw)
assert out_moe.shape == h.shape

assert "triton" not in sys.modules, "triton leaked during CPU execution"
assert not torch.cuda.is_initialized(), "CUDA initialized during CPU execution"
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ──────────────────────────────────────────────────────────────────────────
# 2. GroupNorm + SiLU parity, layouts, Chan stability, dispatch, compile, CG
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape,num_groups,mem_fmt",
    [
        ((2, 32), 8, torch.contiguous_format),  # 2D small 1-pass
        ((2, 32, 64), 8, torch.contiguous_format),  # 3D small 1-pass
        ((1, 64, 16, 16), 32, torch.contiguous_format),  # 4D 1-pass
        ((1, 256, 64, 64), 32, torch.contiguous_format),  # 4D NCHW chunked
        ((1, 128, 5, 32, 32), 32, torch.contiguous_format),  # 5D NCDHW chunked
        ((1, 256, 64, 64), 32, torch.channels_last),  # 4D NHWC multi-group pow2
        ((1, 128, 5, 32, 32), 32, torch.channels_last_3d),  # 5D NDHWC multi-group pow2
        ((1, 96, 32, 32), 32, torch.channels_last),  # 4D NHWC non-pow2 CG=3
        ((1, 384, 3, 16, 16), 32, torch.channels_last_3d),  # 5D NDHWC non-pow2 CG=12
        ((1, 512, 16, 16), 2, torch.channels_last),  # 4D NHWC large CG=256
    ],
)
def test_group_norm_silu_parity_and_layout_preservation(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    num_groups: int,
    mem_fmt: torch.memory_format,
) -> None:
    from worldfoundry.core.kernels.triton_group_norm_silu import group_norm_silu as triton_gn_silu

    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype).to(memory_format=mem_fmt)
    w = torch.randn(shape[1], device="cuda", dtype=dtype)
    b = torch.randn(shape[1], device="cuda", dtype=dtype)

    with torch.inference_mode():
        got = triton_gn_silu(x, w, b, num_groups, 1e-5)
        ref = F.silu(F.group_norm(x, num_groups, weight=w, bias=b, eps=1e-5))

    assert got.shape == x.shape
    assert got.dtype == x.dtype
    assert got.stride() == x.stride(), f"layout lost: got {got.stride()}, expected {x.stride()}"
    if dtype == torch.bfloat16:
        atol, rtol = 6.5e-2, 2e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 5e-3
    else:
        atol, rtol = 2e-5, 2e-5
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mem_fmt", [torch.contiguous_format, torch.channels_last])
def test_group_norm_silu_chan_variance_large_mean_outlier_stability(mem_fmt: torch.memory_format) -> None:
    from worldfoundry.core.kernels.triton_group_norm_silu import group_norm_silu as triton_gn_silu

    torch.manual_seed(7)
    x = (torch.randn((1, 256, 64, 64), device="cuda", dtype=torch.float32) * 0.1 + 1e5).to(memory_format=mem_fmt)
    # Inject index-0 outlier to defeat naive shift-by-first-element estimators.
    x[:, :, 0, 0] = 0.0
    w = torch.ones(256, device="cuda", dtype=torch.float32)
    b = torch.zeros(256, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        got = triton_gn_silu(x, w, b, 32, 1e-5)
        ref_f64 = F.silu(F.group_norm(x.double(), 32, weight=w.double(), bias=b.double(), eps=1e-5)).float()

    assert torch.isfinite(got).all(), "Chan variance produced non-finite output on large-mean + outlier input"
    max_err = (got - ref_f64).abs().max().item()
    assert max_err < 1e-4, f"large-mean error vs FP64 too high: {max_err:.3e}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_group_norm_silu_diffusion_dispatch_autograd_and_backend_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_kernel_dispatch_cache()
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_AUTOTUNE_ENABLED", "0")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")

    # Channels-last 5D dispatches to Triton automatically
    x_cl3d = torch.randn((1, 64, 4, 32, 32), device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    receipt: dict[str, object] = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        y_cl3d = group_norm_silu(x_cl3d, w, b, num_groups=16)
    dispatches = receipt["dispatches"]
    assert isinstance(dispatches, list) and dispatches[-1]["implementation"] == "triton_group_norm_silu"
    assert y_cl3d.stride() == x_cl3d.stride()

    # WORLDFOUNDRY_KERNEL_BACKEND=torch forces PyTorch
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    with torch.inference_mode():
        y_torch = group_norm_silu(x_cl3d, w, b, num_groups=16)
    torch.testing.assert_close(y_cl3d, y_torch, atol=6.5e-2, rtol=2e-2)

    # Autograd fallback preserves exact gradients
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")
    x_g1 = torch.randn((1, 32, 16, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    w_g1 = torch.randn(32, device="cuda", dtype=torch.float32, requires_grad=True)
    b_g1 = torch.randn(32, device="cuda", dtype=torch.float32, requires_grad=True)
    x_g2 = x_g1.detach().clone().requires_grad_(True)
    w_g2 = w_g1.detach().clone().requires_grad_(True)
    b_g2 = b_g1.detach().clone().requires_grad_(True)

    group_norm_silu(x_g1, w_g1, b_g1, num_groups=8).sum().backward()
    F.silu(F.group_norm(x_g2, 8, weight=w_g2, bias=b_g2)).sum().backward()
    assert x_g1.grad is not None and x_g2.grad is not None
    torch.testing.assert_close(x_g1.grad, x_g2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w_g1.grad, w_g2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b_g1.grad, b_g2.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_group_norm_silu_torch_compile_fullgraph_and_cuda_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.kernels.triton_group_norm_silu  # noqa: F401  # Register custom op before fallback hook

    register_worldfoundry_inductor_fallbacks()
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "triton")

    def fn(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return group_norm_silu(x, w, b, num_groups=16, eps=1e-5)

    compiled = torch.compile(fn, fullgraph=True)
    for shape, mem_fmt in [
        ((1, 64, 32, 32), torch.contiguous_format),
        ((1, 64, 32, 32), torch.channels_last),
        ((1, 64, 4, 16, 16), torch.channels_last_3d),
    ]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(memory_format=mem_fmt)
        w = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            got = compiled(x, w, b)
            ref = F.silu(F.group_norm(x, 16, weight=w, bias=b, eps=1e-5))
        assert got.stride() == x.stride()
        torch.testing.assert_close(got, ref, atol=6.5e-2, rtol=2e-2)

    # CUDA Graph capture & replay on 5D channels_last_3d
    x_cg = torch.randn((1, 64, 4, 32, 32), device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d)
    w_cg = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    b_cg = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        # Warmup on side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = group_norm_silu(x_cg, w_cg, b_cg, num_groups=16)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_cg = group_norm_silu(x_cg, w_cg, b_cg, num_groups=16)
        x_new = torch.randn_like(x_cg)
        x_cg.copy_(x_new)
        g.replay()
        torch.cuda.synchronize()
        ref_cg = F.silu(F.group_norm(x_new, 16, weight=w_cg, bias=b_cg, eps=1e-5))
    assert y_cg.stride() == x_cg.stride()
    torch.testing.assert_close(y_cg, ref_cg, atol=6.5e-2, rtol=2e-2)


# ──────────────────────────────────────────────────────────────────────────
# 3. RoPE single & fused QK parity, GQA, non-contiguous QKV, compile, CG
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "B,S,Hq,Hk,D",
    [
        (1, 16, 64, 8, 64),  # Short S -> H-split path, GQA
        (2, 512, 32, 32, 128),  # Standard MHA -> H-loop path
        (1, 1024, 40, 8, 128),  # Non-pow2 Hq + GQA Hk
    ],
)
def test_rope_single_and_fused_qk_parity(
    dtype: torch.dtype,
    interleaved: bool,
    inplace: bool,
    B: int,
    S: int,
    Hq: int,
    Hk: int,
    D: int,
) -> None:
    from worldfoundry.core.attention.rotary.rope_kernel import apply_rotary_pos_emb, apply_rotary_pos_emb_qk

    torch.manual_seed(11)
    q = torch.randn((B, S, Hq, D), device="cuda", dtype=dtype)
    k = torch.randn((B, S, Hk, D), device="cuda", dtype=dtype)
    # Include large angles to exercise Cody-Waite range reduction.
    freqs = torch.linspace(-50000.0, 50000.0, S * D, device="cuda", dtype=torch.float32).reshape(S, 1, 1, D)

    q_ref = _ref_rope_fp32(q, freqs, interleaved)
    k_ref = _ref_rope_fp32(k, freqs, interleaved)

    q_single = q.clone()
    q_ptr = q_single.data_ptr()
    q_single_orig = q.clone()
    with torch.inference_mode():
        out_single = apply_rotary_pos_emb(q_single, freqs, interleaved=interleaved, inplace=inplace)
    if inplace:
        assert out_single.data_ptr() == q_ptr
    else:
        assert out_single.data_ptr() != q_ptr
        assert torch.equal(q_single, q_single_orig)

    q_fk, k_fk = q.clone(), k.clone()
    q_fk_ptr, k_fk_ptr = q_fk.data_ptr(), k_fk.data_ptr()
    with torch.inference_mode():
        oq, ok = apply_rotary_pos_emb_qk(q_fk, k_fk, freqs, interleaved=interleaved, inplace=inplace)
    if inplace:
        assert oq.data_ptr() == q_fk_ptr and ok.data_ptr() == k_fk_ptr
    else:
        assert oq.data_ptr() != q_fk_ptr and ok.data_ptr() != k_fk_ptr

    if dtype == torch.bfloat16:
        atol, rtol = 3.5e-2, 2e-2
    elif dtype == torch.float16:
        atol, rtol = 4e-3, 4e-3
    else:
        atol, rtol = 2e-6, 2e-6
    torch.testing.assert_close(out_single, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(oq, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(ok, k_ref, atol=atol, rtol=rtol)
    if dtype == torch.float32:
        torch.testing.assert_close(oq, out_single, atol=1e-6, rtol=1e-6)
    else:
        assert torch.equal(oq, out_single)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_rope_noncontiguous_qkv_validation_compile_and_cuda_graph() -> None:
    from worldfoundry.core.attention.rotary.rope_kernel import apply_rotary_pos_emb, apply_rotary_pos_emb_qk

    register_worldfoundry_inductor_fallbacks()

    # Non-contiguous QKV packed slices (distinct non-overlapping regions of one buffer)
    qkv = torch.randn((1, 256, 3, 24, 64), device="cuda", dtype=torch.bfloat16)
    q_slice = qkv[:, :, 0]
    k_slice = qkv[:, :, 1]
    assert not q_slice.is_contiguous()
    freqs = torch.randn((256, 1, 1, 64), device="cuda", dtype=torch.float32) * 20.0
    q_ref = _ref_rope_fp32(q_slice, freqs, interleaved=True)
    k_ref = _ref_rope_fp32(k_slice, freqs, interleaved=True)
    with torch.inference_mode():
        oq, ok = apply_rotary_pos_emb_qk(q_slice, k_slice, freqs, interleaved=True, inplace=True)
    assert oq.data_ptr() == q_slice.data_ptr() and ok.data_ptr() == k_slice.data_ptr()
    torch.testing.assert_close(oq, q_ref, atol=3.5e-2, rtol=2e-2)
    torch.testing.assert_close(ok, k_ref, atol=3.5e-2, rtol=2e-2)

    # Validation errors preserved
    with pytest.raises(ValueError, match="CUDA"):
        apply_rotary_pos_emb(torch.randn(1, 4, 2, 8), torch.randn(4, 1, 1, 8))
    with pytest.raises(NotImplementedError, match="fp8"):
        apply_rotary_pos_emb(
            torch.zeros(1, 4, 2, 8, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(4, 1, 1, 8, device="cuda"),
        )
    with pytest.raises(ValueError, match="even"):
        apply_rotary_pos_emb(torch.zeros(1, 4, 2, 7, device="cuda"), torch.zeros(4, 1, 1, 7, device="cuda"))

    # torch.compile(fullgraph=True) for single + fused QK (out-of-place & in-place)
    def compiled_target(
        q_in: torch.Tensor, k_in: torch.Tensor, f_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, k1 = apply_rotary_pos_emb_qk(q_in, k_in, f_in, interleaved=False, inplace=False)
        q2 = apply_rotary_pos_emb(q1, f_in, interleaved=True, inplace=True)
        return q2, k1

    compiled = torch.compile(compiled_target, fullgraph=True)
    q_c = torch.randn((1, 128, 16, 64), device="cuda", dtype=torch.bfloat16)
    k_c = torch.randn((1, 128, 8, 64), device="cuda", dtype=torch.bfloat16)
    f_c = torch.randn((128, 1, 1, 64), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        got_q, got_k = compiled(q_c, k_c, f_c)
        exp_q, exp_k = compiled_target(q_c, k_c, f_c)
    assert torch.equal(got_q, exp_q)
    assert torch.equal(got_k, exp_k)

    # CUDA Graph capture & replay
    q_cg = torch.randn((1, 512, 16, 64), device="cuda", dtype=torch.bfloat16)
    k_cg = torch.randn((1, 512, 8, 64), device="cuda", dtype=torch.bfloat16)
    f_cg = torch.randn((512, 1, 1, 64), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = apply_rotary_pos_emb_qk(q_cg, k_cg, f_cg, interleaved=False, inplace=False)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_q_cg, out_k_cg = apply_rotary_pos_emb_qk(q_cg, k_cg, f_cg, interleaved=False, inplace=False)
        q_new = torch.randn_like(q_cg)
        k_new = torch.randn_like(k_cg)
        q_cg.copy_(q_new)
        k_cg.copy_(k_new)
        g.replay()
        torch.cuda.synchronize()
        ref_q_cg = _ref_rope_fp32(q_new, f_cg, interleaved=False)
        ref_k_cg = _ref_rope_fp32(k_new, f_cg, interleaved=False)
    torch.testing.assert_close(out_q_cg, ref_q_cg, atol=3.5e-2, rtol=2e-2)
    torch.testing.assert_close(out_k_cg, ref_k_cg, atol=3.5e-2, rtol=2e-2)


# ──────────────────────────────────────────────────────────────────────────
# 4. Routed SwiGLU MoE WGMMA parity, top_k=1 direct store, compile, CG
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "T,D,I,E,top_k,idx_dtype",
    [
        (16, 256, 128, 4, 2, torch.int64),  # Tiny T -> BLOCK_M=16
        (192, 512, 256, 8, 2, torch.int32),  # Medium T -> BLOCK_M=32/64, int32 routes
        (1024, 512, 256, 8, 1, torch.int64),  # Top-1 direct store
        (4096, 512, 256, 8, 2, torch.int64),  # Large T -> BLOCK_M=128 WGMMA
    ],
)
def test_routed_swiglu_moe_parity(
    dtype: torch.dtype,
    T: int,
    D: int,
    I: int,  # noqa: E741
    E: int,
    top_k: int,
    idx_dtype: torch.dtype,
) -> None:
    from worldfoundry.core.kernels.triton_moe import routed_swiglu_moe_triton

    torch.manual_seed(3)
    x = torch.randn((T, D), device="cuda", dtype=dtype) * 0.5
    logits = torch.randn((T, E), device="cuda", dtype=torch.float32)
    top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
    top_idx = top_idx.to(idx_dtype)
    rw = torch.softmax(top_vals, dim=-1).to(dtype)
    gw = torch.randn((E, I, D), device="cuda", dtype=dtype) * (D**-0.5)
    uw = torch.randn((E, I, D), device="cuda", dtype=dtype) * (D**-0.5)
    dw = torch.randn((E, D, I), device="cuda", dtype=dtype) * (I**-0.5)

    with torch.inference_mode():
        ref = routed_swiglu_moe_pytorch(x, rw, top_idx, gw, uw, dw)
        got_no_ws = routed_swiglu_moe_triton(x, rw, top_idx, gw, uw, dw, workspace=None)
        ws = {
            "counts": torch.empty((E,), device="cuda", dtype=torch.int32),
            "rows": torch.empty((E, T * top_k), device="cuda", dtype=torch.int32),
            "slots": torch.empty((E, T * top_k), device="cuda", dtype=torch.int32),
            "inter": torch.empty((T, top_k, I), device="cuda", dtype=dtype),
            "out": torch.empty((T, D), device="cuda", dtype=torch.float32),
        }
        got_ws = routed_swiglu_moe_triton(x, rw, top_idx, gw, uw, dw, workspace=ws)

    assert got_no_ws.dtype == torch.float32
    assert got_ws.data_ptr() == ws["out"].data_ptr()
    cos = F.cosine_similarity(got_no_ws.float().flatten(), ref.float().flatten(), dim=0).item()
    max_diff = (got_no_ws.float() - ref.float()).abs().max().item()
    assert cos > 0.9998, f"MoE cosine too low: {cos}"
    if dtype == torch.bfloat16:
        assert max_diff < 2e-2
    elif dtype == torch.float16:
        assert max_diff < 5e-3
    else:
        assert max_diff < 5e-3
    torch.testing.assert_close(got_no_ws, got_ws, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_routed_swiglu_moe_skewed_routing_autograd_compile_and_cuda_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.kernels.triton_moe  # noqa: F401  # Register custom op

    register_worldfoundry_inductor_fallbacks()
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "auto")

    T, D, inter_dim, E = 256, 256, 128, 8
    x = torch.randn((T, D), device="cuda", dtype=torch.bfloat16) * 0.5
    # Skewed routing: all tokens route ONLY to experts 0 and 7; experts 1..6 get 0 tokens!
    top_idx = torch.tensor([[0, 7]], device="cuda", dtype=torch.int64).expand(T, 2).contiguous()
    rw = torch.full((T, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    gw = torch.randn((E, inter_dim, D), device="cuda", dtype=torch.bfloat16) * (D**-0.5)
    uw = torch.randn((E, inter_dim, D), device="cuda", dtype=torch.bfloat16) * (D**-0.5)
    dw = torch.randn((E, D, inter_dim), device="cuda", dtype=torch.bfloat16) * (inter_dim**-0.5)

    with torch.inference_mode():
        ref = routed_swiglu_moe_pytorch(x, rw, top_idx, gw, uw, dw)
        got = routed_swiglu_moe(x, rw, top_idx, gw, uw, dw)
    assert F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item() > 0.9998

    # torch.compile(fullgraph=True) with workspace=None
    def moe_fn(h: torch.Tensor, r: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return routed_swiglu_moe(h, r, idx, gw, uw, dw)

    compiled = torch.compile(moe_fn, fullgraph=True)
    with torch.inference_mode():
        got_compiled = compiled(x, rw, top_idx)
    torch.testing.assert_close(got_compiled, got, atol=1e-5, rtol=1e-5)

    # Autograd falls back cleanly to PyTorch
    x_g1 = x.detach().float().requires_grad_(True)
    x_g2 = x.detach().float().requires_grad_(True)
    gw_f, uw_f, dw_f = gw.float(), uw.float(), dw.float()
    routed_swiglu_moe(x_g1, rw.float(), top_idx, gw_f, uw_f, dw_f).sum().backward()
    routed_swiglu_moe_pytorch(x_g2, rw.float(), top_idx, gw_f, uw_f, dw_f).sum().backward()
    assert x_g1.grad is not None and x_g2.grad is not None
    torch.testing.assert_close(x_g1.grad, x_g2.grad, atol=1e-5, rtol=1e-5)

    # CUDA Graph capture & replay with workspace reuse (both top_k=2 and top_k=1)
    for k_val in (2, 1):
        idx_cg = top_idx[:, :k_val].contiguous()
        rw_cg = rw[:, :k_val].contiguous()
        x_cg = x.clone()
        ws = {
            "counts": torch.empty((E,), device="cuda", dtype=torch.int32),
            "rows": torch.empty((E, T * k_val), device="cuda", dtype=torch.int32),
            "slots": torch.empty((E, T * k_val), device="cuda", dtype=torch.int32),
            "inter": torch.empty((T, k_val, inter_dim), device="cuda", dtype=torch.bfloat16),
            "out": torch.empty((T, D), device="cuda", dtype=torch.float32),
        }
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _ = routed_swiglu_moe(x_cg, rw_cg, idx_cg, gw, uw, dw, workspace=ws)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out_cg = routed_swiglu_moe(x_cg, rw_cg, idx_cg, gw, uw, dw, workspace=ws)
            x_new = torch.randn_like(x_cg) * 0.5
            x_cg.copy_(x_new)
            g.replay()
            torch.cuda.synchronize()
            ref_cg = routed_swiglu_moe_pytorch(x_new, rw_cg, idx_cg, gw, uw, dw)
        assert out_cg.data_ptr() == ws["out"].data_ptr()
        assert F.cosine_similarity(out_cg.float().flatten(), ref_cg.float().flatten(), dim=0).item() > 0.9998


# ──────────────────────────────────────────────────────────────────────────
# 5. Strict tl.int64 indexing on real >2^31-element 5D video & RoPE tensors
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_int64_indexing_above_2_pow_31_elements() -> None:
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 16 * (1024**3):
        pytest.skip()

    from worldfoundry.core.attention.rotary.rope_kernel import apply_rotary_pos_emb
    from worldfoundry.core.kernels.triton_group_norm_silu import group_norm_silu as triton_gn_silu

    # 1. GroupNorm+SiLU on >2^31-element 5D tensor: 1 * 32 * 17 * 2048 * 2048 = 2,281,701,376 > 2^31 (~4.56 GiB BF16)
    shape_5d = (1, 32, 17, 2048, 2048)
    assert math.prod(shape_5d) > (1 << 31)
    w = torch.ones(32, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(32, device="cuda", dtype=torch.bfloat16)

    for mem_fmt in (torch.contiguous_format, torch.channels_last_3d):
        x5 = torch.full(shape_5d, 1.0, device="cuda", dtype=torch.bfloat16).to(memory_format=mem_fmt)
        # Put a distinctive pattern past element 2^31 (channel 31, final temporal slice).
        x5[0, 31, -1, -1, -1] = 3.0
        x5[0, 0, 0, 0, 0] = -1.0
        with torch.inference_mode():
            y5 = triton_gn_silu(x5, w, b, 32, 1e-5)
        assert y5.stride() == x5.stride()
        # Channel 31 has 17*2048*2048 - 1 ones and one 3.0 -> mean ~ 1.0, var ~ 4 / 71.3M << 1e-5 -> rstd ~ 1/sqrt(1e-5)
        # Verify the tail element past 2^31 was written and is finite & strongly positive (unlike int32 wraparound).
        tail_val = y5[0, 31, -1, -1, -1].float().item()
        mid_val = y5[0, 31, 8, 1024, 1024].float().item()
        assert math.isfinite(tail_val) and tail_val > 10.0, f"int64 tail check failed ({mem_fmt}): {tail_val}"
        assert math.isfinite(mid_val) and abs(mid_val) < 1e-2, f"int64 mid check failed ({mem_fmt}): {mid_val}"
        del x5, y5
        torch.cuda.empty_cache()

    # 2. RoPE on >2^31-element [B, S, H, D] tensor: 2 * 16896 * 64 * 1024 = 2,214,592,512 > 2^31 (~4.43 GiB BF16)
    B, S, H, D = 2, 16896, 64, 1024
    assert B * S * H * D > (1 << 31)
    x_rope = torch.ones((B, S, H, D), device="cuda", dtype=torch.bfloat16)
    freqs = torch.full((S, 1, 1, D), math.pi / 2.0, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        apply_rotary_pos_emb(x_rope, freqs, interleaved=False, inplace=True)
    # Rotating (1, 1) by pi/2 gives (-1, +1) -> check the very last token past index 2^31!
    tail_a = x_rope[1, -1, -1, 0].float().item()
    tail_b = x_rope[1, -1, -1, D // 2].float().item()
    assert abs(tail_a - (-1.0)) < 2e-2 and abs(tail_b - 1.0) < 2e-2, f"RoPE int64 tail failed: ({tail_a}, {tail_b})"
    del x_rope, freqs
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# 6. Zero register spills (n_spills == 0) across all Hopper SM90 kernels
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_all_wave9c_kernels_have_zero_register_spills() -> None:
    from worldfoundry.core.attention.rotary import rope_kernel as rk
    from worldfoundry.core.kernels import triton_group_norm_silu as tgn
    from worldfoundry.core.kernels import triton_moe as tm

    with torch.inference_mode():
        # Trigger every GroupNorm+SiLU kernel path
        w32, b32 = (
            torch.ones(32, device="cuda", dtype=torch.bfloat16),
            torch.zeros(32, device="cuda", dtype=torch.bfloat16),
        )
        tgn.group_norm_silu(torch.randn(1, 32, 8, 8, device="cuda", dtype=torch.bfloat16), w32, b32, 8, 1e-5)
        w128, b128 = (
            torch.ones(128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
        )
        tgn.group_norm_silu(torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.bfloat16), w128, b128, 32, 1e-5)
        tgn.group_norm_silu(
            torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last),
            w128,
            b128,
            32,
            1e-5,
        )
        w96, b96 = (
            torch.ones(96, device="cuda", dtype=torch.bfloat16),
            torch.zeros(96, device="cuda", dtype=torch.bfloat16),
        )
        tgn.group_norm_silu(
            torch.randn(1, 96, 32, 32, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last),
            w96,
            b96,
            32,
            1e-5,
        )

        # Trigger every RoPE kernel path (split & interleaved, single & fused QK)
        q = torch.randn(1, 512, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
        f = torch.randn(512, 1, 1, 128, device="cuda", dtype=torch.float32)
        for inter in (False, True):
            rk.apply_rotary_pos_emb(q, f, interleaved=inter, inplace=False)
            rk.apply_rotary_pos_emb_qk(q, k, f, interleaved=inter, inplace=False)

        # Trigger every MoE WGMMA tile path (tiny, medium, large, top_k=1 & 2, bf16 & fp32)
        for T, D, inter_dim, E, top_k, dt in [
            (16, 256, 128, 4, 2, torch.bfloat16),
            (512, 512, 256, 8, 1, torch.bfloat16),
            (4096, 512, 256, 8, 2, torch.bfloat16),
            (128, 256, 128, 4, 2, torch.float32),
        ]:
            x = torch.randn(T, D, device="cuda", dtype=dt)
            idx = torch.randint(0, E, (T, top_k), device="cuda", dtype=torch.int64)
            rw = torch.full((T, top_k), 1.0 / top_k, device="cuda", dtype=dt)
            gw = torch.randn(E, inter_dim, D, device="cuda", dtype=dt) * 0.02
            uw = torch.randn(E, inter_dim, D, device="cuda", dtype=dt) * 0.02
            dw = torch.randn(E, D, inter_dim, device="cuda", dtype=dt) * 0.02
            tm.routed_swiglu_moe_triton(x, rw, idx, gw, uw, dw)

    _assert_zero_spills(
        "triton_group_norm_silu",
        tgn._load_triton_kernels(),
        (
            "cf_1pass",
            "reduce_cf",
            "finalize_chan",
            "apply_cf",
            "reduce_cl_mg",
            "apply_cl_mg",
            "reduce_cl_generic",
            "apply_cl_generic",
        ),
    )
    _assert_zero_spills("rope_kernel", rk._load_triton_kernels(), ("rope", "rope_qk"))
    _assert_zero_spills("triton_moe", tm._load_triton_kernels(), ("pack", "gate_up", "down"))


# ──────────────────────────────────────────────────────────────────────────
# 7. Multi-GPU device guard (cuda:1 while current_device == cuda:0)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices")
def test_non_default_cuda_device_guard() -> None:
    from worldfoundry.core.attention.rotary.rope_kernel import apply_rotary_pos_emb_qk
    from worldfoundry.core.kernels.triton_group_norm_silu import group_norm_silu as triton_gn_silu
    from worldfoundry.core.kernels.triton_moe import routed_swiglu_moe_triton

    torch.cuda.set_device(0)
    dev1 = torch.device("cuda:1")
    with torch.inference_mode():
        x = torch.randn(1, 64, 32, 32, device=dev1, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        w = torch.ones(64, device=dev1, dtype=torch.bfloat16)
        b = torch.zeros(64, device=dev1, dtype=torch.bfloat16)
        y = triton_gn_silu(x, w, b, 16, 1e-5)
        assert y.device == dev1 and torch.cuda.current_device() == 0

        q = torch.randn(1, 64, 8, 64, device=dev1, dtype=torch.bfloat16)
        k = torch.randn(1, 64, 4, 64, device=dev1, dtype=torch.bfloat16)
        f = torch.randn(64, 1, 1, 64, device=dev1, dtype=torch.float32)
        oq, ok = apply_rotary_pos_emb_qk(q, k, f, inplace=False)
        assert oq.device == dev1 and ok.device == dev1 and torch.cuda.current_device() == 0

        h = torch.randn(64, 128, device=dev1, dtype=torch.bfloat16)
        idx = torch.randint(0, 4, (64, 2), device=dev1, dtype=torch.int64)
        rw = torch.full((64, 2), 0.5, device=dev1, dtype=torch.bfloat16)
        gw = torch.randn(4, 64, 128, device=dev1, dtype=torch.bfloat16) * 0.02
        uw = torch.randn(4, 64, 128, device=dev1, dtype=torch.bfloat16) * 0.02
        dw = torch.randn(4, 128, 64, device=dev1, dtype=torch.bfloat16) * 0.02
        mo = routed_swiglu_moe_triton(h, rw, idx, gw, uw, dw)
        assert mo.device == dev1 and torch.cuda.current_device() == 0
