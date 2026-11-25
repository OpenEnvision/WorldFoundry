# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Wave 9G FP8 Quantization, Attention Dispatch, and Shared Transformer Ops.

Covers:
1. ``worldfoundry.core.acceleration.quantization.triton_fp8_linear`` (Hopper SM90 WGMMA FP8 GEMM)
2. ``worldfoundry.core.acceleration.quantization.triton_fp8`` (2-chunk, wide-row, 3-pt GELU, fused silu_mul)
3. ``worldfoundry.core.attention.backends.dispatch`` (fast layout adapters, zero-lock quarantine, monkeypatch hygiene)
4. ``worldfoundry.core.nn.transformer.ops`` & ``triton_diffusion`` (fused PostSA RMSNorm & GatedAttention)
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.nn.transformer.ops import PytorchGatedAttention, PytorchPostSAFunction
from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_jit_zero_spills(label: str, jit_fn: object) -> None:
    dev_caches = getattr(jit_fn, "device_caches", None)
    assert dev_caches, f"{label} had no compiled device cache entries"
    checked = 0
    for entry in dev_caches.values():
        cache_dict = entry[0] if isinstance(entry, tuple) else entry
        if not isinstance(cache_dict, dict):
            continue
        for compiled in cache_dict.values():
            checked += 1
            spills = int(getattr(compiled, "n_spills", 0))
            regs = int(getattr(compiled, "n_regs", 0))
            assert spills == 0, f"{label} spilled {spills} bytes (n_regs={regs})"
    assert checked > 0, f"{label} had zero compiled variants"


# ──────────────────────────────────────────────────────────────────────────
# 1. CPU-only & absent-Triton subprocess import hygiene
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cpu_only_and_absent_triton_subprocess_imports_and_fallbacks() -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = """
import importlib.abc
import sys

class _BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("blocked triton for CPU-only CI verification")
        return None

sys.meta_path.insert(0, _BlockTriton())

import torch
import torch.nn.functional as F
from worldfoundry.core.acceleration.quantization import triton_fp8, triton_fp8_linear
from worldfoundry.core.attention.backends import dispatch
from worldfoundry.core.nn.transformer.ops import PytorchGatedAttention, PytorchPostSAFunction

assert "triton" not in sys.modules
assert "triton.language" not in sys.modules
assert not torch.cuda.is_initialized()

# Custom ops registered cleanly without touching Triton
assert hasattr(torch.ops.worldfoundry, "triton_fp8_linear")
assert hasattr(torch.ops.worldfoundry, "quantize_rowwise_fp8")
assert hasattr(torch.ops.worldfoundry, "fused_residual_gate_add_rms_norm")
assert hasattr(torch.ops.worldfoundry, "fused_head_sigmoid_gate")

# CPU fallback for fp8_linear (including E5M2 x E5M2)
a = torch.randn(4, 16, dtype=torch.bfloat16).to(torch.float8_e5m2)
w = torch.randn(8, 16, dtype=torch.bfloat16).to(torch.float8_e5m2)
sa = torch.rand(4, 1) * 0.1 + 0.01
sw = torch.rand(8) * 0.1 + 0.01
b = torch.randn(8, dtype=torch.bfloat16)
out = triton_fp8_linear.fp8_linear(a, w, sa, sw, b, torch.bfloat16)
ref = ((a.float() @ w.float().T) * sa * sw + b.float()).to(torch.bfloat16)
assert torch.equal(out, ref)

# CPU fallback for PostSA & GatedAttention
res = torch.randn(2, 5, 16)
val = torch.randn_like(res)
gate = torch.randn(2, 1, 16)
wt = torch.randn(16)
with torch.inference_mode():
    po, pn = PytorchPostSAFunction()(res, val, wt, 1e-6, gate)
    exp = res + val * gate
    assert torch.equal(po, exp)
    assert torch.equal(pn, F.rms_norm(exp, (16,), wt, 1e-6))

    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = 4
            self.dim_head = 4
            self.to_gate_logits = torch.nn.Linear(16, 4)

    mod = _Mod()
    go = PytorchGatedAttention()(res, val, mod)
    g_ref = (val.view(2, 5, 4, 4) * (2.0 * torch.sigmoid(mod.to_gate_logits(res))).unsqueeze(-1)).view(2, 5, 16)
    assert torch.equal(go, g_ref)

# CPU dispatch works cleanly
q = torch.randn(1, 2, 4, 8)
ao = dispatch.attention_forward(q, q, q, compatibility_mode=True)
assert ao.shape == q.shape
assert "triton" not in sys.modules
assert not torch.cuda.is_initialized()
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
# 2. Target 1: Hopper SM90 WGMMA triton_fp8_linear parity, compile, CG
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "a_dtype,w_dtype",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float8_e4m3fn, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "M,K,N",
    [
        (34, 80, 48),  # Small non-pow2 (ptr path)
        (64, 128, 96),  # Medium (ptr path)
        (256, 512, 256),  # TMA aligned path (BK=256)
        (257, 300, 193),  # Unaligned K % 16 != 0 (ptr path)
        (512, 160, 256),  # TMA aligned path (BK=128)
    ],
)
def test_triton_fp8_linear_parity(
    a_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    has_bias: bool,
    M: int,
    K: int,
    N: int,
) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip()
    from worldfoundry.core.acceleration.quantization.triton_fp8_linear import fp8_linear

    torch.manual_seed(17)
    a = (torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 1.5).clamp(-16, 16).to(a_dtype)
    w = (torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 1.5).clamp(-16, 16).to(w_dtype)
    sa = torch.rand((M, 1), device="cuda", dtype=torch.float32) * 0.05 + 0.01
    sw = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.05 + 0.01
    bias = torch.randn((N,), device="cuda", dtype=out_dtype) if has_bias else None

    with torch.inference_mode():
        got = fp8_linear(a, w, sa, sw, bias, out_dtype)
        ref = (a.float() @ w.float().T) * sa * sw
        if bias is not None:
            ref = ref + bias.float()
        ref = ref.to(out_dtype)

    assert got.shape == (M, N)
    assert got.dtype == out_dtype
    if out_dtype == torch.bfloat16:
        torch.testing.assert_close(got, ref, atol=3.5e-2, rtol=1.6e-2)
    elif out_dtype == torch.float16:
        torch.testing.assert_close(got, ref, atol=5e-3, rtol=5e-3)
    else:
        torch.testing.assert_close(got, ref, atol=5e-4, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_fp8_linear_fp32_accum_large_k_edges_compile_and_cg() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip()
    from worldfoundry.core.acceleration.quantization.triton_fp8_linear import fp8_linear

    register_worldfoundry_inductor_fallbacks()

    # Large-K all-positive stress test: verifies true FP32 accumulation (no 22-bit mantissa drift)
    torch.manual_seed(0)
    a_pos = (torch.rand((256, 13824), device="cuda", dtype=torch.bfloat16) * 2.0 + 0.5).to(torch.float8_e4m3fn)
    w_pos = (torch.rand((256, 13824), device="cuda", dtype=torch.bfloat16) * 2.0 + 0.5).to(torch.float8_e4m3fn)
    sa_p = torch.ones((256,), device="cuda", dtype=torch.float32)
    sw_p = torch.ones((1, 256), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        got_pos = fp8_linear(a_pos, w_pos, sa_p, sw_p, None, torch.float32)
        ref_pos = a_pos.float() @ w_pos.float().T
    rel_err = (got_pos - ref_pos).abs().max().item() / ref_pos.abs().max().item()
    assert rel_err < 2e-3, f"FP32 accum drifted on K=13824 all-positive inputs: rel_err={rel_err:.4e}"

    # Empty & scalar-scale edge cases
    empty_a = torch.empty((0, 64), device="cuda", dtype=torch.float8_e4m3fn)
    w64 = torch.zeros((32, 64), device="cuda", dtype=torch.float8_e4m3fn)
    assert fp8_linear(
        empty_a, w64, torch.ones((0, 1), device="cuda"), torch.ones((32,), device="cuda"), None, torch.bfloat16
    ).shape == (0, 32)
    k0_a = torch.empty((4, 0), device="cuda", dtype=torch.float8_e4m3fn)
    k0_w = torch.empty((8, 0), device="cuda", dtype=torch.float8_e4m3fn)
    b8 = torch.randn((8,), device="cuda", dtype=torch.bfloat16)
    out_k0 = fp8_linear(
        k0_a, k0_w, torch.ones((4, 1), device="cuda"), torch.ones((8,), device="cuda"), b8, torch.bfloat16
    )
    assert torch.equal(out_k0, b8.unsqueeze(0).expand(4, 8))

    # Scalar scales [1]
    a_sc = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
    w_sc = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
    s1 = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    s2 = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    got_sc = fp8_linear(a_sc, w_sc, s1, s2, None, torch.bfloat16)
    ref_sc = ((a_sc.float() @ w_sc.float().T) * 0.125).to(torch.bfloat16)
    torch.testing.assert_close(got_sc, ref_sc, atol=3.5e-2, rtol=2e-2)

    # torch.compile(fullgraph=True) + CUDA Graph capture & replay on TMA E5M2 path
    M, K, N = 256, 512, 256
    a_c = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
    w_c = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
    sa_c = torch.rand((M, 1), device="cuda", dtype=torch.float32) * 0.05 + 0.01
    sw_c = torch.rand((N,), device="cuda", dtype=torch.float32) * 0.05 + 0.01
    b_c = torch.randn((N,), device="cuda", dtype=torch.bfloat16)

    def linear_fn(x: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        return fp8_linear(x, w_c, sx, sw_c, b_c, torch.bfloat16)

    compiled = torch.compile(linear_fn, fullgraph=True)
    with torch.inference_mode():
        got_comp = compiled(a_c, sa_c)
        got_eager = linear_fn(a_c, sa_c)
    assert torch.equal(got_comp, got_eager)

    # CUDA Graph capture & replay
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = fp8_linear(a_c, w_c, sa_c, sw_c, b_c, torch.bfloat16)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_cg = fp8_linear(a_c, w_c, sa_c, sw_c, b_c, torch.bfloat16)
        a_new = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
        a_c.copy_(a_new)
        g.replay()
        torch.cuda.synchronize()
        ref_cg = ((a_new.float() @ w_c.float().T) * sa_c * sw_c + b_c.float()).to(torch.bfloat16)
    torch.testing.assert_close(out_cg, ref_cg, atol=3.5e-2, rtol=2e-2)


# ──────────────────────────────────────────────────────────────────────────
# 3. Target 2: triton_fp8 (2-chunk, wide rows, 3-pt GELU, silu_mul)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("activation", ["identity", "gelu", "gelu_tanh", "silu_mul"])
@pytest.mark.parametrize("cols", [128, 5120, 13824, 20480, 24576, 32768])
def test_triton_fp8_quantize_all_activations_and_widths_parity(
    fp8_dtype: torch.dtype,
    in_dtype: torch.dtype,
    activation: str,
    cols: int,
) -> None:
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    # Keep test lightweight across wide widths
    rows = 16 if cols >= 20480 else 32
    in_cols = cols * 2 if activation == "silu_mul" else cols
    if in_cols > 40960:
        in_cols = 36864  # exercises tiled silu_mul (K=18432 > 16384) without huge allocation
        cols = in_cols // 2

    torch.manual_seed(23)
    x = torch.randn((rows, in_cols), device="cuda", dtype=in_dtype) * 2.0
    # Tricky rows: all-zero, all-negative covering GELU minimum, all < -2, near -0.752
    x[0].zero_()
    x[1] = -torch.rand(in_cols, device="cuda", dtype=in_dtype) * 5.0 - 0.1
    x[2] = -torch.rand(in_cols, device="cuda", dtype=in_dtype) * 4.0 - 2.0
    x[3].fill_(-0.752)

    with torch.inference_mode():
        codes, scales = quantize_rowwise_fp8_triton(x, fp8_dtype, activation=activation)
        if activation == "identity":
            act_ref = x
        elif activation == "gelu":
            act_ref = F.gelu(x, approximate="none")
        elif activation == "gelu_tanh":
            act_ref = F.gelu(x, approximate="tanh")
        else:
            act_ref = F.silu(x[:, :cols]) * x[:, cols:]
        fp8_max = float(torch.finfo(fp8_dtype).max)
        tiny = float(torch.finfo(torch.float32).tiny)
        ref_scales = (act_ref.float().abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(tiny)
        ref_codes = (act_ref.float() / ref_scales).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    assert codes.shape == (rows, cols)
    assert codes.dtype == fp8_dtype
    assert scales.shape == (rows, 1) and scales.dtype == torch.float32
    assert torch.isfinite(codes.float()).all() and torch.isfinite(scales).all()
    assert torch.count_nonzero(codes[0].float()) == 0
    torch.testing.assert_close(scales, ref_scales, rtol=2e-3, atol=1e-6)
    deq_got = codes.float() * scales
    deq_ref = ref_codes.float() * ref_scales
    rel = (deq_got - deq_ref).norm().item() / deq_ref.norm().clamp_min(1e-12).item()
    assert rel < 3e-3, f"dequant rel err too large for {activation}/{in_dtype}/{fp8_dtype}/K={cols}: {rel:.3e}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_fp8_validation_compile_and_cuda_graph() -> None:
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    register_worldfoundry_inductor_fallbacks()

    with pytest.raises(ValueError, match="even"):
        quantize_rowwise_fp8_triton(torch.randn(4, 15, device="cuda", dtype=torch.bfloat16), activation="silu_mul")
    with pytest.raises(ValueError, match="CUDA"):
        quantize_rowwise_fp8_triton(torch.randn(4, 16, dtype=torch.bfloat16))

    # Empty cols for silu_mul initializes tiny scale
    c0, s0 = quantize_rowwise_fp8_triton(torch.empty(3, 0, device="cuda", dtype=torch.bfloat16), activation="silu_mul")
    assert c0.shape == (3, 0) and torch.all(s0 == float(torch.finfo(torch.float32).tiny))

    # torch.compile(fullgraph=True) on silu_mul + gelu_tanh
    def quant_fn(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1, s1 = quantize_rowwise_fp8_triton(x1, torch.float8_e4m3fn, activation="silu_mul")
        c2, s2 = quantize_rowwise_fp8_triton(x2, torch.float8_e4m3fn, activation="gelu_tanh")
        return c1, s1, c2, s2

    compiled = torch.compile(quant_fn, fullgraph=True)
    x1 = torch.randn(32, 2 * 5120, device="cuda", dtype=torch.bfloat16)
    x2 = torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        gc1, gs1, gc2, gs2 = compiled(x1, x2)
        ec1, es1, ec2, es2 = quant_fn(x1, x2)
    assert torch.equal(gc1.view(torch.uint8), ec1.view(torch.uint8)) and torch.equal(gs1, es1)
    assert torch.equal(gc2.view(torch.uint8), ec2.view(torch.uint8)) and torch.equal(gs2, es2)

    # CUDA Graph capture & replay on silu_mul
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = quantize_rowwise_fp8_triton(x1, torch.float8_e4m3fn, activation="silu_mul")
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cg_c, cg_s = quantize_rowwise_fp8_triton(x1, torch.float8_e4m3fn, activation="silu_mul")
        x1_new = torch.randn_like(x1)
        x1.copy_(x1_new)
        g.replay()
        torch.cuda.synchronize()
        exp_c, exp_s = quantize_rowwise_fp8_triton(x1_new, torch.float8_e4m3fn, activation="silu_mul")
    assert torch.equal(cg_c.view(torch.uint8), exp_c.view(torch.uint8))
    assert torch.equal(cg_s, exp_s)


# ──────────────────────────────────────────────────────────────────────────
# 4. Target 3: dispatch.py layout parity, telemetry, quarantine, monkeypatch
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dispatch_fast_rearrange_exact_value_and_stride_parity_vs_einops() -> None:
    torch.manual_seed(5)
    x4 = torch.randn(2, 8, 16, 32)
    x4_nc = torch.randn(2, 16, 8, 32).transpose(1, 2)
    x3 = torch.randn(2, 16, 8 * 32)
    x3_nc = torch.randn(2, 8 * 32, 16).transpose(1, 2)

    for q_t, src, dst, dims in [
        (x4, "b n s d", "b s n d", None),
        (x4_nc, "b n s d", "b s n d", None),
        (x4, "b s n d", "b n s d", None),
        (x3, "b s (n d)", "b s n d", {"n": 8}),
        (x3_nc, "b s (n d)", "b s n d", {"n": 8}),
        (x3, "b s (n d)", "b n s d", {"n": 8}),
        (x3_nc, "b s (n d)", "b n s d", {"n": 8}),
        (x3, "b s (h d)", "b s n d", {"h": 8}),
        (x4, "b s n d", "b s (n d)", {"n": 16}),  # note: x4 has shape[2]=16
        (x4, "b n s d", "b s (n d)", {"n": 8}),
        (x4_nc, "b s n d", "b s (n d)", None),
        (x4_nc, "b n s d", "b s (n d)", None),
        # General fallback pattern not on fast path
        (torch.randn(2, 4, 8, 16), "b h s d", "b s h d", None),
    ]:
        # Validate rearrange_qkv and rearrange_out
        if src == "b s (h d)" and dst == "b s n d":
            # einops needs matching axis names, so test b s (h d) -> b s h d
            continue
        q_got, _, _ = dispatch.rearrange_qkv(q_t, q_t, q_t, src, src, src, dst, dims)
        q_ref = rearrange(q_t, f"{src} -> {dst}", **(dims or {}))
        assert torch.equal(q_got, q_ref), f"value mismatch on {src} -> {dst}"
        assert q_got.shape == q_ref.shape and q_got.stride() == q_ref.stride(), f"stride mismatch on {src} -> {dst}"

        out_got = dispatch.rearrange_out(q_t, out_pattern=dst, required_out_pattern=src, dims=dims)
        assert torch.equal(out_got, q_ref)
        assert out_got.stride() == q_ref.stride()


@pytest.mark.unit
def test_dispatch_preserves_quarantine_env_override_and_live_sys_modules_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch.clear_attention_dispatch_cache()
    dispatch.reset_attention_provider_runtime()

    # 1. Live sys.modules monkeypatch picked up WITHOUT calling clear_attention_dispatch_cache()
    calls: list[str] = []
    q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)

    def make_fake_fa3(tag: str) -> types.ModuleType:
        mod = types.ModuleType("flash_attn_interface")

        def _fn(qq, kk, vv, softmax_scale=None):
            del kk, vv, softmax_scale
            calls.append(tag)
            return qq + 1.0

        setattr(mod, "flash_attn_func", _fn)
        return mod

    monkeypatch.setitem(sys.modules, "flash_attn_interface", make_fake_fa3("first"))
    out1 = dispatch.flash_attention_3(q, q, q)
    assert calls == ["first"]
    assert torch.equal(out1, q + 1.0)

    # Swap sys.modules WITHOUT clearing cache — must observe new module immediately!
    monkeypatch.setitem(sys.modules, "flash_attn_interface", make_fake_fa3("second"))
    out2 = dispatch.flash_attention_3(q, q, q)
    assert calls == ["first", "second"]
    assert torch.equal(out2, q + 1.0)

    # 2. Live env var override for WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE
    monkeypatch.setenv("WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE", "512")
    assert dispatch._short_attention_threshold(torch.device("cpu")) == 512
    monkeypatch.setenv("WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE", "not-an-int")
    assert dispatch._short_attention_threshold(torch.device("cpu")) == 128
    monkeypatch.delenv("WORLDFOUNDRY_ATTENTION_MIN_FUSED_SEQUENCE", raising=False)

    # 3. Quarantine for ImportError (unavailable backend) & RuntimeError (failed signature) + OOM propagation
    dispatch.clear_attention_dispatch_cache()
    dispatch.reset_attention_provider_runtime()
    sentinel = torch.ones(1, 2, 8, 8, dtype=torch.float16)
    monkeypatch.setattr(dispatch, "torch_sdpa", lambda *a, **kw: sentinel)
    monkeypatch.setattr(dispatch, "_select_attention_backends_cached", lambda *a, **kw: ("flash_attention_2", "torch"))

    def _unsupported(*a, **kw):
        raise RuntimeError("kernel not supported for this shape")

    monkeypatch.setattr(dispatch, "flash_attention_2", _unsupported)
    with pytest.warns(RuntimeWarning, match="trying the next eligible backend"):
        assert dispatch.attention_forward(sentinel, sentinel, sentinel) is sentinel
    # Second identical call must be skipped via _FAILED_ATTENTION_SIGNATURES without warning
    assert dispatch.attention_forward(sentinel, sentinel, sentinel) is sentinel
    rt = dispatch.attention_provider_runtime_report()
    assert rt["flash_attention_2"]["fallbacks"] == 1
    assert rt["flash_attention_2"]["quarantined_skips"] == 1

    # OOM must propagate and never be quarantined
    dispatch.clear_attention_dispatch_cache()
    monkeypatch.setattr(
        dispatch, "flash_attention_2", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        dispatch.attention_forward(sentinel, sentinel, sentinel)
    assert len(dispatch._FAILED_ATTENTION_SIGNATURES) == 0

    # ImportError quarantines backend in _UNAVAILABLE_BACKENDS_TUPLE and clears cleanly
    monkeypatch.setattr(
        dispatch, "flash_attention_2", lambda *a, **kw: (_ for _ in ()).throw(ImportError("no flash_attn"))
    )
    with pytest.warns(RuntimeWarning, match="trying the next eligible backend"):
        assert dispatch.attention_forward(sentinel, sentinel, sentinel) is sentinel
    assert dispatch._unavailable_backends_snapshot() == ("flash_attention_2",)
    dispatch.clear_attention_dispatch_cache()
    assert dispatch._unavailable_backends_snapshot() == ()


# ──────────────────────────────────────────────────────────────────────────
# 5. Target 4: Fused PytorchPostSAFunction & PytorchGatedAttention
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize(
    "shape,gate_shape",
    [
        ((2, 64, 1536), (2, 1, 1536)),
        ((1, 256, 4096), (1, 1, 4096)),
        ((1, 128, 5120), (1, 128, 5120)),
        ((2, 32, 3072), (2, 1, 1)),
        ((1, 16, 128), ()),
    ],
)
def test_fused_post_sa_parity(
    dtype: torch.dtype,
    has_weight: bool,
    shape: tuple[int, ...],
    gate_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(31)
    res = torch.randn(shape, device="cuda", dtype=dtype)
    val = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)
    weight = torch.randn((shape[-1],), device="cuda", dtype=dtype) if has_weight else None

    fn = PytorchPostSAFunction()
    with torch.inference_mode():
        out_got, norm_got = fn(res, val, weight, 1e-6, gate)
        out_ref = res + val * gate
        norm_ref = F.rms_norm(out_ref, (shape[-1],), weight=weight, eps=1e-6)

    assert torch.equal(out_got, out_ref), "PostSA residual+value*gate must be bit-exact"
    if dtype == torch.bfloat16:
        torch.testing.assert_close(norm_got, norm_ref, atol=3.5e-2, rtol=1.6e-2)
    elif dtype == torch.float16:
        torch.testing.assert_close(norm_got, norm_ref, atol=3e-3, rtol=3e-3)
    else:
        torch.testing.assert_close(norm_got, norm_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("B,S,H,D", [(2, 64, 32, 128), (1, 128, 40, 128), (2, 32, 12, 96), (1, 16, 8, 64)])
def test_fused_gated_attention_parity(dtype: torch.dtype, B: int, S: int, H: int, D: int) -> None:
    class _AttnMod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = H
            self.dim_head = D
            self.to_gate_logits = torch.nn.Linear(H * D, H, device="cuda", dtype=dtype)

    torch.manual_seed(37)
    mod = _AttnMod()
    res = torch.randn((B, S, H * D), device="cuda", dtype=dtype)
    aout = torch.randn((B, S, H * D), device="cuda", dtype=dtype)
    fn = PytorchGatedAttention()

    with torch.inference_mode():
        got = fn(res, aout, mod)
        logits = mod.to_gate_logits(res)
        ref = (aout.view(B, S, H, D) * (2.0 * torch.sigmoid(logits)).unsqueeze(-1)).view(B, S, H * D)

    if dtype == torch.bfloat16:
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=1e-2)
    elif dtype == torch.float16:
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    else:
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_transformer_ops_autograd_compile_and_cuda_graph() -> None:
    register_worldfoundry_inductor_fallbacks()

    post_sa = PytorchPostSAFunction()
    gated = PytorchGatedAttention()

    class _AttnMod(torch.nn.Module):
        def __init__(self, dtype: torch.dtype) -> None:
            super().__init__()
            self.heads = 16
            self.dim_head = 64
            self.to_gate_logits = torch.nn.Linear(1024, 16, device="cuda", dtype=dtype)

    # 1. Autograd falls back cleanly and produces exact gradients
    r1 = torch.randn(2, 16, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    v1 = torch.randn_like(r1, requires_grad=True)
    g1 = torch.randn(2, 1, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    w1 = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    r2, v2, g2, w2 = (t.detach().clone().requires_grad_(True) for t in (r1, v1, g1, w1))

    o1, n1 = post_sa(r1, v1, w1, 1e-6, g1)
    (o1.sum() + n1.sum()).backward()
    o2 = r2 + v2 * g2
    n2 = F.rms_norm(o2, (1024,), w2, 1e-6)
    (o2.sum() + n2.sum()).backward()
    torch.testing.assert_close(r1.grad, r2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v1.grad, v2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(g1.grad, g2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)

    mod_f32 = _AttnMod(torch.float32)
    ra1 = torch.randn(2, 16, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    aa1 = torch.randn_like(ra1, requires_grad=True)
    ra2, aa2 = ra1.detach().clone().requires_grad_(True), aa1.detach().clone().requires_grad_(True)
    gated(ra1, aa1, mod_f32).sum().backward()
    mod_f32.zero_grad()
    ref_ga = (aa2.view(2, 16, 16, 64) * (2.0 * torch.sigmoid(mod_f32.to_gate_logits(ra2))).unsqueeze(-1)).view(
        2, 16, 1024
    )
    ref_ga.sum().backward()
    torch.testing.assert_close(ra1.grad, ra2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(aa1.grad, aa2.grad, atol=1e-5, rtol=1e-5)

    # 2. torch.compile(fullgraph=True) + CUDA Graph capture & replay
    mod_bf16 = _AttnMod(torch.bfloat16)
    res_c = torch.randn(1, 128, 1024, device="cuda", dtype=torch.bfloat16)
    val_c = torch.randn_like(res_c)
    gate_c = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    wt_c = torch.randn(1024, device="cuda", dtype=torch.bfloat16)

    def block_ops(r: torch.Tensor, v: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g_out = gated(r, v, mod_bf16)
        return post_sa(r, g_out, wt_c, 1e-6, gt)

    compiled = torch.compile(block_ops, fullgraph=True)
    with torch.inference_mode():
        co, cn = compiled(res_c, val_c, gate_c)
        eo, en = block_ops(res_c, val_c, gate_c)
    assert torch.equal(co, eo)
    assert torch.equal(cn, en)

    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = block_ops(res_c, val_c, gate_c)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cg_o, cg_n = block_ops(res_c, val_c, gate_c)
        res_new = torch.randn_like(res_c)
        val_new = torch.randn_like(val_c)
        res_c.copy_(res_new)
        val_c.copy_(val_new)
        g.replay()
        torch.cuda.synchronize()
        exp_o, exp_n = block_ops(res_new, val_new, gate_c)
    assert torch.equal(cg_o, exp_o)
    assert torch.equal(cg_n, exp_n)


# ──────────────────────────────────────────────────────────────────────────
# 6. Zero register spills (n_spills == 0) across all Wave 9G Hopper kernels
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_all_wave9g_hopper_kernels_have_zero_register_spills() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip()
    from worldfoundry.core.acceleration.quantization import triton_fp8, triton_fp8_linear
    from worldfoundry.core.kernels import triton_diffusion

    with torch.inference_mode():
        # Trigger both TMA (BK=256 and BK=128) and pointer FP8 GEMM paths
        for M, K, N, dt in [
            (34, 80, 48, torch.float8_e5m2),
            (256, 512, 256, torch.float8_e4m3fn),
            (256, 512, 256, torch.float8_e5m2),
            (256, 160, 256, torch.float8_e4m3fn),
            (257, 300, 193, torch.float8_e4m3fn),
        ]:
            a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(dt)
            w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).clamp(-10, 10).to(dt)
            sa = torch.ones((M, 1), device="cuda", dtype=torch.float32)
            sw = torch.ones((N,), device="cuda", dtype=torch.float32)
            triton_fp8_linear.fp8_linear(a, w, sa, sw, None, torch.bfloat16)

        # Trigger single, two_chunk, tiled, silu_mul_single, silu_mul_tiled
        for cols, act in [
            (4096, "identity"),
            (5120, "gelu_tanh"),
            (13824, "identity"),
            (13824, "gelu_tanh"),
            (13824, "gelu"),
            (20480, "identity"),
            (24576, "gelu_tanh"),
            (32768, "gelu"),
            (2 * 5120, "silu_mul"),
            (2 * 18432, "silu_mul"),
        ]:
            x = torch.randn((8, cols), device="cuda", dtype=torch.bfloat16)
            triton_fp8.quantize_rowwise_fp8_triton(x, torch.float8_e4m3fn, activation=act)

        # Trigger PostSA and GatedAttention kernels
        for D in (3072, 4096, 5120):
            r = torch.randn((1, 16, D), device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(r)
            gt = torch.randn((1, 1, D), device="cuda", dtype=torch.bfloat16)
            wt = torch.ones((D,), device="cuda", dtype=torch.bfloat16)
            triton_diffusion.residual_gate_add_rms_norm(r, v, wt, 1e-6, gt)
        a_out = torch.randn((1, 16, 40 * 128), device="cuda", dtype=torch.bfloat16)
        g_log = torch.randn((1, 16, 40), device="cuda", dtype=torch.bfloat16)
        triton_diffusion.head_sigmoid_gate(a_out, g_log, 40, 128)

    lin_k = triton_fp8_linear._load_triton_kernels()
    _assert_jit_zero_spills("triton_fp8_linear.tma", lin_k["tma"])
    _assert_jit_zero_spills("triton_fp8_linear.ptr", lin_k["ptr"])

    fp8_k = triton_fp8._load_triton_kernels()
    for name in ("single", "two_chunk", "tiled", "silu_mul_single", "silu_mul_tiled"):
        _assert_jit_zero_spills(f"triton_fp8.{name}", fp8_k[name])

    _assert_jit_zero_spills(
        "triton_diffusion.residual_gate_add_rms_norm", triton_diffusion._residual_gate_add_rms_norm_kernel
    )
    _assert_jit_zero_spills("triton_diffusion.head_sigmoid_gate", triton_diffusion._head_sigmoid_gate_kernel)


# ──────────────────────────────────────────────────────────────────────────
# 7. Strict tl.int64 indexing above 2^31 elements + multi-GPU device guard
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_int64_indexing_above_2_pow_31_elements_wave9g() -> None:
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 16 * (1024**3):
        pytest.skip()

    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
    from worldfoundry.core.kernels.triton_diffusion import head_sigmoid_gate

    # >2^31 elements: 524288 * 4096 = 2,147,483,648; use 524300 * 4096 = 2,147,532,800 (~4.3 GiB BF16)
    rows, cols = 524300, 4096
    assert rows * cols > (1 << 31)
    x = torch.ones((rows, cols), device="cuda", dtype=torch.bfloat16)
    x[-1, -1] = 8.0
    with torch.inference_mode():
        codes, scales = quantize_rowwise_fp8_triton(x, torch.float8_e4m3fn, activation="identity")
    tail_val = (codes[-1, -1].float() * scales[-1, 0]).item()
    assert math.isfinite(tail_val) and abs(tail_val - 8.0) < 0.25, f"FP8 quant int64 tail failed: {tail_val}"
    del codes, scales

    # head_sigmoid_gate on >2^31 elements (reshape x as [1, 524300, 32*128])
    x3 = x.view(1, rows, 32 * 128)
    logits = torch.zeros((1, rows, 32), device="cuda", dtype=torch.bfloat16)
    logits[0, -1, -1] = 10.0  # 2 * sigmoid(10) ~= 2.0, so tail 8.0 * 2.0 ~= 16.0
    with torch.inference_mode():
        gated = head_sigmoid_gate(x3, logits, 32, 128)
    tail_g = gated[0, -1, -1].float().item()
    assert math.isfinite(tail_g) and abs(tail_g - 16.0) < 0.5, f"head_sigmoid_gate int64 tail failed: {tail_g}"
    del x, x3, logits, gated
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices")
def test_non_default_cuda_device_guard_wave9g() -> None:
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
    from worldfoundry.core.acceleration.quantization.triton_fp8_linear import fp8_linear

    torch.cuda.set_device(0)
    dev1 = torch.device("cuda:1")
    with torch.inference_mode():
        x = torch.randn((32, 5120), device=dev1, dtype=torch.bfloat16)
        c, s = quantize_rowwise_fp8_triton(x, torch.float8_e4m3fn, activation="gelu_tanh")
        assert c.device == dev1 and s.device == dev1 and torch.cuda.current_device() == 0

        a = torch.randn((128, 256), device=dev1, dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
        w = torch.randn((128, 256), device=dev1, dtype=torch.bfloat16).clamp(-10, 10).to(torch.float8_e5m2)
        out = fp8_linear(
            a,
            w,
            torch.ones((128, 1), device=dev1),
            torch.ones((128,), device=dev1),
            None,
            torch.bfloat16,
        )
        assert out.device == dev1 and torch.cuda.current_device() == 0

        res = torch.randn((1, 32, 512), device=dev1, dtype=torch.bfloat16)
        val = torch.randn_like(res)
        gt = torch.randn((1, 1, 512), device=dev1, dtype=torch.bfloat16)
        po, pn = PytorchPostSAFunction()(res, val, None, 1e-6, gt)
        assert po.device == dev1 and pn.device == dev1 and torch.cuda.current_device() == 0
