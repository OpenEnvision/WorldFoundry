# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for Wave 9H core layers, schedulers, and timestep Hopper kernels.

Covers:
- CPU-only execution and lazy Triton imports when Triton is absent.
- Lossless GPU parity vs PyTorch fallback for:
  1. ``LayerNorm2d`` (NCHW & ``channels_last`` NHWC, BF16/FP16/FP32, mixed dtype,
     chunked wide ``C > 4096``, large DC offsets, stride preservation, PSNR >= 60 dB).
  2. ``QuickGELU`` & ``Mish`` (contiguous, ``channels_last``, ``channels_last_3d``,
     BF16/FP16/FP32, <= 1 ULP / bit-exact rounding, stride preservation).
  3. ``SwiGLUFFN`` & ``SwiGLUFFNFused`` (fused ``silu_and_mul`` CUDA inference path).
  4. ``SchedulerInterface.convert_x0_to_noise`` / ``convert_noise_to_x0`` /
     ``convert_velocity_to_x0`` & ``triton_scheduler_linear_comb_fp64`` (bit-exact
     FP64 across BF16/FP16/FP32/FP64, 4D & 5D Wan latents, ``MethodType`` binding,
     zero FP64 HBM temporary allocations).
  5. ``FlowMatchScheduler.step`` / ``return_to_timestep`` / ``add_noise`` (bit-exact
     across BF16/FP16/FP32, scalar vs batched dtype promotion, ``to_final``,
     ``inverse_timesteps`` / ``reverse_sigmas``, tie-breaking, ``channels_last``).
  6. ``get_timestep_embedding`` & module wrappers (``Timesteps``,
     ``ProjectedTimestepEmbedding``, ``PixArtAlphaCombinedTimestepSizeEmbeddings``;
     even & odd dims, ``flip_sin_to_cos``, ``downscale_freq_shift``, ``scale``,
     strided inputs, integer & floating dtypes, ``max_diff < 1e-5``).
- Hopper SM90 zero register spills (``n_spills == 0``) across every compiled variant.
- Strict ``tl.int64`` indexing (> 2^31 element offset when >= 16 GiB free VRAM).
- ``torch.compile(backend="inductor", fullgraph=True)`` +
  ``register_worldfoundry_inductor_fallbacks()`` preserving ``channels_last``.
- CUDA Graph capture & replay with mutated GPU inputs (zero CPU-GPU syncs).
- Autograd fallback, backward hook fallback, and env backend toggles.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.core.nn.blocks.layers import (
    LayerNorm2d,
    Mish,
    QuickGELU,
    SwiGLUFFN,
    SwiGLUFFNFused,
)
from worldfoundry.core.nn.diffusion.schedulers import FlowMatchScheduler, SchedulerInterface
from worldfoundry.core.nn.diffusion.timestep import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    ProjectedTimestepEmbedding,
    Timesteps,
    get_timestep_embedding,
)
from worldfoundry.core.nn.triton_core_layers import (
    _get_triton_suite,
    core_layers_triton_ops_enabled,
    triton_scheduler_linear_comb_fp64,
)

_CUDA = torch.cuda.is_available()
_HOPPER_PLUS = _CUDA and torch.cuda.get_device_capability()[0] >= 9


class _DummyScheduler(SchedulerInterface):
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return clean_latent + noise


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


def _psnr_db(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = actual.float() - expected.float()
    mse = float((diff * diff).mean().item())
    if mse == 0.0:
        return float("inf")
    peak = float(expected.float().abs().max().item())
    if peak == 0.0:
        return float("inf")
    return 10.0 * math.log10((peak * peak) / mse)


def _assert_1ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if actual.numel() == 0:
        return
    if actual.dtype == torch.bfloat16:
        a_i = actual.view(torch.int16).to(torch.int32)
        b_i = expected.view(torch.int16).to(torch.int32)
        a_ord = torch.where(a_i < 0, 0x8000 - a_i, a_i)
        b_ord = torch.where(b_i < 0, 0x8000 - b_i, b_i)
        max_ulp = int((a_ord - b_ord).abs().max().item())
        assert max_ulp <= 1, f"BF16 ULP diff {max_ulp} > 1"
    elif actual.dtype == torch.float16:
        a_i = actual.view(torch.int16).to(torch.int32)
        b_i = expected.view(torch.int16).to(torch.int32)
        a_ord = torch.where(a_i < 0, 0x8000 - a_i, a_i)
        b_ord = torch.where(b_i < 0, 0x8000 - b_i, b_i)
        max_ulp = int((a_ord - b_ord).abs().max().item())
        assert max_ulp <= 1, f"FP16 ULP diff {max_ulp} > 1"
    else:
        a_i = actual.view(torch.int32).to(torch.int64)
        b_i = expected.view(torch.int32).to(torch.int64)
        a_ord = torch.where(a_i < 0, 0x80000000 - a_i, a_i)
        b_ord = torch.where(b_i < 0, 0x80000000 - b_i, b_i)
        max_ulp = int((a_ord - b_ord).abs().max().item())
        assert max_ulp <= 1, f"FP32 ULP diff {max_ulp} > 1"


# ──────────────────────────────────────────────────────────────────────────
# 1. CPU-only lazy Triton import subprocess test
# ──────────────────────────────────────────────────────────────────────────


def test_cpu_only_absent_triton_subprocess() -> None:
    """All core layer, scheduler, and timestep ops must run on CPU with Triton blocked."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "import torch\n"
        "from worldfoundry.core.nn.blocks.layers import LayerNorm2d, QuickGELU, Mish, SwiGLUFFN, SwiGLUFFNFused\n"
        "from worldfoundry.core.nn.diffusion.schedulers import SchedulerInterface, FlowMatchScheduler\n"
        "from worldfoundry.core.nn.diffusion.timestep import (\n"
        "    get_timestep_embedding, Timesteps, ProjectedTimestepEmbedding,\n"
        "    PixArtAlphaCombinedTimestepSizeEmbeddings,\n"
        ")\n"
        "import worldfoundry.core.nn.triton_core_layers as tcl\n"
        "x = torch.randn(2, 16, 4, 4)\n"
        "assert LayerNorm2d(16)(x).shape == x.shape\n"
        "assert QuickGELU()(x).shape == x.shape\n"
        "assert Mish()(x).shape == x.shape\n"
        "assert SwiGLUFFN(16, 32)(torch.randn(2, 4, 16)).shape == (2, 4, 16)\n"
        "assert SwiGLUFFNFused(16, 32)(torch.randn(2, 4, 16)).shape == (2, 4, 16)\n"
        "class S(SchedulerInterface):\n"
        "    def add_noise(self, c, n, t): return c + n\n"
        "s = S(); s.alphas_cumprod = torch.linspace(0.9, 0.1, 10)\n"
        "t = torch.tensor([1, 5])\n"
        "assert s.convert_x0_to_noise(x, x, t).shape == x.shape\n"
        "assert s.convert_noise_to_x0(x, x, t).shape == x.shape\n"
        "assert s.convert_velocity_to_x0(x, x, t).shape == x.shape\n"
        "fms = FlowMatchScheduler(num_inference_steps=10)\n"
        "assert fms.step(x, fms.timesteps[2], x).shape == x.shape\n"
        "assert fms.return_to_timestep(fms.timesteps[2], x, x).shape == x.shape\n"
        "assert fms.add_noise(x, x, fms.timesteps[2]).shape == x.shape\n"
        "ts = torch.tensor([0.0, 500.0, 999.0])\n"
        "assert get_timestep_embedding(ts, 65, flip_sin_to_cos=True).shape == (3, 65)\n"
        "assert Timesteps(32, True, 0)(ts).shape == (3, 32)\n"
        "assert ProjectedTimestepEmbedding(32, 64)(ts, torch.float32).shape == (3, 64)\n"
        "assert PixArtAlphaCombinedTimestepSizeEmbeddings(64, 32)(ts, torch.float32).shape == (3, 64)\n"
        "assert tcl.triton_layer_norm_2d(x, torch.ones(16), torch.zeros(16), 1e-6) is None\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"CPU-only absent-Triton test failed:\n{proc.stdout}\n{proc.stderr}"


# ──────────────────────────────────────────────────────────────────────────
# 2. Target 1: LayerNorm2d, QuickGELU, Mish, SwiGLUFFN GPU parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 8),
        (2, 17, 13, 19),
        (1, 256, 32, 32),
        (2, 768, 16, 16),
        (1, 1280, 16, 16),
        (1, 5120, 4, 4),  # wide C > 4096 chunked NHWC path
    ],
)
def test_gpu_layer_norm_2d_parity(dtype: torch.dtype, layout: str, shape: tuple[int, int, int, int]) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    x = torch.randn(shape, device=device, dtype=dtype)
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
    ln = LayerNorm2d(shape[1], eps=1e-6).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.5, 0.5)

    with torch.inference_mode():
        got = ln(x)
        ref = _run_torch_backend(lambda: ln(x))

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.stride() == x.stride(), f"stride changed: {got.stride()} vs {x.stride()}"
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    elif dtype == torch.float16:
        assert _psnr_db(got, ref) >= 65.0
        torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    else:
        assert _psnr_db(got, ref) >= 60.0


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_layer_norm_2d_edge_cases() -> None:
    device = torch.device("cuda")
    # Mixed dtype: BF16 input with FP32 weight/bias -> FP32 output (preserving channels_last)
    x_mix = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    ln_fp32 = LayerNorm2d(64).to(device=device, dtype=torch.float32).eval()
    with torch.inference_mode():
        got_mix = ln_fp32(x_mix)
        ref_mix = _run_torch_backend(lambda: ln_fp32(x_mix))
    assert got_mix.dtype == ref_mix.dtype == torch.float32
    assert got_mix.is_contiguous(memory_format=torch.channels_last)
    assert _psnr_db(got_mix, ref_mix) >= 60.0

    # Large DC offset stability in FP32 NCHW (shifted variance avoids catastrophic cancellation;
    # 1 FP32 ULP at 1e4 is ~1e-3 so allow 2e-3 for mean-rounding differences).
    x_dc = 1e4 + torch.randn(1, 256, 8, 8, device=device, dtype=torch.float32)
    ln_dc = LayerNorm2d(256).to(device=device, dtype=torch.float32).eval()
    with torch.inference_mode():
        got_dc = ln_dc(x_dc)
        ref_dc = _run_torch_backend(lambda: ln_dc(x_dc))
    torch.testing.assert_close(got_dc, ref_dc, rtol=2e-3, atol=2e-3)

    # Empty batch
    empty = torch.empty(0, 64, 8, 8, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        out_empty = LayerNorm2d(64).to(device=device, dtype=torch.bfloat16).eval()(empty)
    assert out_empty.shape == empty.shape and out_empty.dtype == empty.dtype


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["contig", "channels_last", "channels_last_3d"])
def test_gpu_quick_gelu_and_mish_parity(dtype: torch.dtype, layout: str) -> None:
    torch.manual_seed(22)
    device = torch.device("cuda")
    if layout == "contig":
        x = torch.randn(2, 128, 32, 32, device=device, dtype=dtype) * 5.0
    elif layout == "channels_last":
        x = (torch.randn(2, 64, 16, 16, device=device, dtype=dtype) * 5.0).to(memory_format=torch.channels_last)
    else:
        x = (torch.randn(1, 32, 4, 8, 8, device=device, dtype=dtype) * 5.0).to(memory_format=torch.channels_last_3d)

    qg = QuickGELU().eval()
    mish = Mish().eval()
    with torch.inference_mode():
        gq, rq = qg(x), _run_torch_backend(lambda: qg(x))
        gm, rm = mish(x), _run_torch_backend(lambda: mish(x))

    assert gq.stride() == x.stride() and gm.stride() == x.stride()
    assert qg.state_dict() == {} and mish.state_dict() == {}
    # Mish is bit-exact across all three dtypes; QuickGELU is bit-exact in BF16 and <= 1 ULP in FP16/FP32.
    assert torch.equal(gm, rm), f"Mish mismatch max={(gm.float() - rm.float()).abs().max().item()}"
    if dtype == torch.bfloat16:
        assert torch.equal(gq, rq)
    else:
        _assert_1ulp(gq, rq)

    # Extreme values including softplus threshold=20 boundary (allow subnormal FTZ < 1e-38 at x=-100)
    ext = torch.tensor([-100.0, -25.0, -20.0, -1.0, 0.0, 1.0, 19.9, 20.0, 20.1, 50.0], device=device, dtype=dtype)
    with torch.inference_mode():
        torch.testing.assert_close(mish(ext), _run_torch_backend(lambda: mish(ext)), rtol=0.0, atol=1e-38)
        _assert_1ulp(qg(ext), _run_torch_backend(lambda: qg(ext)))


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("cls", [SwiGLUFFN, SwiGLUFFNFused])
def test_gpu_swiglu_ffn_parity(dtype: torch.dtype, cls: type[SwiGLUFFN]) -> None:
    torch.manual_seed(33)
    device = torch.device("cuda")
    mod = cls(256, 512, 256).to(device=device, dtype=dtype).eval()
    x = torch.randn(2, 64, 256, device=device, dtype=dtype)
    with torch.inference_mode():
        got = mod(x)
        ref = _run_torch_backend(lambda: mod(x))
    assert got.shape == ref.shape and got.dtype == ref.dtype
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


# ──────────────────────────────────────────────────────────────────────────
# 3. Target 2: SchedulerInterface FP64 & FlowMatchScheduler GPU parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape,t_vals,layout",
    [
        ((2, 8, 16, 16), [10, 500], "contig"),
        ((2, 8, 16, 16), [10, 500], "channels_last"),
        ((1, 16, 5, 16, 16), [250], "contig"),
        ((1, 4, 16, 12, 12), [10, 100, 300, 999], "contig"),  # per-frame K == shape[-4]
        ((1, 16, 4, 8, 8), [-1], "contig"),  # negative timestep index wrap
    ],
)
def test_gpu_scheduler_interface_fp64_bit_exact(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    t_vals: list[int],
    layout: str,
) -> None:
    torch.manual_seed(44)
    device = torch.device("cuda")
    sched = _DummyScheduler()
    sched.alphas_cumprod = torch.linspace(0.999, 0.001, 1000, device=device)

    a = torch.randn(shape, device=device, dtype=dtype)
    xt = torch.randn(shape, device=device, dtype=dtype)
    if layout == "channels_last":
        a = a.to(memory_format=torch.channels_last)
        xt = xt.to(memory_format=torch.channels_last)
    t = torch.tensor(t_vals, device=device)

    with torch.inference_mode():
        g1 = sched.convert_x0_to_noise(a, xt, t)
        r1 = _run_torch_backend(lambda: sched.convert_x0_to_noise(a, xt, t))
        g2 = sched.convert_noise_to_x0(a, xt, t)
        r2 = _run_torch_backend(lambda: sched.convert_noise_to_x0(a, xt, t))
        g3 = sched.convert_velocity_to_x0(a, xt, t)
        r3 = _run_torch_backend(lambda: sched.convert_velocity_to_x0(a, xt, t))

    assert torch.equal(g1, r1), f"convert_x0_to_noise diff={(g1.double() - r1.double()).abs().max().item()}"
    assert torch.equal(g2, r2), f"convert_noise_to_x0 diff={(g2.double() - r2.double()).abs().max().item()}"
    assert torch.equal(g3, r3), f"convert_velocity_to_x0 diff={(g3.double() - r3.double()).abs().max().item()}"
    assert g1.stride() == a.stride() and g2.stride() == a.stride() and g3.stride() == a.stride()


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_scheduler_interface_methodtype_and_zero_fp64_vram() -> None:
    """Verify MethodType binding (wan_wrapper pattern), CPU alphas_cumprod, and zero FP64 HBM alloc."""
    device = torch.device("cuda")
    torch.manual_seed(45)

    # Bound via types.MethodType onto a plain object owning only alphas_cumprod (no SchedulerInterface subclass!)
    host = types.SimpleNamespace(alphas_cumprod=torch.linspace(0.99, 0.01, 1000))  # CPU alphas_cumprod
    host.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, host)
    host.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, host)

    shape = (1, 16, 21, 90, 160)  # 4.84M BF16 elements (9.68 MB); 3x FP64 would be 116 MB!
    x0 = torch.randn(shape, device=device, dtype=torch.bfloat16)
    xt = torch.randn(shape, device=device, dtype=torch.bfloat16)
    t_cpu = torch.tensor([500])

    with torch.inference_mode():
        got = host.convert_x0_to_noise(x0, xt, t_cpu)
        ref = _run_torch_backend(lambda: host.convert_x0_to_noise(x0, xt, t_cpu))
    assert torch.equal(got, ref)

    # Measure peak additional VRAM during conversion on GPU alphas_cumprod
    host.alphas_cumprod = host.alphas_cumprod.to(device)
    t_gpu = t_cpu.to(device)
    with torch.inference_mode():
        _ = host.convert_x0_to_noise(x0, xt, t_gpu)  # warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        base_alloc = torch.cuda.memory_allocated(device)
        out = host.convert_x0_to_noise(x0, xt, t_gpu)
        peak_delta = torch.cuda.max_memory_allocated(device) - base_alloc
    expected_out_bytes = x0.numel() * x0.element_size()
    # Must allocate ONLY the BF16 output (~9.7 MB), NOT 3x FP64 temporaries (~116 MB)!
    assert peak_delta <= expected_out_bytes * 1.5, f"peak_delta={peak_delta} vs out={expected_out_bytes}"
    del out

    # Also test direct triton_scheduler_linear_comb_fp64 (mode=3)
    c1 = torch.tensor([0.75], device=device, dtype=torch.float64)
    c2 = torch.tensor([-0.66], device=device, dtype=torch.float64)
    with torch.inference_mode():
        got_lc = triton_scheduler_linear_comb_fp64(xt, x0, c1, c2, out_dtype=torch.bfloat16)
        assert got_lc is not None
        ref_lc = (c1 * xt.double() + c2 * x0.double()).to(torch.bfloat16)
    assert torch.equal(got_lc, ref_lc)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("to_final", [False, True])
@pytest.mark.parametrize("reverse_sigmas", [False, True])
@pytest.mark.parametrize("t_mode", ["gpu_scalar", "cpu_scalar", "py_float", "gpu_batch", "last_step", "midpoint_tie"])
def test_gpu_flow_match_scheduler_bit_exact(
    dtype: torch.dtype,
    to_final: bool,
    reverse_sigmas: bool,
    t_mode: str,
) -> None:
    torch.manual_seed(55)
    device = torch.device("cuda")
    sched = FlowMatchScheduler(
        num_inference_steps=20,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        reverse_sigmas=reverse_sigmas,
    )
    sample = torch.randn(2, 8, 4, 8, 8, device=device, dtype=dtype)
    vel = torch.randn(2, 8, 4, 8, 8, device=device, dtype=dtype)

    if t_mode == "gpu_scalar":
        t: Any = sched.timesteps[5].to(device)
    elif t_mode == "cpu_scalar":
        t = sched.timesteps[5].clone()
    elif t_mode == "py_float":
        t = float(sched.timesteps[5].item())
    elif t_mode == "gpu_batch":
        t = torch.stack([sched.timesteps[3], sched.timesteps[12]]).to(device)
    elif t_mode == "last_step":
        t = sched.timesteps[-1].to(device)
    else:
        # Exact midpoint tie between index 4 and index 5 must tie-break to lower index (4)
        t = ((sched.timesteps[4] + sched.timesteps[5]) * 0.5).to(device)

    with torch.inference_mode():
        gs = sched.step(vel, t, sample, to_final=to_final)
        rs = _run_torch_backend(lambda: sched.step(vel, t, sample, to_final=to_final))
        ga = sched.add_noise(sample, vel, t)
        ra = _run_torch_backend(lambda: sched.add_noise(sample, vel, t))
        # Avoid division by exact 0.0 sigma for return_to_timestep
        t_nz = sched.timesteps[5].to(device) if t_mode in ("last_step",) else t
        gr = sched.return_to_timestep(t_nz, sample, vel)
        rr = _run_torch_backend(lambda: sched.return_to_timestep(t_nz, sample, vel))

    assert gs.dtype == rs.dtype and gr.dtype == rr.dtype and ga.dtype == ra.dtype
    assert torch.equal(gs, rs), f"step mismatch mode={t_mode} max={(gs.float() - rs.float()).abs().max().item()}"
    assert torch.equal(gr, rr), f"ret mismatch mode={t_mode} max={(gr.float() - rr.float()).abs().max().item()}"
    assert torch.equal(ga, ra), f"add_noise mismatch mode={t_mode} max={(ga.float() - ra.float()).abs().max().item()}"


# ──────────────────────────────────────────────────────────────────────────
# 4. Target 3: get_timestep_embedding & module wrappers GPU parity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("n", [0, 1, 2, 17, 64, 512])
@pytest.mark.parametrize("dim", [0, 1, 15, 64, 255, 256, 1024, 3072])
@pytest.mark.parametrize("flip_sin_to_cos", [False, True])
@pytest.mark.parametrize("downscale_freq_shift", [0.0, 1.0])
def test_gpu_get_timestep_embedding_parity(
    n: int,
    dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
) -> None:
    device = torch.device("cuda")
    ts = torch.linspace(0.0, 1000.0, n, device=device, dtype=torch.float32) if n > 0 else torch.empty(0, device=device)
    with torch.inference_mode():
        got = get_timestep_embedding(
            ts,
            dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=1.0,
            max_period=10000,
        )
        ref = _run_torch_backend(
            lambda: get_timestep_embedding(
                ts,
                dim,
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=downscale_freq_shift,
                scale=1.0,
                max_period=10000,
            )
        )
    assert got.shape == ref.shape and got.dtype == torch.float32
    if got.numel() > 0:
        max_diff = float((got - ref).abs().max().item())
        assert max_diff < 1e-5, f"max_diff={max_diff:.3e} for n={n}, dim={dim}"
        if dim % 2 == 1:
            assert bool((got[:, -1] == 0.0).all())


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("in_dtype", [torch.int64, torch.int32, torch.bfloat16, torch.float16, torch.float64])
def test_gpu_get_timestep_embedding_dtypes_strided_and_modules(in_dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    base = torch.arange(0, 1000, 50, device=device).to(in_dtype)
    strided = base[::2]  # non-contiguous stride=2
    with torch.inference_mode():
        got = get_timestep_embedding(strided, 256, flip_sin_to_cos=True, downscale_freq_shift=0.5, scale=10.0)
        ref = _run_torch_backend(
            lambda: get_timestep_embedding(strided, 256, flip_sin_to_cos=True, downscale_freq_shift=0.5, scale=10.0)
        )
    assert float((got - ref).abs().max().item()) < 1e-5

    # Module wrappers
    ts_f = torch.tensor([10.0, 250.5, 999.0], device=device, dtype=torch.float32)
    ts_mod = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
    proj = ProjectedTimestepEmbedding(256, 128).to(device=device, dtype=torch.bfloat16).eval()
    pix = PixArtAlphaCombinedTimestepSizeEmbeddings(128, 64).to(device=device, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        assert float((ts_mod(ts_f) - _run_torch_backend(lambda: ts_mod(ts_f))).abs().max().item()) < 1e-5
        torch.testing.assert_close(
            proj(ts_f, torch.bfloat16), _run_torch_backend(lambda: proj(ts_f, torch.bfloat16)), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            pix(ts_f, torch.bfloat16), _run_torch_backend(lambda: pix(ts_f, torch.bfloat16)), rtol=1e-2, atol=1e-2
        )


# ──────────────────────────────────────────────────────────────────────────
# 5. Hopper SM90 zero register spills (n_spills == 0) across all 8 kernels
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _HOPPER_PLUS, reason="requires Hopper SM90+ GPU")
def test_hopper_all_core_layer_kernels_zero_spills() -> None:
    device = torch.device("cuda")
    torch.manual_seed(66)
    with torch.inference_mode():
        # 1. LayerNorm2d NCHW + NHWC + NHWC chunked (C > 4096)
        x_nchw = torch.randn(2, 768, 16, 16, device=device, dtype=torch.bfloat16)
        x_nhwc = x_nchw.to(memory_format=torch.channels_last)
        ln = LayerNorm2d(768).to(device=device, dtype=torch.bfloat16).eval()
        ln(x_nchw)
        ln(x_nhwc)
        x_wide = torch.randn(1, 5120, 4, 4, device=device, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        LayerNorm2d(5120).to(device=device, dtype=torch.bfloat16).eval()(x_wide)

        # 2. QuickGELU & Mish (BF16 + FP32)
        QuickGELU()(x_nchw)
        QuickGELU()(x_nchw.float())
        Mish()(x_nchw)
        Mish()(x_nchw.float())

        # 3. SchedulerInterface FP64 (modes 0, 1, 2, 3)
        sched = _DummyScheduler()
        sched.alphas_cumprod = torch.linspace(0.99, 0.01, 1000, device=device)
        t_s = torch.tensor([500], device=device)
        t_b = torch.tensor([100, 500], device=device)
        sched.convert_x0_to_noise(x_nchw, x_nchw, t_s)
        sched.convert_noise_to_x0(x_nchw, x_nchw, t_b)
        sched.convert_velocity_to_x0(x_nchw, x_nchw, t_s)
        triton_scheduler_linear_comb_fp64(
            x_nchw,
            x_nchw,
            torch.tensor([0.7], device=device, dtype=torch.float64),
            torch.tensor([-0.3], device=device, dtype=torch.float64),
        )

        # 4. FlowMatchScheduler (step, return_to_timestep, add_noise)
        fms = FlowMatchScheduler(num_inference_steps=50, shift=5.0, sigma_min=0.0, extra_one_step=True)
        fms.step(x_nchw, fms.timesteps[10].to(device), x_nchw)
        fms.return_to_timestep(fms.timesteps[10].to(device), x_nchw, x_nchw)
        fms.add_noise(x_nchw, x_nchw, fms.timesteps[10].to(device))

        # 5. Sinusoidal timestep embedding (even + odd)
        ts = torch.linspace(0.0, 1000.0, 64, device=device, dtype=torch.float32)
        get_timestep_embedding(ts, 1024, flip_sin_to_cos=True, downscale_freq_shift=0)
        get_timestep_embedding(ts[:1], 255, flip_sin_to_cos=False, downscale_freq_shift=1)

    suite = _get_triton_suite()
    kernel_names = (
        "layer_norm_2d_nhwc_kernel",
        "layer_norm_2d_nhwc_chunked_kernel",
        "layer_norm_2d_nchw_kernel",
        "quick_gelu_fwd_kernel",
        "mish_fwd_kernel",
        "scheduler_linear_comb_fp64_kernel",
        "flow_match_euler_step_kernel",
        "sinusoidal_timestep_embedding_kernel",
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
# 6. Strict tl.int64 indexing (> 2^31 element count)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_strict_int64_indexing_above_2_pow_31() -> None:
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 16 * (1024**3):
        # Requires >=16 GiB free VRAM for >2^31 element test.
        pytest.skip()
    device = torch.device("cuda")
    # 2^31 + 2048 BF16 elements (~4.3 GiB); elements past 2^31 overflow signed int32!
    total = (1 << 31) + 2048
    big_x = torch.empty(total, device=device, dtype=torch.bfloat16)
    big_x[:2048].fill_(1.5)
    big_x[-2048:].fill_(-2.0)
    tail_slice = big_x[-2048:].clone()

    with torch.inference_mode():
        out_q = QuickGELU()(big_x)
        tail_ref_q = _run_torch_backend(lambda: QuickGELU()(tail_slice))
        assert torch.equal(out_q[-2048:], tail_ref_q)
        del out_q

        out_m = Mish()(big_x)
        tail_ref_m = _run_torch_backend(lambda: Mish()(tail_slice))
        assert torch.equal(out_m[-2048:], tail_ref_m)
        del out_m

    del big_x
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# 7. torch.compile(fullgraph=True) + register_worldfoundry_inductor_fallbacks()
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_torch_compile_fullgraph_and_inductor_fallbacks() -> None:
    from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

    register_worldfoundry_inductor_fallbacks()
    device = torch.device("cuda")
    torch.manual_seed(77)

    # Confirm all 6 custom ops are registered in torch.ops.worldfoundry
    ns = torch.ops.worldfoundry
    for op_name in (
        "layer_norm_2d_fwd",
        "quick_gelu_fwd",
        "mish_fwd",
        "scheduler_linear_comb_fp64",
        "flow_match_euler_step",
        "sinusoidal_timestep_embedding_fwd",
    ):
        assert hasattr(ns, op_name), f"missing worldfoundry::{op_name}"

    ln = LayerNorm2d(64).to(device=device, dtype=torch.bfloat16).eval()
    qg = QuickGELU().eval()
    mish = Mish().eval()
    swiglu = SwiGLUFFN(64, 128, 64).to(device=device, dtype=torch.bfloat16).eval()
    sched = _DummyScheduler()
    sched.alphas_cumprod = torch.linspace(0.99, 0.01, 1000, device=device)
    fms = FlowMatchScheduler(num_inference_steps=20, shift=5.0, sigma_min=0.0, extra_one_step=True)
    fms.sigmas = fms.sigmas.to(device)
    fms.timesteps = fms.timesteps.to(device)

    x_nchw = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)
    x_tok = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
    xt = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16)
    t_idx = torch.tensor([500], device=device)
    t_fm = fms.timesteps[5]
    ts = torch.tensor([10.0, 500.0], device=device, dtype=torch.float32)

    def combined(
        a_nchw: torch.Tensor,
        a_nhwc: torch.Tensor,
        a_tok: torch.Tensor,
        b_xt: torch.Tensor,
        ti: torch.Tensor,
        tf: torch.Tensor,
        t_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        o_nchw = mish(qg(ln(a_nchw)))
        o_nhwc = ln(a_nhwc)
        o_swi = swiglu(a_tok)
        o_conv = sched.convert_x0_to_noise(a_nchw, b_xt, ti)
        o_step = fms.step(b_xt, tf, a_nchw)
        o_ret = fms.return_to_timestep(tf, a_nchw, b_xt)
        o_emb = get_timestep_embedding(t_vec, 256, flip_sin_to_cos=True, downscale_freq_shift=0)
        return o_nchw, o_nhwc, o_swi, o_conv, o_step, o_ret, o_emb

    compiled = torch.compile(combined, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        ref_outs = combined(x_nchw, x_nhwc, x_tok, xt, t_idx, t_fm, ts)
        got_outs = compiled(x_nchw, x_nhwc, x_tok, xt, t_idx, t_fm, ts)

    # channels_last must be preserved through compiled graph!
    assert got_outs[1].is_contiguous(memory_format=torch.channels_last)
    for idx, (g, r) in enumerate(zip(got_outs, ref_outs, strict=True)):
        if idx == 2:  # SwiGLUFFN linear + inductor silu_and_mul
            torch.testing.assert_close(g, r, rtol=2e-2, atol=2e-2)
        else:
            assert torch.equal(g, r), f"compiled output {idx} mismatch"


# ──────────────────────────────────────────────────────────────────────────
# 8. CUDA Graph capture & replay (zero CPU-GPU syncs)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_cuda_graph_capture_and_replay() -> None:
    device = torch.device("cuda")
    torch.manual_seed(88)

    ln = LayerNorm2d(64).to(device=device, dtype=torch.bfloat16).eval()
    qg = QuickGELU().eval()
    mish = Mish().eval()
    sched = _DummyScheduler()
    sched.alphas_cumprod = torch.linspace(0.99, 0.01, 1000, device=device)
    fms = FlowMatchScheduler(num_inference_steps=20, shift=5.0, sigma_min=0.0, extra_one_step=True)
    fms.sigmas = fms.sigmas.to(device)
    fms.timesteps = fms.timesteps.to(device)

    x_nchw = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)
    xt = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16)
    t_idx = torch.tensor([300], device=device)
    t_fm = fms.timesteps[4].clone()
    ts = torch.tensor([10.0, 500.0], device=device, dtype=torch.float32)

    def run_all() -> tuple[torch.Tensor, ...]:
        return (
            ln(x_nchw),
            ln(x_nhwc),
            qg(x_nchw),
            mish(x_nhwc),
            sched.convert_x0_to_noise(x_nchw, xt, t_idx),
            fms.step(xt, t_fm, x_nchw),
            fms.return_to_timestep(t_fm, x_nchw, xt),
            get_timestep_embedding(ts, 256, flip_sin_to_cos=True, downscale_freq_shift=0),
        )

    with torch.inference_mode():
        _ = run_all()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_outs = run_all()

        # Mutate static GPU inputs including GPU timesteps (verifies zero host syncs!)
        x_nchw.normal_()
        x_nhwc.copy_(x_nchw)
        xt.normal_()
        t_idx.fill_(700)
        t_fm.copy_(fms.timesteps[12])
        ts.copy_(torch.tensor([250.0, 888.0], device=device))

        graph.replay()
        torch.cuda.synchronize()
        e_outs = run_all()

    for g, e in zip(g_outs, e_outs, strict=True):
        assert torch.equal(g, e)


# ──────────────────────────────────────────────────────────────────────────
# 9. Autograd fallback, backward hook fallback, & env toggle
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_gpu_autograd_hooks_and_env_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    torch.manual_seed(99)

    # Env toggle check
    monkeypatch.setenv("WORLDFOUNDRY_CORE_LAYERS_TRITON_OPS", "0")
    assert core_layers_triton_ops_enabled() is False
    monkeypatch.setenv("WORLDFOUNDRY_CORE_LAYERS_TRITON_OPS", "1")
    assert core_layers_triton_ops_enabled() is True

    # Autograd fallback on all modules & functions
    x = torch.randn(2, 32, 8, 8, device=device, dtype=torch.float32, requires_grad=True)
    ln = LayerNorm2d(32).to(device=device)
    Mish()(QuickGELU()(ln(x))).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert ln.weight.grad is not None and ln.bias.grad is not None

    xtok = torch.randn(2, 8, 32, device=device, dtype=torch.float32, requires_grad=True)
    SwiGLUFFN(32, 64, 32).to(device=device)(xtok).sum().backward()
    assert xtok.grad is not None and torch.isfinite(xtok.grad).all()

    sched = _DummyScheduler()
    sched.alphas_cumprod = torch.linspace(0.99, 0.01, 1000, device=device)
    x0_g = torch.randn(2, 4, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
    xt_g = torch.randn(2, 4, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
    sched.convert_x0_to_noise(x0_g, xt_g, torch.tensor([100, 200], device=device)).sum().backward()
    assert x0_g.grad is not None and xt_g.grad is not None

    fms = FlowMatchScheduler(num_inference_steps=20)
    s_g = torch.randn(2, 4, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
    v_g = torch.randn(2, 4, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
    fms.step(v_g, fms.timesteps[5].to(device), s_g).sum().backward()
    assert s_g.grad is not None and v_g.grad is not None

    ts_g = torch.tensor([10.0, 200.0], device=device, dtype=torch.float32, requires_grad=True)
    get_timestep_embedding(ts_g, 64).sum().backward()
    assert ts_g.grad is not None and torch.isfinite(ts_g.grad).all()

    # Module backward hook check (ensures hooks still fire cleanly)
    hook_fired = {"ln": False, "qg": False, "mish": False}
    ln2 = LayerNorm2d(16).to(device=device)
    qg2 = QuickGELU()
    mish2 = Mish()
    ln2.register_full_backward_hook(lambda _m, _gi, _go: hook_fired.__setitem__("ln", True))
    qg2.register_full_backward_hook(lambda _m, _gi, _go: hook_fired.__setitem__("qg", True))
    mish2.register_full_backward_hook(lambda _m, _gi, _go: hook_fired.__setitem__("mish", True))
    xh = torch.randn(1, 16, 4, 4, device=device, dtype=torch.float32, requires_grad=True)
    mish2(qg2(ln2(xh))).sum().backward()
    assert all(hook_fired.values())
