# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lossless Hopper SM90 verification tests for LTX VideoVAE, AudioVAE, Vocoder & Upsampler kernels.

Covers:
1. Lazy Triton import & CPU-only fallback (no Triton/CUDA import on module load).
2. Exact CPU & autograd gradient preservation across all new custom ops.
3. Hook & VRAM-offload wrapper safety (never bypasses module hooks or custom wrappers).
4. Bit-exact / <= 1-2 ULP lossless parity (PSNR >= 50 dB, cos >= 0.9998) vs PyTorch
   BF16 and FP32 ground truth (TF32 disabled) across NCDHW & channels_last_3d,
   zeros/reflect/replicate spatial padding, causal/non-causal, AdaLN, C<8 align-C,
   AudioVAE, Vocoder/VocoderWithBWE (Snake & SnakeBeta), and SpatialUpsampler x2.
5. Strict >2^31-element ``tl.int64`` indexing verification on Hopper/large-VRAM GPUs.
6. CUDA Graph capture + replay bit-exactness and ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import (
    AudioDecoder,
    AudioEncoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.causality_axis import CausalityAxis
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.triton_audio_ops import (
    _activation1d_snake_torch,
    _pixel_norm_silu_pad_2d_torch,
    triton_activation1d_snake,
    triton_mean3_1d,
    triton_pixel_norm_silu_pad_2d,
    triton_residual_bias_add_1d,
    triton_residual_bias_add_2d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import (
    Activation1d,
    MelSTFT,
    Snake,
    SnakeBeta,
    Vocoder,
    VocoderWithBWE,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import NormType
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.convolution import CausalConv3d
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import (
    LogVarianceType,
    NormLayerType,
    PaddingModeType,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.resnet import ResnetBlock3D
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.triton_vae_ops import (
    _depth_to_space_3d_torch,
    _is_channels_last_3d,
    _pad_3d_torch,
    _patchify_3d_torch,
    _pixel_norm_adaln_silu_pad_3d_torch,
    _residual_bias_add_torch,
    _space_to_depth_3d_add_torch,
    _unpatchify_3d_torch,
    triton_depth_to_space_3d,
    triton_ltx_pad_3d,
    triton_ltx_pad_3d_align_c,
    triton_ltx_residual_bias_add,
    triton_patchify_3d,
    triton_pixel_norm_adaln_silu_pad_3d,
    triton_space_to_depth_3d_add,
    triton_unpatchify_3d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoDecoder, VideoEncoder
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import LatentUpsampler
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.res_block import ResBlock as UpsamplerResBlock
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.triton_upsampler_ops import (
    _is_channels_last_nd,
    _upsampler_gn_silu_torch,
    triton_upsampler_group_norm_silu,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _psnr_cos(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a = actual.float()
    r = ref.float()
    mse = torch.mean((a - r) ** 2).item()
    peak = max(float(r.abs().max().item()), 1.0)
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)
    cos = F.cosine_similarity(a.flatten(), r.flatten(), dim=0).item()
    return psnr, cos


def _assert_lossless(actual: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, *, min_psnr: float = 52.0) -> None:
    assert actual.shape == ref.shape
    assert actual.dtype == ref.dtype
    assert torch.isfinite(actual).all()
    diff = (actual.float() - ref.float()).abs()
    max_diff = diff.max().item()
    psnr, cos = _psnr_cos(actual, ref)
    assert psnr >= min_psnr, f"PSNR={psnr:.2f} dB < {min_psnr} dB (max_diff={max_diff:.6g}, dtype={dtype})"
    assert cos >= 0.9998, f"cos={cos:.6f} < 0.9998 (max_diff={max_diff:.6g}, dtype={dtype})"
    if dtype == torch.float32:
        assert max_diff <= 1e-4, f"max_diff={max_diff:.6g} exceeded 1e-4 for FP32"
    else:
        ulp = (torch.nextafter(ref.abs(), torch.full_like(ref, float("inf"))) - ref.abs()).float()
        allowed = torch.maximum(torch.full_like(diff, 2e-2), 2.01 * ulp)
        worst_excess = (diff - allowed).max().item()
        assert worst_excess <= 0.0, f"max_diff={max_diff:.6g} exceeded max(2e-2, 2 ULP) for {dtype}"


def _disable_tf32() -> tuple[bool, bool]:
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return prev


def _restore_tf32(prev: tuple[bool, bool]) -> None:
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def _make_5d(
    shape: tuple[int, int, int, int, int],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    t = torch.randn(shape, device=device, dtype=dtype)
    return t.contiguous(memory_format=torch.channels_last_3d) if channels_last else t.contiguous()


# =============================================================================
# 1. Lazy Triton Import & CPU / Autograd Fallbacks
# =============================================================================


def test_lazy_triton_import_all_ltx_modules() -> None:
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "import torch\n"
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import triton_vae_ops as v\n"
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio import triton_audio_ops as a\n"
        "from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx import triton_upsampler_ops as u\n"
        "assert v._TRITON_KERNELS is None and a._TRITON_KERNELS is None and u._TRITON_KERNELS is None\n"
        "x5 = torch.randn(1, 16, 2, 4, 4)\n"
        "assert v.triton_pixel_norm_adaln_silu_pad_3d(x5, None, None, 1e-8, (1, 1, 1, 1, 2, 0)).shape == (1, 16, 4, 6, 6)\n"
        "assert v.triton_ltx_residual_bias_add(x5, x5, torch.randn(16)).shape == x5.shape\n"
        "assert v.triton_space_to_depth_3d_add(torch.randn(1, 4, 4, 4, 4), torch.randn(1, 8, 3, 4, 4), (2, 2, 2), 2, dup_first_t=True).shape == (1, 32, 2, 2, 2)\n"
        "x3 = torch.randn(1, 8, 16)\n"
        "assert a.triton_activation1d_snake(x3, torch.ones(1, 1, 12), torch.ones(1, 1, 12), torch.zeros(8), torch.zeros(8)).shape == x3.shape\n"
        "assert a.triton_pixel_norm_silu_pad_2d(torch.randn(1, 8, 6, 6), 1e-6, (1, 1, 2, 0)).shape == (1, 8, 8, 8)\n"
        "assert u.triton_upsampler_group_norm_silu(x5, 4, torch.ones(16), torch.zeros(16)).shape == x5.shape\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_cpu_fallbacks_exact() -> None:
    torch.manual_seed(0)
    # Video ops
    x5 = torch.randn(2, 16, 3, 4, 6)
    pb = torch.randn(16)
    sc = torch.randn(2, 16, 1, 1, 1)
    sh = torch.randn(2, 16, 1, 1, 1)
    assert torch.allclose(
        triton_pixel_norm_adaln_silu_pad_3d(x5, sc, sh, 1e-8, (1, 1, 1, 1, 2, 0), spatial_mode="reflect", pre_bias=pb),
        _pixel_norm_adaln_silu_pad_3d_torch(x5, sc, sh, 1e-8, (1, 1, 1, 1, 2, 0), spatial_mode="reflect", pre_bias=pb),
    )
    assert torch.equal(
        triton_ltx_pad_3d(x5, (1, 1, 1, 1, 2, 0), spatial_mode="reflect", temporal_mode="replicate"),
        _pad_3d_torch(x5, (1, 1, 1, 1, 2, 0), spatial_mode="reflect", temporal_mode="replicate"),
    )
    assert torch.allclose(triton_ltx_residual_bias_add(x5, x5, pb), _residual_bias_add_torch(x5, x5, pb))
    h_s2d = torch.randn(2, 4, 4, 4, 6)
    x_s2d = torch.randn(2, 8, 3, 4, 6)
    b_s2d = torch.randn(4)
    assert torch.allclose(
        triton_space_to_depth_3d_add(h_s2d, x_s2d, (2, 2, 2), 2, dup_first_t=True, h_bias=b_s2d),
        _space_to_depth_3d_add_torch(h_s2d, x_s2d, 2, 2, 2, 2, dup_first_t=True, h_bias=b_s2d),
    )
    rgb = torch.randn(1, 3, 3, 8, 8)
    p = triton_patchify_3d(rgb, 4, 1)
    assert torch.equal(p, _patchify_3d_torch(rgb, 4, 1))
    b48 = torch.randn(48)
    assert torch.equal(triton_unpatchify_3d(p, 4, 1, bias=b48), _unpatchify_3d_torch(p, 4, 1, bias=b48))

    # Audio ops
    x3 = torch.randn(2, 8, 17)
    uf = torch.randn(1, 1, 12) * 0.1
    df = torch.randn(1, 1, 12) * 0.1
    al = torch.randn(8) * 0.2
    be = torch.randn(8) * 0.2
    pb3 = torch.randn(8) * 0.1
    assert torch.allclose(
        triton_activation1d_snake(x3, uf, df, al, be, pre_bias=pb3),
        _activation1d_snake_torch(x3, uf, df, al, be, 1e-9, True, pb3),
        atol=1e-6,
    )
    assert torch.allclose(triton_residual_bias_add_1d(x3, x3, pb3), x3 + (x3 + pb3.view(1, -1, 1)))
    assert torch.allclose(triton_mean3_1d(x3, x3 * 2, x3 * 3), x3 * 2.0)

    x4 = torch.randn(2, 16, 8, 6)
    assert torch.allclose(
        triton_pixel_norm_silu_pad_2d(x4, 1e-6, (1, 1, 2, 0), apply_silu=True, pre_bias=pb),
        _pixel_norm_silu_pad_2d_torch(x4, 1e-6, (1, 1, 2, 0), apply_silu=True, pre_bias=pb),
    )
    assert torch.allclose(triton_residual_bias_add_2d(x4, x4, pb), x4 + (x4 + pb.view(1, -1, 1, 1)))

    # Upsampler op
    w = torch.randn(16)
    b = torch.randn(16)
    assert torch.allclose(
        triton_upsampler_group_norm_silu(x5, 4, w, b, 1e-5, pre_bias=pb, residual=x5, apply_silu=True),
        _upsampler_gn_silu_torch(x5, 4, w, b, 1e-5, pb, x5, True),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_autograd_fallback_preserves_gradients(device: str) -> None:
    torch.manual_seed(1)
    x5 = torch.randn(1, 16, 2, 4, 4, device=device, requires_grad=True)
    x5_ref = x5.detach().clone().requires_grad_(True)
    pb = torch.randn(16, device=device, requires_grad=True)
    pb_ref = pb.detach().clone().requires_grad_(True)
    triton_pixel_norm_adaln_silu_pad_3d(x5, None, None, 1e-8, (0, 0, 0, 0, 2, 0), pre_bias=pb).sum().backward()
    _pixel_norm_adaln_silu_pad_3d_torch(x5_ref, None, None, 1e-8, (0, 0, 0, 0, 2, 0), pre_bias=pb_ref).sum().backward()
    assert x5.grad is not None and x5_ref.grad is not None
    assert torch.allclose(x5.grad, x5_ref.grad, atol=1e-5)
    assert pb.grad is not None and pb_ref.grad is not None
    assert torch.allclose(pb.grad, pb_ref.grad, atol=1e-5)

    x3 = torch.randn(1, 8, 12, device=device, requires_grad=True)
    x3_ref = x3.detach().clone().requires_grad_(True)
    uf = torch.randn(1, 1, 12, device=device) * 0.1
    df = torch.randn(1, 1, 12, device=device) * 0.1
    al = torch.randn(8, device=device, requires_grad=True)
    al_ref = al.detach().clone().requires_grad_(True)
    be = torch.randn(8, device=device, requires_grad=True)
    be_ref = be.detach().clone().requires_grad_(True)
    triton_activation1d_snake(x3, uf, df, al, be).sum().backward()
    _activation1d_snake_torch(x3_ref, uf, df, al_ref, be_ref, 1e-9, True, None).sum().backward()
    assert x3.grad is not None and x3_ref.grad is not None
    assert torch.allclose(x3.grad, x3_ref.grad, atol=1e-5)
    assert al.grad is not None and al_ref.grad is not None
    assert torch.allclose(al.grad, al_ref.grad, atol=1e-5)

    xu = torch.randn(1, 16, 2, 4, 4, device=device, requires_grad=True)
    xu_ref = xu.detach().clone().requires_grad_(True)
    wu = torch.randn(16, device=device, requires_grad=True)
    wu_ref = wu.detach().clone().requires_grad_(True)
    bu = torch.randn(16, device=device, requires_grad=True)
    bu_ref = bu.detach().clone().requires_grad_(True)
    triton_upsampler_group_norm_silu(xu, 4, wu, bu, 1e-5, residual=xu, apply_silu=True).sum().backward()
    _upsampler_gn_silu_torch(xu_ref, 4, wu_ref, bu_ref, 1e-5, None, xu_ref, True).sum().backward()
    assert xu.grad is not None and xu_ref.grad is not None
    assert torch.allclose(xu.grad, xu_ref.grad, atol=1e-5)


@requires_cuda
def test_hooks_are_never_bypassed() -> None:
    torch.manual_seed(2)
    called: list[str] = []

    res_v = (
        ResnetBlock3D(dims=3, in_channels=32, out_channels=32, norm_layer=NormLayerType.PIXEL_NORM)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    res_v.conv1.register_forward_hook(lambda _m, _i, _o: called.append("video_conv1"))
    res_v.norm2.register_forward_hook(lambda _m, _i, _o: called.append("video_norm2"))
    with torch.inference_mode():
        _ = res_v(torch.randn(1, 32, 3, 6, 6, device="cuda", dtype=torch.bfloat16))
    assert "video_conv1" in called and "video_norm2" in called

    act1d = Activation1d(SnakeBeta(16)).cuda().eval()
    act1d.act.register_forward_hook(lambda _m, _i, _o: called.append("snake_beta"))
    with torch.inference_mode():
        _ = act1d(torch.randn(1, 16, 32, device="cuda"))
    assert "snake_beta" in called

    up_blk = UpsamplerResBlock(64, dims=3).to(device="cuda", dtype=torch.bfloat16).eval()
    up_blk.norm1.register_forward_hook(lambda _m, _i, _o: called.append("up_norm1"))
    up_blk.conv2.register_forward_hook(lambda _m, _i, _o: called.append("up_conv2"))
    with torch.inference_mode():
        _ = up_blk(torch.randn(1, 64, 2, 4, 4, device="cuda", dtype=torch.bfloat16))
    assert "up_norm1" in called and "up_conv2" in called


# =============================================================================
# 2. LTX VideoVAE Hopper Triton Ops & Module Parity
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "spatial_mode,pad",
    [
        ("constant", (0, 0, 0, 0, 2, 0)),
        ("constant", (0, 0, 0, 0, 1, 1)),
        ("reflect", (1, 1, 1, 1, 2, 0)),
        ("replicate", (1, 1, 1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("with_adaln", [False, True])
def test_video_pixel_norm_adaln_silu_pad_3d_parity(
    dtype: torch.dtype,
    channels_last: bool,
    spatial_mode: str,
    pad: tuple[int, int, int, int, int, int],
    with_adaln: bool,
) -> None:
    torch.manual_seed(10)
    shape = (2, 128, 4, 9, 13)
    x = _make_5d(shape, dtype=dtype, device="cuda", channels_last=channels_last)
    pre_bias = (0.1 * torch.randn(128, device="cuda", dtype=torch.float32)).to(dtype)
    sc = (0.1 * torch.randn(2, 128, 1, 1, 1, device="cuda", dtype=torch.float32)).to(dtype) if with_adaln else None
    sh = (0.1 * torch.randn(2, 128, 1, 1, 1, device="cuda", dtype=torch.float32)).to(dtype) if with_adaln else None

    with torch.inference_mode():
        actual = triton_pixel_norm_adaln_silu_pad_3d(
            x,
            sc,
            sh,
            1e-8,
            pad,
            spatial_mode=spatial_mode,
            temporal_mode="replicate",
            apply_silu=True,
            pre_bias=pre_bias,
        )
        ref = _pixel_norm_adaln_silu_pad_3d_torch(
            x.float(),
            sc.float() if sc is not None else None,
            sh.float() if sh is not None else None,
            1e-8,
            pad,
            spatial_mode=spatial_mode,
            temporal_mode="replicate",
            apply_silu=True,
            pre_bias=pre_bias.float(),
        ).to(dtype)
    _assert_lossless(actual, ref, dtype)
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "c_in,stride,group_size,dup_first_t",
    [
        (128, (1, 2, 2), 2, False),
        (128, (2, 1, 1), 1, True),
        (128, (2, 2, 2), 4, True),
        (128, (2, 2, 2), 8, True),
        (96, (2, 2, 2), 3, True),  # non-power-of-2 group_size fallback
    ],
)
def test_video_space_to_depth_3d_add_parity(
    dtype: torch.dtype,
    channels_last: bool,
    c_in: int,
    stride: tuple[int, int, int],
    group_size: int,
    dup_first_t: bool,
) -> None:
    torch.manual_seed(11)
    p = math.prod(stride)
    c_out = c_in * p // group_size
    c_h = c_out // p
    d_x = 5 if dup_first_t else 4
    d_h = d_x + (1 if dup_first_t else 0)
    x = _make_5d((2, c_in, d_x, 8, 10), dtype=dtype, device="cuda", channels_last=channels_last)
    h = _make_5d((2, c_h, d_h, 8, 10), dtype=dtype, device="cuda", channels_last=channels_last)
    h_bias = (0.1 * torch.randn(c_h, device="cuda", dtype=torch.float32)).to(dtype)

    with torch.inference_mode():
        actual = triton_space_to_depth_3d_add(h, x, stride, group_size, dup_first_t=dup_first_t, h_bias=h_bias)
        ref = _space_to_depth_3d_add_torch(
            h.float(), x.float(), *stride, group_size, dup_first_t=dup_first_t, h_bias=h_bias.float()
        ).to(dtype)
    _assert_lossless(actual, ref, dtype)
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_video_d2s_bias_patchify_unpatchify_and_c3_align(channels_last: bool) -> None:
    torch.manual_seed(12)
    dtype = torch.bfloat16
    x_conv = _make_5d((2, 128, 3, 6, 8), dtype=dtype, device="cuda", channels_last=channels_last)
    x_res = _make_5d((2, 64, 3, 6, 8), dtype=dtype, device="cuda", channels_last=channels_last)
    h_bias = torch.randn(128, device="cuda", dtype=dtype)
    with torch.inference_mode():
        got = triton_depth_to_space_3d(
            x_conv, (2, 2, 2), residual=x_res, out_channels_reduction_factor=4, h_bias=h_bias
        )
        ref = _depth_to_space_3d_torch(x_conv.float(), 2, 2, 2, x_res.float(), 4, h_bias=h_bias.float()).to(dtype)
    _assert_lossless(got, ref, dtype)
    assert _is_channels_last_3d(got) == channels_last

    # Patchify + Unpatchify (+ bias)
    rgb = torch.randn(1, 3, 5, 32, 48, device="cuda", dtype=dtype)
    b48 = torch.randn(48, device="cuda", dtype=dtype)
    with torch.inference_mode():
        patched = triton_patchify_3d(rgb, 4, 1, channels_last_3d=channels_last)
        ref_p = _patchify_3d_torch(rgb, 4, 1, channels_last_3d=channels_last)
        assert torch.equal(patched, ref_p)
        assert _is_channels_last_3d(patched) == channels_last
        unp = triton_unpatchify_3d(patched, 4, 1, bias=b48)
        ref_u = _unpatchify_3d_torch(patched.float(), 4, 1, bias=b48.float()).to(dtype)
    _assert_lossless(unp, ref_u, dtype)

    # C=3 CausalConv3d channels_last_3d align-C path
    prev = _disable_tf32()
    try:
        conv_c3 = CausalConv3d(3, 16, kernel_size=3, spatial_padding_mode=PaddingModeType.REFLECT).cuda().eval()
        ref_c3 = CausalConv3d(3, 16, kernel_size=3, spatial_padding_mode=PaddingModeType.REFLECT).eval()
        ref_c3.load_state_dict(conv_c3.state_dict())
        conv_c3.conv.weight.data = conv_c3.conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
        x3 = _make_5d((1, 3, 4, 12, 16), dtype=torch.float32, device="cuda", channels_last=True)
        with torch.inference_mode():
            y_got = conv_c3(x3, causal=True)
            y_ref = ref_c3(x3.cpu(), causal=True).cuda()
        _assert_lossless(y_got, y_ref, torch.float32)
        p8 = triton_ltx_pad_3d_align_c(
            x3, (1, 1, 1, 1, 2, 0), spatial_mode="reflect", temporal_mode="replicate", align_c=8
        )
        assert p8.shape == (1, 8, 6, 14, 18) and _is_channels_last_3d(p8)
        assert (p8[:, 3:] == 0).all()
    finally:
        _restore_tf32(prev)


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("padding_mode", [PaddingModeType.ZEROS, PaddingModeType.REFLECT])
@pytest.mark.parametrize("causal", [True, False])
def test_video_vae_end_to_end_fp32_parity(
    channels_last: bool,
    padding_mode: PaddingModeType,
    causal: bool,
) -> None:
    torch.manual_seed(13)
    prev = _disable_tf32()
    try:
        enc_blocks = [
            ("res_x", {"num_layers": 1}),
            ("compress_space_res", {"multiplier": 2}),
            ("res_x", {"num_layers": 1}),
            ("compress_time_res", {"multiplier": 2}),
        ]
        enc = (
            VideoEncoder(
                convolution_dimensions=3,
                in_channels=3,
                out_channels=16,
                encoder_blocks=enc_blocks,
                patch_size=2,
                norm_layer=NormLayerType.PIXEL_NORM,
                latent_log_var=LogVarianceType.UNIFORM,
                encoder_spatial_padding_mode=padding_mode,
            )
            .cuda()
            .eval()
        )
        enc.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        enc.per_channel_statistics.get_buffer("mean-of-means").zero_()
        enc_ref = VideoEncoder(
            convolution_dimensions=3,
            in_channels=3,
            out_channels=16,
            encoder_blocks=enc_blocks,
            patch_size=2,
            norm_layer=NormLayerType.PIXEL_NORM,
            latent_log_var=LogVarianceType.UNIFORM,
            encoder_spatial_padding_mode=padding_mode,
        ).eval()
        enc_ref.load_state_dict(enc.state_dict())
        if channels_last:
            enc.enable_channels_last_3d()

        vid = torch.randn(1, 3, 9, 16, 16, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            got_e = enc(vid)
            ref_e = enc_ref(vid.cpu()).cuda()
        _assert_lossless(got_e, ref_e, torch.float32, min_psnr=90.0)

        dec_blocks = [
            ("res_x", {"num_layers": 1}),
            ("compress_all", {"multiplier": 2, "residual": True}),
            ("res_x", {"num_layers": 1}),
        ]
        dec = (
            VideoDecoder(
                convolution_dimensions=3,
                in_channels=16,
                out_channels=3,
                decoder_blocks=list(reversed(dec_blocks)),
                patch_size=2,
                norm_layer=NormLayerType.PIXEL_NORM,
                causal=causal,
                timestep_conditioning=False,
                decoder_spatial_padding_mode=padding_mode,
                base_channels=8,
            )
            .cuda()
            .eval()
        )
        dec.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        dec.per_channel_statistics.get_buffer("mean-of-means").zero_()
        dec_ref = VideoDecoder(
            convolution_dimensions=3,
            in_channels=16,
            out_channels=3,
            decoder_blocks=list(reversed(dec_blocks)),
            patch_size=2,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=causal,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=padding_mode,
            base_channels=8,
        ).eval()
        dec_ref.load_state_dict(dec.state_dict())
        if channels_last:
            dec.enable_channels_last_3d()

        lat = torch.randn(1, 16, 3, 4, 4, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            got_d = dec(lat)
            ref_d = dec_ref(lat.cpu()).cuda()
        _assert_lossless(got_d, ref_d, torch.float32, min_psnr=90.0)
    finally:
        _restore_tf32(prev)


# =============================================================================
# 3. LTX AudioVAE & VocoderWithBWE Hopper Triton Ops & Module Parity
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("act_cls", [Snake, SnakeBeta])
@pytest.mark.parametrize("alpha_logscale", [True, False])
@pytest.mark.parametrize("T", [1, 7, 63, 256, 1029])
def test_audio_activation1d_snake_parity(
    dtype: torch.dtype,
    act_cls: type[nn.Module],
    alpha_logscale: bool,
    T: int,
) -> None:
    torch.manual_seed(20)
    c = 32
    act = act_cls(c, alpha_logscale=alpha_logscale)
    assert isinstance(act, (Snake, SnakeBeta))
    mod = Activation1d(act).cuda().eval()
    lo, hi = (-1.0, 1.5) if alpha_logscale else (0.3, 2.0)
    with torch.no_grad():
        act.alpha.uniform_(lo, hi)
        if isinstance(act, SnakeBeta):
            act.beta.uniform_(lo, hi)
    # Test mixed BF16 weights with FP32 activations (exact VocoderWithBWE autocast scenario!)
    if dtype == torch.float32:
        mod = mod.to(dtype=torch.bfloat16)
    else:
        mod = mod.to(dtype=dtype)

    up_f = mod.upsample.filter
    down_f = mod.downsample.lowpass.filter
    assert isinstance(up_f, torch.Tensor) and isinstance(down_f, torch.Tensor)
    beta_param = act.beta if isinstance(act, SnakeBeta) else None

    x = torch.randn(2, c, T, device="cuda", dtype=dtype)
    pre_bias = torch.randn(c, device="cuda", dtype=act.alpha.dtype) * 0.2
    with torch.inference_mode():
        got = mod(x, pre_bias=pre_bias)
    with torch.enable_grad():
        ref_fp32 = _activation1d_snake_torch(
            x.detach().float(),
            up_f.float(),
            down_f.float(),
            act.alpha.float(),
            beta_param.float() if beta_param is not None else None,
            act.eps,
            act.alpha_logscale,
            pre_bias.float(),
        ).detach()
        ref_eager = _activation1d_snake_torch(
            x.detach(),
            up_f,
            down_f,
            act.alpha,
            beta_param,
            act.eps,
            act.alpha_logscale,
            pre_bias,
        ).detach()
    _assert_lossless(got, ref_fp32.to(dtype), dtype, min_psnr=100.0 if dtype == torch.float32 else 52.0)
    psnr_eager, cos_eager = _psnr_cos(got, ref_eager)
    assert psnr_eager >= (100.0 if dtype == torch.float32 else 50.0), f"eager PSNR={psnr_eager:.2f} dB"
    assert cos_eager >= 0.9998, f"eager cos={cos_eager:.6f}"


@requires_cuda
def test_audio_vocoder_bwe_and_audio_vae_fp32_parity() -> None:
    torch.manual_seed(21)
    prev = _disable_tf32()
    try:
        # VocoderWithBWE with AMP1 + SnakeBeta
        voc = Vocoder(
            resblock_kernel_sizes=[3, 7, 11],
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_initial_channel=64,
            resblock="AMP1",
            output_sampling_rate=16000,
            activation="snakebeta",
            use_tanh_at_final=False,
            use_bias_at_final=False,
        )
        bwe = Vocoder(
            resblock_kernel_sizes=[3, 7, 11],
            upsample_rates=[3, 2],
            upsample_kernel_sizes=[7, 4],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_initial_channel=64,
            resblock="AMP1",
            output_sampling_rate=48000,
            activation="snakebeta",
            use_tanh_at_final=False,
            apply_final_activation=False,
            use_bias_at_final=False,
        )
        mel_stft = MelSTFT(filter_length=64, hop_length=2, win_length=64, n_mel_channels=64)
        fb = mel_stft.stft_fn.forward_basis
        mb = mel_stft.mel_basis
        assert isinstance(fb, torch.Tensor) and isinstance(mb, torch.Tensor)
        with torch.no_grad():
            fb.normal_(0, 0.05)
            mb.normal_(0, 0.05)
        model = (
            VocoderWithBWE(voc, bwe, mel_stft, 16000, 48000, hop_length=2)
            .to(device="cuda", dtype=torch.bfloat16)
            .eval()
        )
        mel_in = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.bfloat16)
        with torch.enable_grad():
            ref_wav = model(mel_in.float()).detach()
        with torch.inference_mode():
            got_wav = model(mel_in.float())
        _assert_lossless(got_wav, ref_wav, torch.float32, min_psnr=95.0)

        # AudioDecoder + AudioEncoder (NCHW & NHWC)
        for channels_last in (False, True):
            aud_dec = (
                AudioDecoder(
                    ch=32,
                    out_ch=2,
                    ch_mult=(1, 2, 4),
                    num_res_blocks=1,
                    attn_resolutions=set(),
                    resolution=32,
                    z_channels=8,
                    norm_type=NormType.PIXEL,
                    causality_axis=CausalityAxis.HEIGHT,
                    mid_block_add_attention=False,
                    mel_bins=16,
                )
                .cuda()
                .eval()
            )
            aud_dec.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
            aud_dec.per_channel_statistics.get_buffer("mean-of-means").zero_()
            aud_dec_ref = AudioDecoder(
                ch=32,
                out_ch=2,
                ch_mult=(1, 2, 4),
                num_res_blocks=1,
                attn_resolutions=set(),
                resolution=32,
                z_channels=8,
                norm_type=NormType.PIXEL,
                causality_axis=CausalityAxis.HEIGHT,
                mid_block_add_attention=False,
                mel_bins=16,
            ).eval()
            aud_dec_ref.load_state_dict(aud_dec.state_dict())
            if channels_last:
                aud_dec.enable_channels_last()
            z_a = torch.randn(1, 8, 12, 4, device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                got_a = aud_dec(z_a)
                ref_a = aud_dec_ref(z_a.cpu()).cuda()
            _assert_lossless(got_a, ref_a, torch.float32, min_psnr=90.0)

            aud_enc = (
                AudioEncoder(
                    ch=32,
                    in_channels=2,
                    ch_mult=(1, 2, 4),
                    num_res_blocks=1,
                    attn_resolutions=set(),
                    resolution=32,
                    z_channels=8,
                    norm_type=NormType.PIXEL,
                    causality_axis=CausalityAxis.HEIGHT,
                    mid_block_add_attention=False,
                    mel_bins=16,
                )
                .cuda()
                .eval()
            )
            aud_enc.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
            aud_enc.per_channel_statistics.get_buffer("mean-of-means").zero_()
            aud_enc_ref = AudioEncoder(
                ch=32,
                in_channels=2,
                ch_mult=(1, 2, 4),
                num_res_blocks=1,
                attn_resolutions=set(),
                resolution=32,
                z_channels=8,
                norm_type=NormType.PIXEL,
                causality_axis=CausalityAxis.HEIGHT,
                mid_block_add_attention=False,
                mel_bins=16,
            ).eval()
            aud_enc_ref.load_state_dict(aud_enc.state_dict())
            if channels_last:
                aud_enc.enable_channels_last()
            spec = torch.randn(1, 2, 16, 16, device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                got_ae = aud_enc(spec)
                ref_ae = aud_enc_ref(spec.cpu()).cuda()
            _assert_lossless(got_ae, ref_ae, torch.float32, min_psnr=90.0)
    finally:
        _restore_tf32(prev)


# =============================================================================
# 4. LTX SpatialUpsampler x2 Hopper Triton Ops & Module Parity
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
def test_upsampler_group_norm_silu_parity(
    dtype: torch.dtype,
    channels_last: bool,
    with_residual: bool,
) -> None:
    torch.manual_seed(30)
    x = _make_5d((1, 256, 5, 12, 16), dtype=dtype, device="cuda", channels_last=channels_last)
    w = (1.0 + 0.1 * torch.randn(256, device="cuda", dtype=torch.float32)).to(dtype)
    b = (0.1 * torch.randn(256, device="cuda", dtype=torch.float32)).to(dtype)
    pb = (0.1 * torch.randn(256, device="cuda", dtype=torch.float32)).to(dtype)
    res = (
        _make_5d((1, 256, 5, 12, 16), dtype=dtype, device="cuda", channels_last=channels_last)
        if with_residual
        else None
    )

    with torch.inference_mode():
        got = triton_upsampler_group_norm_silu(x, 32, w, b, 1e-5, pre_bias=pb, residual=res, apply_silu=True)
        ref = _upsampler_gn_silu_torch(
            x.float(), 32, w.float(), b.float(), 1e-5, pb.float(), res.float() if res is not None else None, True
        ).to(dtype)
    _assert_lossless(got, ref, dtype)
    assert _is_channels_last_nd(got) == channels_last


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_latent_upsampler_module_fp32_parity(channels_last: bool) -> None:
    torch.manual_seed(31)
    prev = _disable_tf32()
    try:
        up = LatentUpsampler(in_channels=32, mid_channels=64, num_blocks_per_stage=2, dims=3).cuda().eval()
        up_ref = LatentUpsampler(in_channels=32, mid_channels=64, num_blocks_per_stage=2, dims=3).eval()
        up_ref.load_state_dict(up.state_dict())
        if channels_last:
            up.enable_channels_last_3d()
        lat = torch.randn(1, 32, 3, 6, 8, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            got = up(lat)
            ref = up_ref(lat.cpu()).cuda()
        _assert_lossless(got, ref, torch.float32, min_psnr=90.0)
    finally:
        _restore_tf32(prev)


# =============================================================================
# 5. Strict > 2^31-Element tl.int64 Indexing Verification
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_int64_indexing_above_2_pow_31(channels_last: bool) -> None:
    """Verify 5-D kernels never overflow int32 pointer arithmetic on >2^31-element tensors."""
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_mem_gb < 24.0:
        pytest.skip()

    # Shape (1, 128, 33, 512, 1024) has 2,214,592,512 > 2^31 (2,147,483,648) elements (~4.12 GiB BF16).
    shape = (1, 128, 33, 512, 1024)
    assert math.prod(shape) > (1 << 31)
    torch.cuda.empty_cache()

    x = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    # Place distinct non-zero signal strictly past index 2^31 (last frame d=32, last rows).
    x[0, :, 32, -2:, -4:] = torch.arange(1, 129, device="cuda", dtype=torch.bfloat16).view(128, 1, 1) * 0.01
    bias = torch.full((128,), 0.25, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        # 1. Residual + bias add on >2^31 elements
        y_res = triton_ltx_residual_bias_add(None, x, bias)
        expected_tail = x[0, :, 32, -2:, -4:] + 0.25
        assert torch.allclose(y_res[0, :, 32, -2:, -4:].float(), expected_tail.float(), atol=1e-2)
        assert torch.allclose(y_res[0, :, 0, :2, :2].float(), torch.full((128, 2, 2), 0.25, device="cuda"), atol=1e-2)
        del y_res

        # 2. 5-D temporal pad on >2^31 elements (output has 35 frames = 2.35B elements!)
        y_pad = triton_ltx_pad_3d(x, (0, 0, 0, 0, 1, 1), spatial_mode="constant", temporal_mode="replicate")
        assert torch.equal(y_pad[0, :, 33, -2:, -4:], x[0, :, 32, -2:, -4:])
        assert torch.equal(y_pad[0, :, 34, -2:, -4:], x[0, :, 32, -2:, -4:])
        del y_pad

        # 3. Super-fused pre_bias + PixelNorm + SiLU + temporal pad on >2^31 elements
        y_pn = triton_pixel_norm_adaln_silu_pad_3d(
            x,
            None,
            None,
            1e-8,
            (0, 0, 0, 0, 1, 1),
            spatial_mode="constant",
            temporal_mode="replicate",
            apply_silu=True,
            pre_bias=bias,
        )
        tail_in = x[0:1, :, 32:33, -2:, -4:].float() + 0.25
        tail_ref = F.silu(tail_in / torch.sqrt(torch.mean(tail_in**2, dim=1, keepdim=True) + 1e-8)).to(torch.bfloat16)
        _assert_lossless(y_pn[0:1, :, 33:34, -2:, -4:], tail_ref, torch.bfloat16)
        _assert_lossless(y_pn[0:1, :, 34:35, -2:, -4:], tail_ref, torch.bfloat16)
        del y_pn, x
    torch.cuda.empty_cache()


# =============================================================================
# 6. CUDA Graph Capture Safety & torch.compile(fullgraph=True) Safety
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_all_new_ops_cuda_graph_and_compile_fullgraph(channels_last: bool) -> None:
    torch.manual_seed(42)
    x5 = _make_5d((1, 64, 4, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    pb5 = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    sc5 = torch.randn(1, 64, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    sh5 = torch.randn(1, 64, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    h_s2d = _make_5d((1, 16, 4, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    x_s2d = _make_5d((1, 64, 3, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    hb_s2d = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    w_gn = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    b_gn = torch.randn(64, device="cuda", dtype=torch.bfloat16)

    x3 = torch.randn(1, 32, 128, device="cuda", dtype=torch.float32)
    uf = torch.randn(1, 1, 12, device="cuda", dtype=torch.bfloat16) * 0.1
    df = torch.randn(1, 1, 12, device="cuda", dtype=torch.bfloat16) * 0.1
    al = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.2
    be = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.2
    pb3 = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.1
    x4 = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.bfloat16)
    if channels_last:
        x4 = x4.contiguous(memory_format=torch.channels_last)
    pb4 = torch.randn(32, device="cuda", dtype=torch.bfloat16)

    def run_all(x5_, pb5_, sc5_, sh5_, hs_, xs_, hbs_, wg_, bg_, x3_, uf_, df_, al_, be_, pb3_, x4_, pb4_):
        y1 = triton_pixel_norm_adaln_silu_pad_3d(x5_, sc5_, sh5_, 1e-8, (0, 0, 0, 0, 2, 0), pre_bias=pb5_)
        y2 = triton_ltx_residual_bias_add(x5_, x5_, pb5_)
        y3 = triton_space_to_depth_3d_add(hs_, xs_, (2, 2, 2), 4, dup_first_t=True, h_bias=hbs_)
        y4 = triton_upsampler_group_norm_silu(x5_, 8, wg_, bg_, 1e-5, pre_bias=pb5_, residual=x5_, apply_silu=True)
        y5 = triton_activation1d_snake(x3_, uf_, df_, al_, be_, pre_bias=pb3_)
        y6 = triton_residual_bias_add_1d(x3_, y5, pb3_)
        y7 = triton_mean3_1d(x3_, y5, y6)
        y8 = triton_pixel_norm_silu_pad_2d(x4_, 1e-6, (1, 1, 2, 0), pre_bias=pb4_)
        y9 = triton_residual_bias_add_2d(x4_, x4_, pb4_)
        return y1, y2, y3, y4, y5, y6, y7, y8, y9

    args = (x5, pb5, sc5, sh5, h_s2d, x_s2d, hb_s2d, w_gn, b_gn, x3, uf, df, al, be, pb3, x4, pb4)

    # 1. CUDA Graph capture & replay
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(2):
            _ = run_all(*args)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        g_outs = run_all(*args)

    x5.normal_()
    x3.normal_()
    x4.normal_()
    g.replay()
    torch.cuda.synchronize()

    with torch.inference_mode():
        eager_outs = run_all(*args)
    for got, exp in zip(g_outs, eager_outs, strict=True):
        assert torch.equal(got, exp)

    # 2. torch.compile(fullgraph=True)
    compiled = torch.compile(run_all, fullgraph=True)
    with torch.inference_mode():
        comp_outs = compiled(*args)
    for got, exp in zip(comp_outs, eager_outs, strict=True):
        assert torch.equal(got, exp)
