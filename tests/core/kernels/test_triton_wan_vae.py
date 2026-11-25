# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit, parity, torch.compile, and int64-indexing tests for WanVideoVAE / WanVideoVAE38 Triton kernels."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    AttentionBlock,
    AvgDown3D,
    Down_ResidualBlock,
    DupUp3D,
    Resample,
    ResidualBlock,
    RMS_norm,
    Up_ResidualBlock,
    Upsample,
    VideoVAE38_,
    VideoVAE_,
    patchify,
    unpatchify,
)
from worldfoundry.core.kernels.triton_wan_vae import (
    can_use_wan_vae_rms_norm,
    is_channels_last_2d,
    is_channels_last_3d,
    wan_vae_avg_down_3d,
    wan_vae_avg_down_3d_add,
    wan_vae_causal_pad_cache,
    wan_vae_dup_up_3d,
    wan_vae_dup_up_3d_add,
    wan_vae_nearest_upsample_2x,
    wan_vae_patchify_2x,
    wan_vae_residual_bias_add,
    wan_vae_rms_norm,
    wan_vae_rms_norm_silu_causal_pad,
    wan_vae_time_interleave_upsample_2x,
    wan_vae_unpatchify_2x,
    wan_vae_zero_pad_2d_rb,
)


def test_cpu_fallback_and_eligibility() -> None:
    x = torch.randn(1, 16, 3, 8, 8)
    assert not can_use_wan_vae_rms_norm(x)
    rms = RMS_norm(16, images=False).eval()
    with torch.inference_mode():
        y = rms(x)
    ref = F.normalize(x, dim=1) * rms.scale * rms.gamma
    assert torch.allclose(y, ref, atol=1e-6)


def test_cpu_fallbacks_for_all_wan_vae_ops() -> None:
    """Verify every public wrapper runs cleanly on CPU without importing Triton or CUDA."""
    torch.manual_seed(0)
    with torch.inference_mode():
        x5 = torch.randn(1, 16, 3, 8, 8)
        g = torch.randn(16, 1, 1, 1)
        b = torch.randn(16)
        pb = torch.randn(16)
        y_rms = wan_vae_rms_norm(x5, g, b.view(16, 1, 1, 1), scale=4.0, apply_silu=True, pre_bias=pb)
        ref_rms = F.silu(F.normalize(x5 + pb.view(1, 16, 1, 1, 1), dim=1) * 4.0 * g + b.view(1, 16, 1, 1, 1))
        assert torch.allclose(y_rms, ref_rms, atol=1e-5)

        out_pad, out_cache = wan_vae_rms_norm_silu_causal_pad(
            x5,
            g,
            None,
            cache_in=None,
            pad_w=1,
            pad_h=1,
            pad_t_left=2,
            scale=4.0,
            write_cache=True,
            cache_t=2,
            pre_bias=pb,
        )
        assert out_pad.shape == (1, 16, 5, 10, 10)
        assert out_cache.shape == (1, 16, 2, 8, 8)

        cpad, ccache = wan_vae_causal_pad_cache(
            x5[:, :3], cache_in=None, pad_w=1, pad_h=1, pad_t_left=2, write_cache=True, cache_t=2, align_c=8
        )
        assert cpad.shape == (1, 8, 5, 10, 10)
        assert ccache.shape == (1, 3, 2, 8, 8)

        radd = wan_vae_residual_bias_add(x5, x5 * 2, b, pb)
        assert torch.allclose(radd, (x5 + pb.view(1, 16, 1, 1, 1)) + (x5 * 2 + b.view(1, 16, 1, 1, 1)), atol=1e-5)

        x4 = torch.randn(2, 8, 4, 6)
        assert wan_vae_nearest_upsample_2x(x4).shape == (2, 8, 8, 12)
        assert wan_vae_zero_pad_2d_rb(x4).shape == (2, 8, 5, 7)

        x_ti = torch.randn(1, 16, 2, 4, 6)
        assert wan_vae_time_interleave_upsample_2x(x_ti, torch.randn(16)).shape == (4, 8, 8, 12)

        assert wan_vae_dup_up_3d(x5, out_channels=8, factor_t=2, factor_s=2, first_chunk=True).shape == (
            1,
            8,
            5,
            16,
            16,
        )
        assert wan_vae_dup_up_3d_add(
            torch.randn(1, 8, 5, 16, 16), x5, torch.randn(8), out_channels=8, factor_t=2, factor_s=2, first_chunk=True
        ).shape == (1, 8, 5, 16, 16)

        assert wan_vae_avg_down_3d(x5, out_channels=32, factor_t=2, factor_s=2).shape == (1, 32, 2, 4, 4)
        assert wan_vae_avg_down_3d_add(
            torch.randn(1, 32, 2, 4, 4), x5, torch.randn(32), out_channels=32, factor_t=2, factor_s=2
        ).shape == (1, 32, 2, 4, 4)

        rgb = torch.randn(1, 3, 2, 8, 8)
        patched = wan_vae_patchify_2x(rgb)
        assert patched.shape == (1, 12, 2, 4, 4)
        assert torch.equal(wan_vae_unpatchify_2x(patched), rgb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("with_pre_bias", [False, True])
@pytest.mark.parametrize("apply_silu", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wan_vae_rms_norm_parity(
    channels_last: bool,
    with_bias: bool,
    with_pre_bias: bool,
    apply_silu: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    x = torch.randn(1, 96, 3, 24, 32, device=device, dtype=dtype)
    # Include an all-zero pixel to verify eps=1e-12 clamp safety.
    x[:, :, 0, 0, 0] = 0.0
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    gamma = torch.randn(96, 1, 1, 1, device=device, dtype=dtype)
    bias = torch.randn(96, 1, 1, 1, device=device, dtype=dtype) if with_bias else None
    pre_bias = torch.randn(96, device=device, dtype=dtype) if with_pre_bias else None

    with torch.inference_mode():
        inp = x + pre_bias.view(1, 96, 1, 1, 1) if pre_bias is not None else x
        ref = F.normalize(inp, dim=1) * math.sqrt(96) * gamma
        if bias is not None:
            ref = ref + bias
        if apply_silu:
            ref = F.silu(ref)
        out = wan_vae_rms_norm(x, gamma, bias, scale=math.sqrt(96), apply_silu=apply_silu, pre_bias=pre_bias)

    assert is_channels_last_3d(out) == channels_last
    tol = 0.08 if dtype == torch.bfloat16 else 1e-4
    diff = (ref.float() - out.float()).abs()
    assert diff.max().item() <= tol
    cos = F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()
    assert cos >= 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_pre_bias", [False, True])
@pytest.mark.parametrize("t_in,has_prev", [(1, False), (1, True), (3, False), (3, True)])
def test_wan_vae_super_fused_causal_pad_parity(
    channels_last: bool,
    with_pre_bias: bool,
    t_in: int,
    has_prev: bool,
) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(1, 96, t_in, 16, 20, device=device, dtype=dtype)
    cache_in = torch.randn(1, 96, 2, 16, 20, device=device, dtype=dtype) if has_prev else None
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
        if cache_in is not None:
            cache_in = cache_in.to(memory_format=torch.channels_last_3d)
    gamma = torch.randn(96, 1, 1, 1, device=device, dtype=dtype)
    pre_bias = torch.randn(96, device=device, dtype=dtype) if with_pre_bias else None

    with torch.inference_mode():
        inp = x + pre_bias.view(1, 96, 1, 1, 1) if pre_bias is not None else x
        y = F.silu(F.normalize(inp, dim=1) * math.sqrt(96) * gamma)
        ref_cache = y[:, :, -2:].clone()
        if ref_cache.shape[2] < 2 and cache_in is not None:
            ref_cache = torch.cat([cache_in[:, :, -1:, :, :], ref_cache], dim=2)
        if cache_in is not None:
            y_cat = torch.cat([cache_in, y], dim=2)
            pad_t_rem = 0
        else:
            y_cat = y
            pad_t_rem = 2
        ref_pad = F.pad(y_cat, (1, 1, 1, 1, pad_t_rem, 0))

        out_pad, out_cache = wan_vae_rms_norm_silu_causal_pad(
            x,
            gamma,
            None,
            cache_in=cache_in,
            pad_w=1,
            pad_h=1,
            pad_t_left=2,
            scale=math.sqrt(96),
            write_cache=True,
            cache_t=2,
            pre_bias=pre_bias,
        )

    assert out_pad.shape == ref_pad.shape
    assert out_cache.shape == ref_cache.shape
    assert is_channels_last_3d(out_pad) == channels_last
    assert is_channels_last_3d(out_cache) == channels_last
    assert F.cosine_similarity(ref_pad.float().flatten(), out_pad.float().flatten(), dim=0).item() >= 0.9999
    assert F.cosine_similarity(ref_cache.float().flatten(), out_cache.float().flatten(), dim=0).item() >= 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("mode", ["std", "rep_zero_left", "align_c_3_to_32", "downsample3d_pre_bias"])
def test_wan_vae_causal_pad_cache_bit_exact(channels_last: bool, mode: str) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    with torch.inference_mode():
        if mode == "align_c_3_to_32":
            full = torch.randn(1, 3, 6, 16, 20, device=device, dtype=dtype)
            if channels_last:
                full = full.to(memory_format=torch.channels_last_3d)
            x = full[:, :, 1:5]  # non-contiguous temporal slice
            cache_in = torch.randn(1, 3, 1, 16, 20, device=device, dtype=dtype)
            if channels_last:
                cache_in = cache_in.to(memory_format=torch.channels_last_3d)
            out_pad, out_cache = wan_vae_causal_pad_cache(
                x, cache_in=cache_in, pad_w=1, pad_h=1, pad_t_left=2, write_cache=True, cache_t=2, align_c=32
            )
            ref_pad = F.pad(torch.cat([cache_in, x], dim=2), (1, 1, 1, 1, 1, 0))
            assert out_pad.shape == (1, 32, 6, 18, 22)
            assert is_channels_last_3d(out_pad)
            assert torch.equal(out_pad[:, :3], ref_pad)
            assert torch.all(out_pad[:, 3:] == 0)
            assert torch.equal(out_cache, x[:, :, -2:])
        elif mode == "rep_zero_left":
            x = torch.randn(1, 32, 1, 12, 16, device=device, dtype=dtype)
            if channels_last:
                x = x.to(memory_format=torch.channels_last_3d)
            out_pad, out_cache = wan_vae_causal_pad_cache(
                x, cache_in=None, pad_w=0, pad_h=0, pad_t_left=2, write_cache=True, cache_t=2, pad_cache_zero_left=True
            )
            ref_pad = F.pad(x, (0, 0, 0, 0, 2, 0))
            ref_cache = torch.cat([torch.zeros_like(x), x], dim=2)
            assert is_channels_last_3d(out_pad) == channels_last
            assert is_channels_last_3d(out_cache) == channels_last
            assert torch.equal(out_pad, ref_pad)
            assert torch.equal(out_cache, ref_cache)
        elif mode == "downsample3d_pre_bias":
            x = torch.randn(1, 64, 4, 12, 16, device=device, dtype=dtype)
            cache_in = torch.randn(1, 64, 1, 12, 16, device=device, dtype=dtype)
            if channels_last:
                x = x.to(memory_format=torch.channels_last_3d)
                cache_in = cache_in.to(memory_format=torch.channels_last_3d)
            pb = torch.randn(64, device=device, dtype=dtype)
            out_pad, out_cache = wan_vae_causal_pad_cache(
                x, cache_in=cache_in, pad_w=0, pad_h=0, pad_t_left=1, write_cache=True, cache_t=1, pre_bias=pb
            )
            x_b = x + pb.view(1, 64, 1, 1, 1)
            ref_pad = torch.cat([cache_in, x_b], dim=2)
            ref_cache = x_b[:, :, -1:]
            assert is_channels_last_3d(out_pad) == channels_last
            assert torch.equal(out_pad, ref_pad)
            assert torch.equal(out_cache, ref_cache)
        else:
            x = torch.randn(1, 48, 1, 12, 16, device=device, dtype=dtype)
            cache_in = torch.randn(1, 48, 2, 12, 16, device=device, dtype=dtype)
            if channels_last:
                x = x.to(memory_format=torch.channels_last_3d)
                cache_in = cache_in.to(memory_format=torch.channels_last_3d)
            out_pad, out_cache = wan_vae_causal_pad_cache(
                x, cache_in=cache_in, pad_w=1, pad_h=1, pad_t_left=2, write_cache=True, cache_t=2
            )
            ref_pad = F.pad(torch.cat([cache_in, x], dim=2), (1, 1, 1, 1, 0, 0))
            ref_cache = torch.cat([cache_in[:, :, -1:], x], dim=2)
            assert is_channels_last_3d(out_pad) == channels_last
            assert torch.equal(out_pad, ref_pad)
            assert torch.equal(out_cache, ref_cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ndim", [4, 5])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("has_x", [False, True])
@pytest.mark.parametrize("has_x_bias", [False, True])
def test_wan_vae_residual_bias_add_bit_exact(
    ndim: int,
    channels_last: bool,
    has_x: bool,
    has_x_bias: bool,
) -> None:
    if not has_x and has_x_bias:
        return
    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (2, 96, 12, 16) if ndim == 4 else (1, 96, 4, 12, 16)
    mem_fmt = torch.channels_last if ndim == 4 else torch.channels_last_3d
    h = torch.randn(shape, device=device, dtype=dtype)
    x = torch.randn(shape, device=device, dtype=dtype) if has_x else None
    if channels_last:
        h = h.to(memory_format=mem_fmt)
        if x is not None:
            x = x.to(memory_format=mem_fmt)
    hb = torch.randn(96, device=device, dtype=dtype)
    xb = torch.randn(96, device=device, dtype=dtype) if has_x_bias else None

    with torch.inference_mode():
        out = wan_vae_residual_bias_add(x, h, hb, xb)
        bsh = (1, 96, 1, 1) if ndim == 4 else (1, 96, 1, 1, 1)
        ref = h + hb.view(bsh)
        if x is not None:
            ref = (x + xb.view(bsh) if xb is not None else x) + ref

    assert (is_channels_last_2d(out) if ndim == 4 else is_channels_last_3d(out)) == channels_last
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_nearest_upsample_2x_bit_exact(channels_last: bool) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    x = torch.randn(3, 64, 17, 23, device=device, dtype=torch.bfloat16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    ref = F.interpolate(x.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").to(x.dtype)
    out = wan_vae_nearest_upsample_2x(x)
    assert is_channels_last_2d(out) == channels_last
    assert torch.equal(ref, out)
    # Also verify via Upsample module
    up = Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
    with torch.inference_mode():
        assert torch.equal(ref, up(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_wan_vae_time_interleave_upsample_2x_bit_exact(channels_last: bool, with_bias: bool) -> None:
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, c, t, h, w = 2, 64, 3, 12, 16
    x = torch.randn(b, 2 * c, t, h, w, device=device, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    bias = torch.randn(2 * c, device=device, dtype=dtype) if with_bias else None

    with torch.inference_mode():
        out = wan_vae_time_interleave_upsample_2x(x, bias)
        xe = x + bias.view(1, 2 * c, 1, 1, 1) if bias is not None else x
        xe = xe.reshape(b, 2, c, t, h, w)
        xe = torch.stack((xe[:, 0], xe[:, 1]), dim=3).reshape(b, c, t * 2, h, w)
        x4 = xe.permute(0, 2, 1, 3, 4).reshape(b * t * 2, c, h, w)
        ref = F.interpolate(x4.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").to(dtype)

    assert is_channels_last_2d(out) == channels_last
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_zero_pad_2d_rb_bit_exact(channels_last: bool) -> None:
    torch.manual_seed(14)
    device = torch.device("cuda")
    x = torch.randn(3, 64, 15, 21, device=device, dtype=torch.bfloat16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        out = wan_vae_zero_pad_2d_rb(x)
        ref = F.pad(x, (0, 1, 0, 1))
    assert is_channels_last_2d(out) == channels_last
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("first_chunk", [False, True])
def test_wan_vae_dup_up_3d_bit_exact(channels_last: bool, first_chunk: bool) -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    dup = DupUp3D(64, 32, factor_t=2, factor_s=2)
    x = torch.randn(1, 64, 3, 12, 16, device=device, dtype=torch.bfloat16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    with torch.enable_grad():
        ref = dup(x, first_chunk=first_chunk)
    with torch.inference_mode():
        out = wan_vae_dup_up_3d(x, out_channels=32, factor_t=2, factor_s=2, first_chunk=first_chunk)
        mod_out = dup(x, first_chunk=first_chunk)
    assert is_channels_last_3d(out) == channels_last
    assert torch.equal(ref, out)
    assert torch.equal(ref, mod_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("h_4d", [False, True])
@pytest.mark.parametrize(
    "c_in,c_out,ft,fs,first_chunk",
    [
        (64, 64, 2, 2, False),
        (64, 64, 2, 2, True),
        (64, 32, 2, 2, False),
        (64, 32, 1, 2, False),
        (32, 32, 1, 1, False),
    ],
)
def test_wan_vae_dup_up_3d_add_bit_exact(
    channels_last: bool,
    h_4d: bool,
    c_in: int,
    c_out: int,
    ft: int,
    fs: int,
    first_chunk: bool,
) -> None:
    torch.manual_seed(15)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, t_in, h_in, w_in = 1, 2, 8, 10
    t_out = t_in * ft - ((ft - 1) if first_chunk else 0)
    h_out, w_out = h_in * fs, w_in * fs

    x = torch.randn(b, c_in, t_in, h_in, w_in, device=device, dtype=dtype)
    h5 = torch.randn(b, c_out, t_out, h_out, w_out, device=device, dtype=dtype)
    hb = torch.randn(c_out, device=device, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
        h5 = h5.to(memory_format=torch.channels_last_3d)
    if h_4d:
        h = (
            h5.permute(0, 2, 3, 4, 1).reshape(b * t_out, h_out, w_out, c_out).permute(0, 3, 1, 2)
            if channels_last
            else h5.permute(0, 2, 1, 3, 4).reshape(b * t_out, c_out, h_out, w_out).contiguous()
        )
    else:
        h = h5

    dup = DupUp3D(c_in, c_out, factor_t=ft, factor_s=fs)
    with torch.enable_grad():
        ref = (h5 + hb.view(1, c_out, 1, 1, 1)) + dup(x, first_chunk=first_chunk)
    with torch.inference_mode():
        out = wan_vae_dup_up_3d_add(h, x, hb, out_channels=c_out, factor_t=ft, factor_s=fs, first_chunk=first_chunk)

    assert is_channels_last_3d(out) == channels_last
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("h_4d", [False, True])
@pytest.mark.parametrize(
    "c_in,c_out,t_in,ft,fs",
    [
        (32, 32, 4, 1, 2),
        (32, 64, 4, 2, 2),
        (32, 64, 1, 2, 2),  # tests pad_t = 1
        (64, 32, 4, 1, 2),  # tests general path group_size = 8
    ],
)
def test_wan_vae_avg_down_3d_and_add_parity(
    channels_last: bool,
    h_4d: bool,
    c_in: int,
    c_out: int,
    t_in: int,
    ft: int,
    fs: int,
) -> None:
    torch.manual_seed(16)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, h_in, w_in = 1, 16, 20
    pad_t = (ft - t_in % ft) % ft
    t_out = (t_in + pad_t) // ft
    h_out, w_out = h_in // fs, w_in // fs

    x = torch.randn(b, c_in, t_in, h_in, w_in, device=device, dtype=dtype)
    h5 = torch.randn(b, c_out, t_out, h_out, w_out, device=device, dtype=dtype)
    hb = torch.randn(c_out, device=device, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
        h5 = h5.to(memory_format=torch.channels_last_3d)
    if h_4d:
        h = (
            h5.permute(0, 2, 3, 4, 1).reshape(b * t_out, h_out, w_out, c_out).permute(0, 3, 1, 2)
            if channels_last
            else h5.permute(0, 2, 1, 3, 4).reshape(b * t_out, c_out, h_out, w_out).contiguous()
        )
    else:
        h = h5

    avg_mod = AvgDown3D(c_in, c_out, factor_t=ft, factor_s=fs)
    with torch.enable_grad():
        ref_sc = avg_mod(x)
        ref_add = (h5 + hb.view(1, c_out, 1, 1, 1)) + ref_sc
    with torch.inference_mode():
        out_sc = wan_vae_avg_down_3d(x, out_channels=c_out, factor_t=ft, factor_s=fs)
        out_add = wan_vae_avg_down_3d_add(h, x, hb, out_channels=c_out, factor_t=ft, factor_s=fs)

    assert is_channels_last_3d(out_sc) == channels_last
    assert is_channels_last_3d(out_add) == channels_last
    assert F.cosine_similarity(ref_sc.float().flatten(), out_sc.float().flatten(), dim=0).item() >= 0.9999
    assert F.cosine_similarity(ref_add.float().flatten(), out_add.float().flatten(), dim=0).item() >= 0.9999
    assert (ref_add.float() - out_add.float()).abs().max().item() <= 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("in_cl3d", [False, True])
@pytest.mark.parametrize("out_cl3d", [None, False, True])
def test_wan_vae_patchify_unpatchify_2x_bit_exact(in_cl3d: bool, out_cl3d: bool | None) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(1, 3, 5, 24, 32, device=device, dtype=dtype)
    if in_cl3d:
        x = x.to(memory_format=torch.channels_last_3d)
    with torch.enable_grad():
        ref_p = patchify(x, 2)
        ref_u = unpatchify(ref_p, 2)
    with torch.inference_mode():
        p = wan_vae_patchify_2x(x, output_cl3d=out_cl3d)
        expected_cl = in_cl3d if out_cl3d is None else out_cl3d
        assert is_channels_last_3d(p) == expected_cl
        assert torch.equal(p, ref_p)

        pb = torch.randn(12, device=device, dtype=dtype)
        u = wan_vae_unpatchify_2x(p, output_cl3d=in_cl3d)
        assert is_channels_last_3d(u) == in_cl3d
        assert torch.equal(u, ref_u)

        u_pb = wan_vae_unpatchify_2x(p, pre_bias=pb, output_cl3d=in_cl3d)
        with torch.enable_grad():
            ref_u_pb = unpatchify(ref_p + pb.view(1, 12, 1, 1, 1), 2)
        assert torch.equal(u_pb, ref_u_pb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_residual_block_streaming_parity_and_hooks(channels_last: bool) -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = ResidualBlock(32, 64).to(device=device, dtype=dtype).eval()
    chunks = [torch.randn(1, 32, t, 16, 20, device=device, dtype=dtype) for t in (1, 2, 2)]
    if channels_last:
        chunks = [c.to(memory_format=torch.channels_last_3d) for c in chunks]

    # Reference with hooks attached (bypasses super-fusion safely)
    hook_calls = 0

    def _noop_hook(_m: nn.Module, _inp: tuple, _out: torch.Tensor) -> None:
        nonlocal hook_calls
        hook_calls += 1

    h = block.residual[2].register_forward_hook(_noop_hook)
    ref_cache: list[torch.Tensor | None] = [None, None]
    ref_outs = []
    with torch.inference_mode():
        for c in chunks:
            y, ref_cache, _ = block(c, feat_cache=ref_cache, feat_idx=[0])
            ref_outs.append(y)
    h.remove()
    assert hook_calls == len(chunks)

    # Super-fused path (no hooks)
    opt_cache: list[torch.Tensor | None] = [None, None]
    opt_outs = []
    with torch.inference_mode():
        for c in chunks:
            y, opt_cache, _ = block(c, feat_cache=opt_cache, feat_idx=[0])
            opt_outs.append(y)

    for r, o in zip(ref_outs, opt_outs, strict=True):
        assert is_channels_last_3d(o) == channels_last
        cos = F.cosine_similarity(r.float().flatten(), o.float().flatten(), dim=0).item()
        assert cos >= 0.9998


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_blocks_layout_preservation_and_streaming_parity(channels_last: bool) -> None:
    """Verify Resample, Resample38, AttentionBlock, Down_ResidualBlock, Up_ResidualBlock, VideoVAE_, VideoVAE38_."""
    torch.manual_seed(21)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    def _fmt(t: torch.Tensor) -> torch.Tensor:
        return t.to(memory_format=torch.channels_last_3d) if channels_last else t

    # 1. AttentionBlock
    attn = AttentionBlock(32).to(device=device, dtype=dtype).eval()
    nn.init.normal_(attn.proj.weight, std=0.05)
    xa = _fmt(torch.randn(1, 32, 2, 8, 8, device=device, dtype=dtype))
    with torch.enable_grad():
        ref_a = attn(xa).detach()
    with torch.inference_mode():
        opt_a = attn(xa)
    assert is_channels_last_3d(opt_a) == channels_last
    assert F.cosine_similarity(ref_a.float().flatten(), opt_a.float().flatten(), dim=0).item() >= 0.9998

    # 2. Resample all 4 modes
    for mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
        mod = Resample(32, mode).to(device=device, dtype=dtype).eval()
        ts = (1, 1, 2) if "upsample" in mode else (1, 4, 4)
        chunks = [_fmt(torch.randn(1, 32, t, 16, 16, device=device, dtype=dtype)) for t in ts]
        rc, oc = [None], [None]
        for c in chunks:
            with torch.enable_grad():
                ry, rc, _ = mod(c, feat_cache=rc, feat_idx=[0])
                ry = ry.detach()
            with torch.inference_mode():
                oy, oc, _ = mod(c, feat_cache=oc, feat_idx=[0])
            assert is_channels_last_3d(oy) == channels_last
            assert F.cosine_similarity(ry.float().flatten(), oy.float().flatten(), dim=0).item() >= 0.9998

    # 3. Down_ResidualBlock & Up_ResidualBlock (Wan2.2 VAE38)
    down = (
        Down_ResidualBlock(32, 64, dropout=0.0, mult=1, temperal_downsample=True, down_flag=True)
        .to(device=device, dtype=dtype)
        .eval()
    )
    d_chunks = [_fmt(torch.randn(1, 32, t, 16, 16, device=device, dtype=dtype)) for t in (1, 4, 4)]
    rc_d, oc_d = [None] * 4, [None] * 4
    for c in d_chunks:
        with torch.enable_grad():
            ry, rc_d, _ = down(c, feat_cache=rc_d, feat_idx=[0])
            ry = ry.detach()
        with torch.inference_mode():
            oy, oc_d, _ = down(c, feat_cache=oc_d, feat_idx=[0])
        assert is_channels_last_3d(oy) == channels_last
        assert F.cosine_similarity(ry.float().flatten(), oy.float().flatten(), dim=0).item() >= 0.9995

    up = (
        Up_ResidualBlock(64, 32, dropout=0.0, mult=1, temperal_upsample=True, up_flag=True)
        .to(device=device, dtype=dtype)
        .eval()
    )
    u_chunks = [_fmt(torch.randn(1, 64, 1, 8, 8, device=device, dtype=dtype)) for _ in range(3)]
    rc_u, oc_u = [None] * 4, [None] * 4
    for i, c in enumerate(u_chunks):
        with torch.enable_grad():
            ry, rc_u, _ = up(c, feat_cache=rc_u, feat_idx=[0], first_chunk=(i == 0))
            ry = ry.detach()
        with torch.inference_mode():
            oy, oc_u, _ = up(c, feat_cache=oc_u, feat_idx=[0], first_chunk=(i == 0))
        assert is_channels_last_3d(oy) == channels_last
        assert F.cosine_similarity(ry.float().flatten(), oy.float().flatten(), dim=0).item() >= 0.9995

    # 4. Tiny E2E VideoVAE_ & VideoVAE38_ encode/decode parity and output layout contract
    vae21 = VideoVAE_(dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1).to(device=device, dtype=dtype).eval()
    scale21 = [torch.zeros(8, device=device, dtype=dtype), torch.ones(8, device=device, dtype=dtype)]
    vid21 = _fmt(torch.randn(1, 3, 5, 16, 16, device=device, dtype=dtype))
    lat21 = _fmt(torch.randn(1, 8, 2, 2, 2, device=device, dtype=dtype))
    with torch.enable_grad():
        ref_enc21 = vae21.encode(vid21, scale21).detach()
        ref_dec21 = vae21.decode(lat21, scale21).detach()
    with torch.inference_mode():
        opt_enc21 = vae21.encode(vid21, scale21)
        opt_dec21 = vae21.decode(lat21, scale21)
    assert is_channels_last_3d(opt_enc21) == channels_last
    assert is_channels_last_3d(opt_dec21) == channels_last
    assert F.cosine_similarity(ref_enc21.float().flatten(), opt_enc21.float().flatten(), dim=0).item() >= 0.999
    assert F.cosine_similarity(ref_dec21.float().flatten(), opt_dec21.float().flatten(), dim=0).item() >= 0.999

    vae38 = (
        VideoVAE38_(dim=16, z_dim=8, dec_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
        .to(device=device, dtype=dtype)
        .eval()
    )
    scale38 = [torch.zeros(8, device=device, dtype=dtype), torch.ones(8, device=device, dtype=dtype)]
    vid38 = _fmt(torch.randn(1, 3, 5, 32, 32, device=device, dtype=dtype))
    lat38 = _fmt(torch.randn(1, 8, 2, 2, 2, device=device, dtype=dtype))
    with torch.enable_grad():
        ref_enc38 = vae38.encode(vid38, scale38).detach()
        ref_dec38 = vae38.decode(lat38, scale38).detach()
    with torch.inference_mode():
        opt_enc38 = vae38.encode(vid38, scale38)
        opt_dec38 = vae38.decode(lat38, scale38)
    assert is_channels_last_3d(opt_enc38) == channels_last
    assert is_channels_last_3d(opt_dec38) == channels_last
    assert F.cosine_similarity(ref_enc38.float().flatten(), opt_enc38.float().flatten(), dim=0).item() >= 0.999
    assert F.cosine_similarity(ref_dec38.float().flatten(), opt_dec38.float().flatten(), dim=0).item() >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_custom_ops_torch_compile_fullgraph(channels_last: bool) -> None:
    """Verify every custom op's register_fake strides match under torch.compile(fullgraph=True)."""
    torch.manual_seed(31)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    def _f5(t: torch.Tensor) -> torch.Tensor:
        return t.to(memory_format=torch.channels_last_3d) if channels_last else t

    def _f4(t: torch.Tensor) -> torch.Tensor:
        return t.to(memory_format=torch.channels_last) if channels_last else t

    x4 = _f4(torch.randn(2, 32, 8, 10, device=device, dtype=dtype))
    x5 = _f5(torch.randn(1, 32, 2, 8, 10, device=device, dtype=dtype))
    x5_2c = _f5(torch.randn(1, 64, 2, 8, 10, device=device, dtype=dtype))
    g = torch.randn(32, 1, 1, 1, device=device, dtype=dtype)
    b32 = torch.randn(32, device=device, dtype=dtype)
    b64 = torch.randn(64, device=device, dtype=dtype)
    h_dup = _f5(torch.randn(1, 32, 4, 16, 20, device=device, dtype=dtype))
    h_avg = _f5(torch.randn(1, 64, 1, 4, 5, device=device, dtype=dtype))
    rgb = _f5(torch.randn(1, 3, 2, 8, 10, device=device, dtype=dtype))

    def _run_all(a4, a5, a5_2c, gamma, bias32, bias64, hd, ha, img):
        u2 = wan_vae_nearest_upsample_2x(a4)
        zp = wan_vae_zero_pad_2d_rb(a4)
        rn = wan_vae_rms_norm(a5, gamma, None, scale=math.sqrt(32.0), apply_silu=True, pre_bias=bias32)
        rp, rc = wan_vae_rms_norm_silu_causal_pad(
            a5,
            gamma,
            None,
            cache_in=None,
            pad_w=1,
            pad_h=1,
            pad_t_left=2,
            scale=math.sqrt(32.0),
            write_cache=True,
            cache_t=2,
            pre_bias=bias32,
        )
        cp, cc = wan_vae_causal_pad_cache(
            img, cache_in=None, pad_w=1, pad_h=1, pad_t_left=2, write_cache=True, cache_t=2, align_c=32
        )
        ra = wan_vae_residual_bias_add(a5, a5, bias32, bias32)
        ti = wan_vae_time_interleave_upsample_2x(a5_2c, bias64)
        du = wan_vae_dup_up_3d_add(hd, a5_2c, bias32, out_channels=32, factor_t=2, factor_s=2, first_chunk=False)
        ad = wan_vae_avg_down_3d_add(ha, a5, bias64, out_channels=64, factor_t=2, factor_s=2)
        pt = wan_vae_patchify_2x(img)
        up = wan_vae_unpatchify_2x(pt)
        return u2, zp, rn, rp, rc, cp, cc, ra, ti, du, ad, pt, up

    compiled = torch.compile(_run_all, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    with torch.inference_mode():
        eager_outs = _run_all(x4, x5, x5_2c, g, b32, b64, h_dup, h_avg, rgb)
        comp_outs = compiled(x4, x5, x5_2c, g, b32, b64, h_dup, h_avg, rgb)

    for e, c in zip(eager_outs, comp_outs, strict=True):
        assert e.shape == c.shape
        assert e.stride() == c.stride()
        assert torch.allclose(e.float(), c.float(), atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_rms_norm_int64_indexing_above_2_pow_31(channels_last: bool) -> None:
    """Verify >2^31-element tensors (e.g. 480p 65f 128ch) never wrap int32 offsets."""
    device = torch.device("cuda")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 20 * 1024**3:
        pytest.skip()
    x = torch.empty((1, 128, 65, 480, 832), device=device, dtype=torch.bfloat16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    ch_vals = torch.linspace(-1.5, 1.5, 128, device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
    x.copy_(ch_vals.expand_as(x))
    gamma = torch.linspace(0.5, 1.5, 128, device=device, dtype=torch.bfloat16).view(128, 1, 1, 1)

    with torch.inference_mode():
        out = wan_vae_rms_norm(x, gamma, None, scale=math.sqrt(128.0), apply_silu=True)
        ref_1 = F.silu(F.normalize(ch_vals, dim=1) * math.sqrt(128.0) * gamma.view(1, 128, 1, 1, 1)).squeeze()
        for t, h, w in ((0, 0, 0), (32, 240, 416), (64, 479, 831)):
            got = out[0, :, t, h, w]
            diff = (got.float() - ref_1.float()).abs().max().item()
            assert diff <= 0.05, f"int64 mismatch at ({t},{h},{w}): max_diff={diff}"
    del x, out
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_residual_bias_add_int64_indexing_above_2_pow_31(channels_last: bool) -> None:
    """Verify >2^31-element 5D residual_bias_add never wraps int32 offsets."""
    device = torch.device("cuda")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 25 * 1024**3:
        pytest.skip()
    # 1 * 128 * 65 * 480 * 832 = 3,322,675,200 elements (> 2^31 - 1)
    h = torch.empty((1, 128, 65, 480, 832), device=device, dtype=torch.bfloat16)
    x = torch.empty((1, 128, 65, 480, 832), device=device, dtype=torch.bfloat16)
    if channels_last:
        h = h.to(memory_format=torch.channels_last_3d)
        x = x.to(memory_format=torch.channels_last_3d)
    hv = torch.linspace(-1.0, 1.0, 128, device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
    xv = torch.linspace(0.25, -0.75, 128, device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
    hb = torch.linspace(-0.5, 0.5, 128, device=device, dtype=torch.bfloat16)
    xb = torch.linspace(0.1, -0.1, 128, device=device, dtype=torch.bfloat16)
    h.copy_(hv.expand_as(h))
    x.copy_(xv.expand_as(x))

    with torch.inference_mode():
        out = wan_vae_residual_bias_add(x, h, hb, xb)
        ref_1 = (xv.squeeze() + xb) + (hv.squeeze() + hb)
        for t, hh, ww in ((0, 0, 0), (32, 240, 416), (64, 479, 831)):
            got = out[0, :, t, hh, ww]
            assert torch.equal(got, ref_1), f"int64 mismatch at ({t},{hh},{ww})"
    del h, x, out
    torch.cuda.empty_cache()
