# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and parity tests for the lossless SANA ``GLUMBConv`` Triton kernel."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.sana.basic_modules import GLUMBConv, GLUMBConvTemp
from worldfoundry.base_models.diffusion_model.models.networks.sana.triton_glumbconv import (
    can_use_sana_silu_dwconv3x3_glu,
    fused_sana_silu_dwconv3x3_glu_nhwc,
)


def _reference_silu_dwconv_glu(h_nhwc: torch.Tensor, dw_conv: torch.nn.Conv2d) -> torch.Tensor:
    h_act = F.silu(h_nhwc)
    h_dw = dw_conv(h_act.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    a, g = h_dw.chunk(2, dim=-1)
    return a * F.silu(g)


def test_cpu_eligibility_and_lazy_import() -> None:
    x = torch.randn(2, 4, 5, 16)
    dw = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
    assert not can_use_sana_silu_dwconv3x3_glu(x, dw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("bt", "height", "width", "c_mid"),
    [
        (1, 1, 1, 8),
        (2, 3, 5, 32),
        (3, 7, 13, 96),
        (4, 15, 26, 256),
        (2, 30, 52, 640),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("use_bias", [True, False])
def test_fused_sana_silu_dwconv3x3_glu_parity(
    bt: int,
    height: int,
    width: int,
    c_mid: int,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    torch.manual_seed(0)
    two_c = 2 * c_mid
    h = torch.randn(bt, height, width, two_c, device="cuda", dtype=dtype)
    dw = torch.nn.Conv2d(two_c, two_c, 3, padding=1, groups=two_c, bias=use_bias, device="cuda", dtype=dtype)
    dw.weight.data = dw.weight.data.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        assert can_use_sana_silu_dwconv3x3_glu(h, dw)
        out_fused = fused_sana_silu_dwconv3x3_glu_nhwc(h, dw.weight, dw.bias)
        out_ref = _reference_silu_dwconv_glu(h, dw)

    assert out_fused.shape == out_ref.shape
    assert out_fused.dtype == out_ref.dtype
    diff = (out_fused.float() - out_ref.float()).abs()
    mse = (diff**2).mean().item()
    peak = max(out_ref.float().abs().max().item(), 1e-6)
    psnr = 10.0 * math.log10((peak * peak) / max(mse, 1e-12))
    if dtype == torch.float32:
        assert diff.max().item() < 1e-4
    elif dtype == torch.float16:
        assert psnr > 60.0
    else:
        # BF16 has 7-bit mantissa (~42 dB per single ULP); tiny 8-element tensors can hit 1 ULP.
        min_psnr = 50.0 if out_ref.numel() < 64 else 60.0
        assert psnr > min_psnr
        torch.testing.assert_close(out_fused, out_ref, rtol=1.6e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eligibility_guards_and_empty_tensor() -> None:
    h = torch.randn(2, 4, 6, 32, device="cuda", dtype=torch.bfloat16)
    dw_ok = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, device="cuda", dtype=torch.bfloat16)

    # Grad enabled disables custom op.
    with torch.enable_grad():
        assert not can_use_sana_silu_dwconv3x3_glu(h, dw_ok)

    with torch.inference_mode():
        assert can_use_sana_silu_dwconv3x3_glu(h, dw_ok)

        # Kernel size != 3
        dw_k5 = torch.nn.Conv2d(32, 32, 5, padding=2, groups=32, device="cuda", dtype=torch.bfloat16)
        assert not can_use_sana_silu_dwconv3x3_glu(h, dw_k5)

        # Non-depthwise groups
        dw_grp = torch.nn.Conv2d(32, 32, 3, padding=1, groups=1, device="cuda", dtype=torch.bfloat16)
        assert not can_use_sana_silu_dwconv3x3_glu(h, dw_grp)

        # Forward hook present
        handle = dw_ok.register_forward_hook(lambda m, i, o: o)
        try:
            assert not can_use_sana_silu_dwconv3x3_glu(h, dw_ok)
        finally:
            handle.remove()

        # Empty batch handled cleanly
        h_empty = torch.empty(0, 4, 6, 32, device="cuda", dtype=torch.bfloat16)
        out_empty = fused_sana_silu_dwconv3x3_glu_nhwc(h_empty, dw_ok.weight, dw_ok.bias)
        assert out_empty.shape == (0, 4, 6, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_glumbconv_and_glumbconvtemp_end_to_end_parity() -> None:
    torch.manual_seed(7)
    b, t, h, w, c, c_mid = 2, 3, 6, 10, 64, 160
    x = torch.randn(b, t * h * w, c, device="cuda", dtype=torch.bfloat16)

    for cls in (GLUMBConv, GLUMBConvTemp):
        mod = cls(in_features=c, hidden_features=c_mid, use_bias=(True, True, False), act=("silu", "silu", None))
        if isinstance(mod, GLUMBConvTemp):
            mod.t_conv.weight.data.normal_(0.0, 0.05)
        mod = mod.to(device="cuda", dtype=torch.bfloat16).eval()

        with torch.inference_mode():
            mod._fused_glumbconv = False
            out_ref = mod(x, HW=(t, h, w))

            mod._fused_glumbconv = True
            assert mod.depth_conv.conv is not None
            mod.depth_conv.conv.weight.data = mod.depth_conv.conv.weight.data.to(memory_format=torch.channels_last)
            out_fused = mod(x, HW=(t, h, w))

        diff = (out_fused.float() - out_ref.float()).abs()
        cos = F.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0).item()
        assert cos > 0.99995, f"{cls.__name__} cos={cos}"
        assert diff.mean().item() < 2e-3

        # Verify forward hooks on t_conv bypass the fused temporal path cleanly.
        if isinstance(mod, GLUMBConvTemp):
            called: list[int] = []
            handle = mod.t_conv.register_forward_hook(lambda m, i, o: called.append(1) or o)
            try:
                with torch.inference_mode():
                    out_hooked = mod(x, HW=(t, h, w))
                assert called == [1]
                assert (
                    F.cosine_similarity(out_hooked.float().flatten(), out_ref.float().flatten(), dim=0).item() > 0.99995
                )
            finally:
                handle.remove()
