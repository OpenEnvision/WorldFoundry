# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact parity and layout-preservation tests for ``triton_pad_3d``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.kernels.triton_pad_3d import can_use_triton_pad_3d, triton_pad_3d


def test_cpu_eligibility_and_fallback() -> None:
    x = torch.randn(1, 4, 3, 5, 6)
    pad = (1, 1, 1, 1, 2, 0)
    assert not can_use_triton_pad_3d(x, pad, spatial_mode="replicate")
    got = triton_pad_3d(x, pad, spatial_mode="replicate")
    ref = F.pad(x, pad, mode="replicate")
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["ncdhw", "channels_last_3d"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("spatial_mode", "temporal_mode"),
    [
        ("constant", "constant"),
        ("replicate", "replicate"),
        ("reflect", "reflect"),
        ("reflect", "constant"),
        ("reflect", "replicate"),
        ("replicate", "constant"),
    ],
)
def test_triton_pad_3d_bit_exact_and_preserves_layout(
    layout: str,
    dtype: torch.dtype,
    spatial_mode: str,
    temporal_mode: str,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 96, 5, 19, 23, device="cuda", dtype=dtype)
    if layout == "channels_last_3d":
        x = x.to(memory_format=torch.channels_last_3d)
    pad = (1, 2, 2, 1, 2, 0)

    with torch.inference_mode():
        assert can_use_triton_pad_3d(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)
        got = triton_pad_3d(x, pad, spatial_mode=spatial_mode, temporal_mode=temporal_mode)
        ref = F.pad(x.contiguous(), (pad[0], pad[1], pad[2], pad[3], 0, 0), mode=spatial_mode)
        ref = F.pad(ref, (0, 0, 0, 0, pad[4], pad[5]), mode=temporal_mode)

    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got.contiguous(), ref.contiguous())
    if layout == "channels_last_3d":
        assert got.is_contiguous(memory_format=torch.channels_last_3d)
    else:
        assert got.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pad_3d_cudagraph_and_autograd_fallback() -> None:
    x = torch.randn(1, 32, 4, 8, 10, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pad = (1, 1, 1, 1, 2, 0)
    # Autograd active -> falls back cleanly with gradients intact
    assert not can_use_triton_pad_3d(x, pad, spatial_mode="replicate")
    y = triton_pad_3d(x, pad, spatial_mode="replicate")
    y.sum().backward()
    assert x.grad is not None

    # Inference + CUDA Graph capture safe
    x_cl = x.detach().to(memory_format=torch.channels_last_3d)
    with torch.inference_mode():
        eager = triton_pad_3d(x_cl, pad, spatial_mode="replicate")
        _ = triton_pad_3d(x_cl, pad, spatial_mode="replicate")
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = triton_pad_3d(x_cl, pad, spatial_mode="replicate")
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(eager, captured)
    assert captured.is_contiguous(memory_format=torch.channels_last_3d)
