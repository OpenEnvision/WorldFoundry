# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU and GPU parity tests for Cosmos3 AVAE audio decoder Triton kernels."""

from __future__ import annotations

import math

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos3.audio import (
    Cosmos3AVAEAudioDecoder,
    Snake1d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos3.triton_audio_ops import (
    can_use_cosmos3_snake1d,
    fused_cosmos3_snake1d,
)


def _snake1d_ref(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    a = torch.exp(alpha)
    b = torch.exp(beta)
    return x + (b + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


def test_cpu_eligibility_and_lazy_import() -> None:
    x = torch.randn(1, 32, 128)
    alpha = torch.zeros(1, 32, 1)
    beta = torch.zeros(1, 32, 1)
    assert not can_use_cosmos3_snake1d(x, alpha, beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("b", "c", "length"), [(1, 32, 17), (2, 320, 4097), (1, 320, 65536)])
def test_cosmos3_snake1d_parity_and_guards(dtype: torch.dtype, b: int, c: int, length: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(b, c, length, device="cuda", dtype=dtype)
    mod = Snake1d(c).to(device="cuda", dtype=dtype).eval()
    mod.alpha.data.normal_(0.0, 0.5)
    mod.beta.data.normal_(0.0, 0.5)

    with torch.inference_mode():
        assert can_use_cosmos3_snake1d(x, mod.alpha, mod.beta)
        out = mod(x)
        direct = fused_cosmos3_snake1d(x, mod.alpha, mod.beta)
        ref = _snake1d_ref(x, mod.alpha, mod.beta)

    torch.testing.assert_close(out, direct, rtol=0, atol=0)
    diff = (out.float() - ref.float()).abs()
    mse = diff.square().mean().item()
    peak = max(ref.float().abs().max().item(), 1.0)
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((peak * peak) / mse)
    if dtype == torch.float32:
        assert diff.max().item() < 1e-5
    elif dtype == torch.float16:
        assert psnr > 65.0
    else:
        assert psnr > 55.0

    # Autograd must bypass the inference-only custom op.
    with torch.enable_grad():
        assert not can_use_cosmos3_snake1d(x, mod.alpha, mod.beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cosmos3_snake1d_cudagraph_and_decoder_smoke() -> None:
    torch.manual_seed(4)
    x = torch.randn(1, 64, 2048, device="cuda", dtype=torch.float32)
    alpha = torch.randn(1, 64, 1, device="cuda", dtype=torch.float32) * 0.3
    beta = torch.randn(1, 64, 1, device="cuda", dtype=torch.float32) * 0.3

    with torch.inference_mode():
        eager = fused_cosmos3_snake1d(x, alpha, beta)
        _ = fused_cosmos3_snake1d(x, alpha, beta)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = fused_cosmos3_snake1d(x, alpha, beta)
        g.replay()
        torch.cuda.synchronize()
    assert torch.equal(eager, captured)

    # Small end-to-end decoder parity vs pure eager training-mode bypass.
    dec = Cosmos3AVAEAudioDecoder(
        vocoder_input_dim=8,
        dec_dim=16,
        dec_c_mults=(1, 2),
        dec_strides=(2, 4),
        dec_out_channels=2,
    ).cuda()
    latents = torch.randn(1, 8, 12, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        dec.eval()
        out_fused = dec.decode(latents)
        dec.train()
        out_ref = dec.decode(latents)
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-4, atol=1e-4)
