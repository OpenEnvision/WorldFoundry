# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and parity tests for SANA-Video fused Triton ``LiteLAReLURope``."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.sana.normalization import RMSNorm
from worldfoundry.base_models.diffusion_model.models.networks.sana.sana_blocks import LiteLAReLURope
from worldfoundry.base_models.diffusion_model.models.networks.sana.triton_litela_rope import (
    can_use_sana_rms_norm,
    can_use_triton_litela_relu_rope,
    fused_sana_litela_relu_rope,
    fused_sana_rms_norm,
)


def test_cpu_eligibility_returns_false_without_importing_triton() -> None:
    qkv = torch.randn(1, 16, 3 * 64)
    cos = torch.ones(1, 16, 1, 16)
    sin = torch.zeros(1, 16, 1, 16)
    assert not can_use_triton_litela_relu_rope(qkv, cos, sin, num_heads=2, head_dim=32)
    assert not can_use_sana_rms_norm(torch.randn(2, 16, 64), torch.ones(64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("heads", "dim"), [(8, 32), (20, 112)])
@pytest.mark.parametrize("qk_norm", [True, False])
@pytest.mark.parametrize("use_fp32_attention", [True, False])
@pytest.mark.parametrize("rope_dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("seq_len", [127, 2048])
def test_sana_litela_triton_lossless_parity(
    heads: int,
    dim: int,
    qk_norm: bool,
    use_fp32_attention: bool,
    rope_dtype: torch.dtype,
    seq_len: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 2
    hidden = heads * dim

    layer = LiteLAReLURope(hidden, hidden, heads=heads, dim=dim, qk_norm=qk_norm).to(device=device, dtype=dtype).eval()
    layer.fp32_attention = use_fp32_attention
    if qk_norm:
        layer.q_norm.weight.data.normal_(1.0, 0.1)
        layer.k_norm.weight.data.normal_(1.0, 0.1)
        layer.q_norm._use_fused_rms_norm = True
        layer.k_norm._use_fused_rms_norm = True

    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    angles = torch.randn(1, seq_len, 1, dim // 2, device=device, dtype=rope_dtype)
    cos, sin = torch.cos(angles), torch.sin(angles)

    with torch.inference_mode():
        # Reference: disable fast triton path via _fast_litela_rope=False
        layer._fast_litela_rope = False
        ref = layer(x, rotary_emb=(cos, sin))

        # Fused Triton path
        layer._fast_litela_rope = True
        out = layer(x, rotary_emb=(cos, sin))

    diff = (ref.float() - out.float()).abs()
    cos_sim = F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()
    mse = diff.square().mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((ref.float().abs().max().item() ** 2 + 1e-12) / mse)
    assert cos_sim >= 0.9998, f"cosine similarity too low: {cos_sim}"
    assert psnr >= 45.0, f"PSNR too low: {psnr:.2f} dB (max_abs={diff.max().item()})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [512, 1536])
def test_sana_litela_triton_cudagraph_capture_safe(seq_len: int) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, n, h, d = 2, seq_len, 8, 32
    c = h * d
    qkv = torch.randn(b, n, 3 * c, device=device, dtype=dtype)
    angles = torch.randn(1, n, 1, d // 2, device=device, dtype=torch.float64)
    cos, sin = torch.cos(angles), torch.sin(angles)
    qw = torch.ones(c, device=device, dtype=dtype)
    kw = torch.ones(c, device=device, dtype=dtype)

    with torch.inference_mode():
        eager = fused_sana_litela_relu_rope(
            qkv, cos, sin, num_heads=h, head_dim=d, q_weight=qw, k_weight=kw, use_fp32_attention=True
        )
        # Warmup outside graph
        _ = fused_sana_litela_relu_rope(
            qkv, cos, sin, num_heads=h, head_dim=d, q_weight=qw, k_weight=kw, use_fp32_attention=True
        )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = fused_sana_litela_relu_rope(
                qkv, cos, sin, num_heads=h, head_dim=d, q_weight=qw, k_weight=kw, use_fp32_attention=True
            )
        g.replay()
        torch.cuda.synchronize()

    assert torch.equal(eager, captured)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("features", [112, 1152, 2240])
def test_sana_rms_norm_triton_exact_parity_and_module(dtype: torch.dtype, features: int) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    x = torch.randn(2, 128, features, device=device, dtype=dtype)
    mod = RMSNorm(features, eps=1e-6).to(device=device, dtype=dtype).eval()
    mod.weight.data.normal_(1.0, 0.2)

    with torch.inference_mode():
        mod._use_fused_rms_norm = False
        ref = mod(x)
        mod._use_fused_rms_norm = True
        out = mod(x)
        direct = fused_sana_rms_norm(x, mod.weight, 1e-6)

    torch.testing.assert_close(out, direct, rtol=0, atol=0)
    diff = (ref.float() - out.float()).abs()
    mse = diff.square().mean().item()
    peak = max(ref.float().abs().max().item(), 1.0)
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((peak * peak) / mse)
    assert psnr >= 60.0, f"RMSNorm PSNR too low: {psnr:.2f} dB (max_abs={diff.max().item()})"
