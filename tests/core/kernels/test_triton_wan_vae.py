# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and parity tests for WanVideoVAE / WanVideoVAE38 lossless Triton kernels."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    DupUp3D,
    ResidualBlock,
    RMS_norm,
    Upsample,
)
from worldfoundry.core.kernels.triton_wan_vae import (
    can_use_wan_vae_rms_norm,
    wan_vae_dup_up_3d,
    wan_vae_nearest_upsample_2x,
    wan_vae_rms_norm,
    wan_vae_rms_norm_silu_causal_pad,
)


def test_cpu_fallback_and_eligibility() -> None:
    x = torch.randn(1, 16, 3, 8, 8)
    assert not can_use_wan_vae_rms_norm(x)
    rms = RMS_norm(16, images=False).eval()
    with torch.inference_mode():
        y = rms(x)
    ref = F.normalize(x, dim=1) * rms.scale * rms.gamma
    assert torch.allclose(y, ref, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("apply_silu", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wan_vae_rms_norm_parity(
    channels_last: bool,
    with_bias: bool,
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

    with torch.inference_mode():
        ref = F.normalize(x, dim=1) * math.sqrt(96) * gamma
        if bias is not None:
            ref = ref + bias
        if apply_silu:
            ref = F.silu(ref)
        out = wan_vae_rms_norm(x, gamma, bias, scale=math.sqrt(96), apply_silu=apply_silu)

    assert out.is_contiguous(memory_format=torch.channels_last_3d) == channels_last
    tol = 0.08 if dtype == torch.bfloat16 else 1e-4
    diff = (ref.float() - out.float()).abs()
    assert diff.max().item() <= tol
    cos = F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()
    assert cos >= 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("t_in,has_prev", [(1, False), (1, True), (3, False), (3, True)])
def test_wan_vae_super_fused_causal_pad_parity(channels_last: bool, t_in: int, has_prev: bool) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(1, 64, t_in, 16, 20, device=device, dtype=dtype)
    cache_in = torch.randn(1, 64, 2, 16, 20, device=device, dtype=dtype) if has_prev else None
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
        if cache_in is not None:
            cache_in = cache_in.to(memory_format=torch.channels_last_3d)
    gamma = torch.randn(64, 1, 1, 1, device=device, dtype=dtype)

    with torch.inference_mode():
        y = F.silu(F.normalize(x, dim=1) * math.sqrt(64) * gamma)
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
            scale=math.sqrt(64),
            write_cache=True,
            cache_t=2,
        )

    assert out_pad.shape == ref_pad.shape
    assert out_cache.shape == ref_cache.shape
    assert F.cosine_similarity(ref_pad.float().flatten(), out_pad.float().flatten(), dim=0).item() >= 0.9999
    assert F.cosine_similarity(ref_cache.float().flatten(), out_cache.float().flatten(), dim=0).item() >= 0.9999


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
    assert torch.equal(ref, out)
    # Also verify via Upsample module
    up = Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")
    with torch.inference_mode():
        assert torch.equal(ref, up(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("first_chunk", [False, True])
def test_wan_vae_dup_up_3d_bit_exact(first_chunk: bool) -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    dup = DupUp3D(64, 32, factor_t=2, factor_s=2)
    x = torch.randn(1, 64, 3, 12, 16, device=device, dtype=torch.bfloat16)
    # Exact eager reference (grad enabled forces PyTorch reference branch)
    with torch.enable_grad():
        ref = dup(x, first_chunk=first_chunk)
    with torch.inference_mode():
        out = wan_vae_dup_up_3d(x, out_channels=32, factor_t=2, factor_s=2, first_chunk=first_chunk)
        mod_out = dup(x, first_chunk=first_chunk)
    assert torch.equal(ref, out)
    assert torch.equal(ref, mod_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wan_vae_residual_block_streaming_parity_and_hooks() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = ResidualBlock(32, 64).to(device=device, dtype=dtype).eval()
    chunks = [torch.randn(1, 32, t, 16, 20, device=device, dtype=dtype) for t in (1, 2, 2)]

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
        cos = F.cosine_similarity(r.float().flatten(), o.float().flatten(), dim=0).item()
        assert cos >= 0.9998


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels_last", [False, True])
def test_wan_vae_rms_norm_int64_indexing_above_2_pow_31(channels_last: bool) -> None:
    """Verify >2^31-element tensors (e.g. 480p 65f 128ch) never wrap int32 offsets."""
    device = torch.device("cuda")
    # 1 * 128 * 65 * 480 * 832 = 3,322,675,200 elements (> 2^31 - 1)
    # Use small constant-pattern fill + spot check near tail channels (c >= 83)
    # where int32 c * spatial_size would overflow negative.
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 20 * 1024**3:
        pytest.skip()
    x = torch.empty((1, 128, 65, 480, 832), device=device, dtype=torch.bfloat16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last_3d)
    # Fill deterministic channel-varying values without allocating extra 6.6 GB temporaries.
    ch_vals = torch.linspace(-1.5, 1.5, 128, device=device, dtype=torch.bfloat16).view(1, 128, 1, 1, 1)
    x.copy_(ch_vals.expand_as(x))
    gamma = torch.linspace(0.5, 1.5, 128, device=device, dtype=torch.bfloat16).view(128, 1, 1, 1)

    with torch.inference_mode():
        out = wan_vae_rms_norm(x, gamma, None, scale=math.sqrt(128.0), apply_silu=True)
        # Exact reference computed on a single voxel (all spatial voxels are identical).
        ref_1 = F.silu(F.normalize(ch_vals, dim=1) * math.sqrt(128.0) * gamma.view(1, 128, 1, 1, 1)).squeeze()
        # Verify both early and tail voxels across all 128 channels (especially c in 83..127).
        for t, h, w in ((0, 0, 0), (32, 240, 416), (64, 479, 831)):
            got = out[0, :, t, h, w]
            diff = (got.float() - ref_1.float()).abs().max().item()
            assert diff <= 0.05, f"int64 mismatch at ({t},{h},{w}): max_diff={diff}"
    del x, out
    torch.cuda.empty_cache()
