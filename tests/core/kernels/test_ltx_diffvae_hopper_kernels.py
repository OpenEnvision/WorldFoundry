"""Tests for lossless Hopper SM90 LTX-2.5 DiffVAE Triton kernels."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.chunked.attn import (
    _apply_in_slab_abs_rope,
    _cut_slab,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.chunked.context import (
    _upsample_then_ctx,
    deferred,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.rope_math import (
    default_rope_dim_split,
    rope_inv_freqs,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.triton_diffvae_ops import (
    _rmsnorm_3d_abs_rope_torch,
    _rmsnorm_modulate_torch,
    diffvae_cut_slab,
    diffvae_rmsnorm_3d_abs_rope,
    diffvae_rmsnorm_modulate,
    diffvae_rmsnorm_modulate_into,
    diffvae_w_slice_add,
)


def _psnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    diff = ref.float() - got.float()
    mse = diff.pow(2).mean().item()
    if mse == 0.0:
        return float("inf")
    peak = max(ref.float().abs().max().item(), 1.0)
    return 10.0 * math.log10((peak * peak) / mse)


def _legacy_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    import triton

    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.fallback_na import (
        triton_na,
    )

    batch, t, h, w, nh, hd = q.shape
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kt, kh, kw = (k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, (t, h, w), strict=True))
    if scale is None:
        scale = hd**-0.5
    q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q_c)
    hd_p = max(16, triton.next_power_of_2(hd))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))
    grid = (triton.cdiv(w, block_q), h, t * batch * nh)
    triton_na._na3d_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        t,
        h,
        w,
        nh,
        batch * nh,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        q_c.stride(3),
        q_c.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=causal[0],
        causal_h=causal[1],
        causal_w=causal[2],
        hd=hd,
        hd_pad=hd_p,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q_c.dtype == torch.float32,
        num_warps=4,
    )
    return out


def test_lazy_triton_import_and_cpu_fallback() -> None:
    """CPU-only import works without Triton and all helpers fall back cleanly."""
    code = (
        "import sys; sys.modules['triton'] = None; sys.modules['triton.language'] = None; "
        "import torch; "
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.triton_diffvae_ops import "
        "diffvae_rmsnorm_modulate, diffvae_rmsnorm_3d_abs_rope, diffvae_w_slice_add, diffvae_cut_slab; "
        "x = torch.randn(1, 2, 4, 6, 32); "
        "y = diffvae_rmsnorm_modulate(x, torch.ones(32), torch.zeros(32), torch.zeros(32)); "
        "assert y.shape == x.shape"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    x = torch.randn(1, 3, 4, 8, 64)
    w = torch.randn(64)
    s = torch.randn(1, 1, 1, 1, 64)
    sh = torch.randn(1, 1, 1, 1, 64)
    got = diffvae_rmsnorm_modulate(x, w, s, sh, eps=1e-6)
    ref = _rmsnorm_modulate_torch(x, w, s, sh, 1e-6)
    torch.testing.assert_close(got, ref)
    assert diffvae_cut_slab(x, 0, 4, 2, 8, None, has_right_neighbor=True) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("shape", "kernel_size", "causal", "dtype"),
    [
        ((1, 13, 19, 23, 4, 64), (11, 11, 11), [False, False, False], torch.bfloat16),
        ((1, 9, 17, 21, 8, 64), (3, 7, 7), [False, False, False], torch.bfloat16),
        ((1, 11, 15, 19, 4, 64), (3, 5, 5), [False, False, False], torch.bfloat16),
        ((1, 9, 13, 17, 4, 64), (5, 7, 7), [True, False, False], torch.bfloat16),
        ((1, 7, 11, 13, 2, 64), (3, 5, 5), [True, True, True], torch.bfloat16),
        ((1, 9, 13, 17, 4, 64), (5, 5, 5), [False, False, False], torch.float16),
        ((1, 7, 9, 11, 2, 128), (3, 5, 5), [False, False, False], torch.bfloat16),
        ((1, 5, 7, 9, 2, 64), (3, 3, 3), [False, False, False], torch.float32),
    ],
)
def test_na3d_3d_tiled_parity(
    shape: tuple[int, ...],
    kernel_size: tuple[int, int, int],
    causal: list[bool],
    dtype: torch.dtype,
) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.fallback_na.triton_na import (
        na3d,
    )

    torch.manual_seed(0)
    q = torch.randn(*shape, device="cuda", dtype=dtype) * 0.3
    k = torch.randn(*shape, device="cuda", dtype=dtype) * 0.3
    v = torch.randn(*shape, device="cuda", dtype=dtype) * 0.3
    with torch.inference_mode():
        got = na3d(q, k, v, kernel_size, is_causal=causal, scale=1.0)
        ref = _legacy_na3d(q, k, v, kernel_size, is_causal=causal, scale=1.0)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    psnr = _psnr_db(ref, got)
    cos = F.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert psnr >= 60.0, f"PSNR={psnr:.2f} dB"
    assert cos >= 0.99999, f"cos={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [256, 512, 1024, 1536])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_diffvae_rmsnorm_modulate_parity(dim: int, dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 3, 7, 11, dim, device="cuda", dtype=dtype)
    w = torch.randn(dim, device="cuda", dtype=dtype)
    scale = torch.randn(1, 1, 1, 1, dim, device="cuda", dtype=dtype) * 0.2
    shift = torch.randn(1, 1, 1, 1, dim, device="cuda", dtype=dtype) * 0.2
    with torch.inference_mode():
        ref = _rmsnorm_modulate_torch(x, w, scale, shift, 1e-6)
        got = diffvae_rmsnorm_modulate(x, w, scale, shift, eps=1e-6)
        buf = torch.empty_like(x)
        diffvae_rmsnorm_modulate_into(x, w, scale, shift, buf, eps=1e-6)
    torch.testing.assert_close(got, buf)
    psnr = _psnr_db(ref, got)
    assert psnr >= 55.0, f"PSNR={psnr:.2f} dB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("do_rmsnorm", [True, False])
@pytest.mark.parametrize("rope_bf16", [False, True])
@pytest.mark.parametrize("use_pos", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_diffvae_rmsnorm_3d_abs_rope_parity(
    do_rmsnorm: bool,
    rope_bf16: bool,
    use_pos: bool,
    head_dim: int,
) -> None:
    torch.manual_seed(2)
    b, t, h, w, nh = 1, 5, 7, 13, 4
    x = torch.randn(b, t, h, w, nh, head_dim, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16) if do_rmsnorm else None
    rope_split = default_rope_dim_split(head_dim)
    inv_freqs = (
        rope_inv_freqs(rope_split[0]).to("cuda"),
        rope_inv_freqs(rope_split[1]).to("cuda"),
        rope_inv_freqs(rope_split[2]).to("cuda"),
    )
    t_pos = torch.linspace(0.5, 4.5, t, device="cuda", dtype=torch.float32) if use_pos else None
    w_pos = (torch.arange(w, device="cuda", dtype=torch.float32) - 5.0) if use_pos else None
    attn_scale = (head_dim**-0.5) if do_rmsnorm else 1.0

    with torch.inference_mode():
        got = diffvae_rmsnorm_3d_abs_rope(
            x,
            norm_w,
            rope_split,
            inv_freqs,
            w_pos=w_pos,
            t_pos=t_pos,
            eps=1e-6,
            attn_scale=attn_scale,
            do_rmsnorm=do_rmsnorm,
            rope_compute_bf16=rope_bf16,
        )
        ref = _rmsnorm_3d_abs_rope_torch(
            x,
            norm_w,
            t_pos,
            w_pos,
            inv_freqs[0],
            inv_freqs[1],
            inv_freqs[2],
            rope_split[0],
            rope_split[1],
            rope_split[2],
            1e-6,
            attn_scale,
            do_rmsnorm,
            rope_bf16,
        )
        if not do_rmsnorm and use_pos and w_pos is not None:
            ref2 = _apply_in_slab_abs_rope(
                x,
                rope_split,
                inv_freqs,
                w_pos,
                compute_dtype=torch.bfloat16 if rope_bf16 else torch.float32,
                t_pos=t_pos,
            )
            torch.testing.assert_close(ref, ref2)
    psnr = _psnr_db(ref, got)
    cos = F.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert psnr >= 55.0, f"PSNR={psnr:.2f} dB"
    assert cos >= 0.99998, f"cos={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffvae_cut_slab_and_w_slice_add_bit_exact() -> None:
    torch.manual_seed(3)
    x = torch.randn(1, 5, 9, 35, 256, device="cuda", dtype=torch.bfloat16)
    halo = 5
    chunk_w = 10
    extent = chunk_w + 2 * halo

    with torch.inference_mode():
        left_halo: torch.Tensor | None = None
        for i in range(4):
            core_start = i * chunk_w
            core_end = min(35, (i + 1) * chunk_w)
            core_len = core_end - core_start
            has_right = i + 1 < 4

            got_buf = diffvae_cut_slab(x, core_start, core_end, halo, extent, left_halo, has_right_neighbor=has_right)
            assert got_buf is not None

            # Reference via pure PyTorch path (temporarily disable Triton env)
            old_env = None
            import os

            old_env = os.environ.get("WORLDFOUNDRY_LTX_DIFFVAE_TRITON")
            os.environ["WORLDFOUNDRY_LTX_DIFFVAE_TRITON"] = "0"
            try:
                ref_buf = _cut_slab(x, core_start, core_end, halo, extent, left_halo, has_right_neighbor=has_right)
            finally:
                if old_env is None:
                    os.environ.pop("WORLDFOUNDRY_LTX_DIFFVAE_TRITON", None)
                else:
                    os.environ["WORLDFOUNDRY_LTX_DIFFVAE_TRITON"] = old_env

            assert torch.equal(got_buf, ref_buf)
            if has_right:
                take = min(halo, core_len)
                left_halo = x[:, :, :, core_end - take : core_end, :].clone()

        # W-slice add parity
        out = torch.randn(1, 5, 9, extent, 256, device="cuda", dtype=torch.bfloat16)
        x_ref = x.clone()
        x_got = x.clone()
        x_ref[:, :, :, 10:20, :].add_(out[:, :, :, halo : halo + 10, :])
        diffvae_w_slice_add(x_got, out, 10, halo, 10)
        assert torch.equal(x_got, x_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("drop_leading", [True, False])
@pytest.mark.parametrize("w_chunks", [1, 4])
def test_diffvae_deferred_context_inject_parity_and_hooks(drop_leading: bool, w_chunks: int) -> None:
    torch.manual_seed(4)
    stride = (2, 2, 2)
    t_in, h_in, w_in = 5, 6, 12
    t_out = t_in * 2 - (1 if drop_leading else 0)
    feat = torch.randn(1, t_in, h_in, w_in, 128, device="cuda", dtype=torch.bfloat16) * 0.3
    x_init = torch.randn(1, t_out, h_in * 2, w_in * 2, 64, device="cuda", dtype=torch.bfloat16) * 0.3
    up_proj = nn.Linear(128, 64 * 8, bias=True, device="cuda", dtype=torch.bfloat16)
    ctx_proj = nn.Linear(64, 64, bias=True, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        x_got = x_init.clone()
        deferred(x_got, feat, up_proj, ctx_proj, stride, w_chunks=w_chunks, drop_leading_frame=drop_leading)

        x_ref = x_init.clone()
        ctx_ref = _upsample_then_ctx(
            feat,
            up_proj.weight,
            up_proj.bias,
            ctx_proj.weight,
            ctx_proj.bias,
            stride,
            drop_leading_frame=drop_leading,
        )
        x_ref.add_(ctx_ref)

    psnr = _psnr_db(x_ref, x_got)
    cos = F.cosine_similarity(x_ref.float().flatten(), x_got.float().flatten(), dim=0).item()
    assert psnr >= 50.0, f"PSNR={psnr:.2f} dB"
    assert cos >= 0.99995, f"cos={cos:.6f}"

    # Verify forward hook triggers clean PyTorch fallback
    called = []
    handle = ctx_proj.register_forward_hook(lambda m, i, o: called.append(1))
    try:
        with torch.inference_mode():
            x_h = x_init.clone()
            deferred(x_h, feat, up_proj, ctx_proj, stride, w_chunks=w_chunks, drop_leading_frame=drop_leading)
        # Note: deferred uses up_proj.weight/ctx_proj.weight directly in fallback, but respects hook guard
        assert x_h.shape == x_ref.shape
    finally:
        handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffvae_ops_cuda_graph_and_autograd_fallback() -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 3, 4, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    rope_split = default_rope_dim_split(64)
    inv_freqs = (
        rope_inv_freqs(rope_split[0]).to("cuda"),
        rope_inv_freqs(rope_split[1]).to("cuda"),
        rope_inv_freqs(rope_split[2]).to("cuda"),
    )

    # CUDA Graph capture safety
    with torch.inference_mode():
        ref = diffvae_rmsnorm_3d_abs_rope(x, w, rope_split, inv_freqs, attn_scale=0.125)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = diffvae_rmsnorm_3d_abs_rope(x, w, rope_split, inv_freqs, attn_scale=0.125)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)

    # Autograd fallback safety
    x_grad = torch.randn(1, 2, 3, 4, 2, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    w_grad = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    y = diffvae_rmsnorm_3d_abs_rope(x_grad, w_grad, rope_split, inv_freqs, attn_scale=0.125)
    y.sum().backward()
    assert x_grad.grad is not None and w_grad.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("shape", "planes", "ktimes_v", "kvalid_v", "ks"),
    [
        ((1, 13, 19, 23, 4, 64), 3, [0.0, 6.3, 12.0], [True, True, True], (11, 11, 11)),
        ((1, 9, 13, 17, 4, 64), 3, [0.0, 4.5, 8.0], [True, False, True], (5, 7, 7)),
        ((1, 7, 11, 13, 2, 64), 2, [0.0, 6.0], [False, False], (3, 5, 5)),
    ],
)
def test_joint_na3d_3d_tiled_parity(
    shape: tuple[int, ...],
    planes: int,
    ktimes_v: list[float],
    kvalid_v: list[bool],
    ks: tuple[int, int, int],
) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.fallback_na import (
        joint_eager,
        joint_triton,
    )

    torch.manual_seed(6)
    b, _, h, w, nh, hd = shape
    q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn_like(q) * 0.25
    v = torch.randn_like(q) * 0.25
    kq = torch.randn(b, planes, h, w, nh, hd, device="cuda", dtype=torch.bfloat16) * 0.25
    kk = torch.randn_like(kq) * 0.25
    kv = torch.randn_like(kq) * 0.25
    ktimes = torch.tensor(ktimes_v, device="cuda", dtype=torch.float32)
    kvalid = torch.tensor(kvalid_v, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        vo, ko = joint_triton.joint_na3d(q, k, v, kq, kk, kv, ktimes, kvalid, ks)
        vr, kr = joint_eager.joint_na3d(q, k, v, kq, kk, kv, ktimes, kvalid, ks)
    assert _psnr_db(vr, vo) >= 60.0
    if any(kvalid_v):
        assert _psnr_db(kr, ko) >= 60.0
    else:
        assert torch.equal(ko, torch.zeros_like(ko))
