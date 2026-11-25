"""Parity, layout, CUDA-Graph, compile, and fallback tests for VideoVAE Triton operators."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.original_decoder import (
    DecoderCausal3D,
    ResnetBlockCausal3D,
    UpsampleCausal3D,
    prepare_causal_attention_mask,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.original_encoder import (
    EncoderCausal3D,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.triton_vae_ops import (
    _group_norm_3d_silu_torch,
    _is_channels_last_3d,
    _upsample_causal_3d_torch,
    triton_group_norm_3d_silu,
    triton_upsample_causal_3d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import PixelNorm
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import NormLayerType
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.resnet import ResnetBlock3D
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.sampling import DepthToSpaceUpsample
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.triton_vae_ops import (
    _depth_to_space_3d_torch,
    _pixel_norm_adaln_silu_torch,
    triton_depth_to_space_3d,
    triton_pixel_norm_adaln_silu,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.norm import (
    FusedGroupNorm3D,
    SpatialNorm3D,
    TemporalIsolatedGroupNorm,
    fused_group_norm,
    fused_spatial_norm,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.triton_norm import (
    _ACT_TO_CODE,
    _temporal_isolated_group_norm_act_torch,
    triton_temporal_isolated_group_norm_act,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _psnr_db(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.float()
    r = ref.float()
    mse = torch.mean((a - r) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = max(float(r.abs().max().item()), 1.0)
    return 10.0 * math.log10((peak * peak) / mse)


def _assert_lossless(actual: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> None:
    assert actual.shape == ref.shape
    assert actual.dtype == ref.dtype
    diff = (actual.float() - ref.float()).abs()
    max_diff = diff.max().item()
    if dtype == torch.float32:
        assert max_diff <= 1e-5, f"max_abs_diff={max_diff:.6g} exceeded 1e-5 for {dtype}"
    else:
        # BF16 has 7 mantissa bits (1 ULP = 2^-7 * |v|, e.g. 0.015625 for |v| in [2, 4));
        # verify <= 1e-2 absolute OR <= 1.01 ULP at that value, plus PSNR >= 55 dB.
        ulp = (torch.nextafter(ref.abs(), torch.full_like(ref, float("inf"))) - ref.abs()).float()
        allowed = torch.maximum(torch.full_like(diff, 1e-2), 1.01 * ulp)
        worst_excess = (diff - allowed).max().item()
        assert worst_excess <= 0.0, f"max_abs_diff={max_diff:.6g} exceeded max(1e-2, 1 ULP) for {dtype}"
    psnr = _psnr_db(actual, ref)
    assert psnr >= 55.0, f"PSNR={psnr:.2f} dB < 55 dB for {dtype}"


def _disable_tf32():
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return prev_cudnn, prev_matmul


def _restore_tf32(prev: tuple[bool, bool]) -> None:
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def _make_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    t = torch.randn(shape, device=device, dtype=dtype)
    if channels_last and len(shape) == 5:
        return t.contiguous(memory_format=torch.channels_last_3d)
    if channels_last and len(shape) == 4:
        return t.contiguous(memory_format=torch.channels_last)
    return t.contiguous()


# =============================================================================
# 1. Lazy Triton Import & CPU / Autograd Fallbacks
# =============================================================================


def test_lazy_triton_import_does_not_load_triton() -> None:
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "import torch\n"
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video import triton_vae_ops as hy\n"
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video import triton_norm as mm\n"
        "from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import triton_vae_ops as ltx\n"
        "assert hy._TRITON_KERNELS is None and mm._TRITON_KERNELS is None and ltx._TRITON_KERNELS is None\n"
        "x = torch.randn(1, 16, 2, 4, 4)\n"
        "assert hy.triton_group_norm_3d_silu(x, 4).shape == x.shape\n"
        "assert hy.triton_upsample_causal_3d(x, (2, 2, 2)).shape == (1, 16, 3, 8, 8)\n"
        "assert mm.triton_temporal_isolated_group_norm_act(x, 4).shape == x.shape\n"
        "assert ltx.triton_pixel_norm_adaln_silu(x).shape == x.shape\n"
        "assert ltx.triton_depth_to_space_3d(torch.randn(1, 32, 2, 4, 4), (2, 2, 2)).shape == (1, 4, 3, 8, 8)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_cpu_fallbacks_all_ops() -> None:
    torch.manual_seed(0)
    # Hunyuan GroupNorm + Upsample + Causal mask
    x = torch.randn(2, 16, 3, 5, 7)
    w = torch.randn(16)
    b = torch.randn(16)
    out_gn = triton_group_norm_3d_silu(x, 4, w, b, 1e-6, apply_silu=True)
    ref_gn = F.silu(F.group_norm(x, 4, w, b, 1e-6))
    assert torch.allclose(out_gn, ref_gn, atol=1e-6)

    out_up = triton_upsample_causal_3d(x, (2, 2, 2))
    ref_up = _upsample_causal_3d_torch(x, 2, 2, 2)
    assert torch.equal(out_up, ref_up)

    # Vectorized causal mask bitwise identical
    for nf, nhw in [(1, 1), (3, 5), (5, 7)]:
        seq = nf * nhw
        ref_mask = torch.full((seq, seq), float("-inf"))
        for i in range(seq):
            ref_mask[i, : (i // nhw + 1) * nhw] = 0
        got_mask = prepare_causal_attention_mask(nf, nhw, torch.float32, torch.device("cpu"))
        assert torch.equal(got_mask, ref_mask)

    # MiniMax-H3 temporal-isolated GN + spatial affine
    ds = torch.randn(2, 16, 2, 3, 4)
    db = torch.randn(2, 16, 2, 3, 4)
    out_mm = triton_temporal_isolated_group_norm_act(x, 4, w, b, ds, db, 1e-6, "silu", temporal_isolated=True)
    ref_mm = _temporal_isolated_group_norm_act_torch(x, 4, w, b, ds, db, 1e-6, _ACT_TO_CODE["silu"], True)
    assert torch.allclose(out_mm, ref_mm, atol=1e-6)

    # LTX PixelNorm + AdaLN + SiLU
    sc = torch.randn(2, 16, 1, 1, 1)
    sh = torch.randn(2, 16, 1, 1, 1)
    out_pn = triton_pixel_norm_adaln_silu(x, sc, sh, 1e-8, apply_silu=True)
    ref_pn = _pixel_norm_adaln_silu_torch(x, sc, sh, 1e-8, True)
    assert torch.allclose(out_pn, ref_pn, atol=1e-6)

    # LTX DepthToSpace
    x_d2s = torch.randn(2, 32, 3, 4, 5)
    x_res = torch.randn(2, 16, 3, 4, 5)
    out_d2s = triton_depth_to_space_3d(x_d2s, (2, 2, 2), residual=x_res, out_channels_reduction_factor=4)
    ref_d2s = _depth_to_space_3d_torch(x_d2s, 2, 2, 2, x_res, 4)
    assert torch.allclose(out_d2s, ref_d2s, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_autograd_fallback_preserves_gradients(device: str) -> None:
    torch.manual_seed(1)
    x1 = torch.randn(1, 16, 2, 4, 4, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    w1 = torch.randn(16, device=device, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    b1 = torch.randn(16, device=device, requires_grad=True)
    b2 = b1.detach().clone().requires_grad_(True)

    triton_group_norm_3d_silu(x1, 4, w1, b1, 1e-6, apply_silu=True).sum().backward()
    _group_norm_3d_silu_torch(x2, 4, w2, b2, 1e-6, True).sum().backward()
    assert x1.grad is not None and x2.grad is not None
    assert w1.grad is not None and w2.grad is not None
    assert b1.grad is not None and b2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(w1.grad, w2.grad, atol=1e-5)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-5)

    # Upsample autograd
    u1 = torch.randn(1, 8, 2, 3, 3, device=device, requires_grad=True)
    u2 = u1.detach().clone().requires_grad_(True)
    triton_upsample_causal_3d(u1, (2, 2, 2)).sum().backward()
    _upsample_causal_3d_torch(u2, 2, 2, 2).sum().backward()
    assert u1.grad is not None and u2.grad is not None
    assert torch.allclose(u1.grad, u2.grad, atol=1e-5)

    # MiniMax-H3 autograd
    m1 = torch.randn(1, 16, 2, 4, 4, device=device, requires_grad=True)
    m2 = m1.detach().clone().requires_grad_(True)
    triton_temporal_isolated_group_norm_act(m1, 4, w1.detach(), b1.detach(), None, None, 1e-6, "silu").sum().backward()
    _temporal_isolated_group_norm_act_torch(
        m2, 4, w1.detach(), b1.detach(), None, None, 1e-6, _ACT_TO_CODE["silu"], True
    ).sum().backward()
    assert m1.grad is not None and m2.grad is not None
    assert torch.allclose(m1.grad, m2.grad, atol=1e-5)

    # LTX PixelNorm + D2S autograd
    p1 = torch.randn(1, 16, 2, 4, 4, device=device, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_(True)
    sc = torch.randn(1, 16, 1, 1, 1, device=device)
    sh = torch.randn(1, 16, 1, 1, 1, device=device)
    triton_pixel_norm_adaln_silu(p1, sc, sh, 1e-8, apply_silu=True).sum().backward()
    _pixel_norm_adaln_silu_torch(p2, sc, sh, 1e-8, True).sum().backward()
    assert p1.grad is not None and p2.grad is not None
    assert torch.allclose(p1.grad, p2.grad, atol=1e-5)

    d1 = torch.randn(1, 32, 2, 3, 3, device=device, requires_grad=True)
    d2 = d1.detach().clone().requires_grad_(True)
    r1 = torch.randn(1, 32, 2, 3, 3, device=device, requires_grad=True)
    r2 = r1.detach().clone().requires_grad_(True)
    triton_depth_to_space_3d(d1, (2, 2, 2), residual=r1, out_channels_reduction_factor=8).sum().backward()
    _depth_to_space_3d_torch(d2, 2, 2, 2, r2, 8).sum().backward()
    assert d1.grad is not None and d2.grad is not None
    assert r1.grad is not None and r2.grad is not None
    assert torch.allclose(d1.grad, d2.grad, atol=1e-5)
    assert torch.allclose(r1.grad, r2.grad, atol=1e-5)


# =============================================================================
# 2. HunyuanVideo Triton Ops (CUDA)
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("apply_silu", [True, False])
@pytest.mark.parametrize(
    "shape,num_groups,affine",
    [
        ((1, 128, 3, 8, 8), 32, True),  # small (one-pass)
        ((1, 256, 5, 32, 32), 32, True),  # large (3-stage chunked)
        ((2, 96, 3, 19, 23), 32, True),  # odd spatial + non-pow2 CG=3
        ((1, 64, 2, 13, 17), 16, False),  # affine=False
    ],
)
def test_hunyuan_group_norm_3d_silu_parity(
    dtype: torch.dtype,
    channels_last: bool,
    apply_silu: bool,
    shape: tuple[int, int, int, int, int],
    num_groups: int,
    affine: bool,
) -> None:
    torch.manual_seed(42)
    x = _make_tensor(shape, dtype=dtype, device="cuda", channels_last=channels_last)
    c = shape[1]
    w = torch.randn(c, device="cuda", dtype=dtype) if affine else None
    b = torch.randn(c, device="cuda", dtype=dtype) if affine else None

    with torch.no_grad():
        actual = triton_group_norm_3d_silu(x, num_groups, w, b, 1e-6, apply_silu=apply_silu)
        ref = _group_norm_3d_silu_torch(
            x.float(),
            num_groups,
            w.float() if w is not None else None,
            b.float() if b is not None else None,
            1e-6,
            apply_silu,
        ).to(dtype)

    _assert_lossless(actual, ref, dtype)
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
def test_hunyuan_group_norm_large_mean_shift_stability() -> None:
    """Chan's shifted variance beats naive FP32 E[x^2]-E[x]^2 against FP64 truth when mean >> std."""
    torch.manual_seed(7)
    x = (500.0 + 0.05 * torch.randn(1, 128, 4, 32, 32, device="cuda", dtype=torch.float32)).contiguous(
        memory_format=torch.channels_last_3d
    )
    w = torch.randn(128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        actual = triton_group_norm_3d_silu(x, 32, w, b, 1e-6, apply_silu=True)
        ref_f32 = _group_norm_3d_silu_torch(x, 32, w, b, 1e-6, True)
        ref_f64 = _group_norm_3d_silu_torch(x.double(), 32, w.double(), b.double(), 1e-6, True).float()
    triton_err = (actual - ref_f64).abs().max().item()
    torch_err = (ref_f32 - ref_f64).abs().max().item()
    assert triton_err <= max(torch_err, 1e-3), f"triton_err={triton_err} > torch_err={torch_err}"
    assert _psnr_db(actual, ref_f64) >= 75.0


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "shape,factor",
    [
        ((1, 128, 1, 9, 13), (2, 2, 2)),
        ((1, 128, 5, 17, 23), (2, 2, 2)),
        ((2, 64, 4, 15, 19), (1, 2, 2)),
        ((1, 48, 3, 7, 11), (2, 1, 1)),
        ((1, 32, 3, 5, 7), (1, 3, 3)),  # non-pow2 fallback
    ],
)
def test_hunyuan_upsample_causal_3d_bit_exact(
    dtype: torch.dtype,
    channels_last: bool,
    shape: tuple[int, int, int, int, int],
    factor: tuple[int, int, int],
) -> None:
    torch.manual_seed(11)
    x = _make_tensor(shape, dtype=dtype, device="cuda", channels_last=channels_last)
    with torch.no_grad():
        actual = triton_upsample_causal_3d(x, factor)
        ref = _upsample_causal_3d_torch(x, *factor)
    assert torch.equal(actual, ref), "UpsampleCausal3D must be bit-exact"
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_hunyuan_decoder_and_encoder_module_parity(channels_last: bool) -> None:
    torch.manual_seed(123)
    dec = DecoderCausal3D(
        in_channels=8,
        out_channels=3,
        block_out_channels=[32, 64],
        layers_per_block=1,
        num_groups=8,
        time_compression_ratio=2,
        spatial_compression_ratio=2,
    ).to(device="cuda", dtype=torch.bfloat16)
    enc = EncoderCausal3D(
        in_channels=3,
        out_channels=8,
        block_out_channels=[32, 64],
        layers_per_block=1,
        num_groups=8,
        time_compression_ratio=2,
        spatial_compression_ratio=2,
    ).to(device="cuda", dtype=torch.bfloat16)
    if channels_last:
        from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

        convert_convolution_weight_layouts(dec, conv3d=True)
        convert_convolution_weight_layouts(enc, conv3d=True)

    # Verify state_dict keys unchanged
    assert "mid_block.resnets.0.norm1.weight" in dec.state_dict()
    assert "conv_norm_out.weight" in dec.state_dict()
    assert "conv_norm_out.weight" in enc.state_dict()

    z = _make_tensor((1, 8, 3, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    with torch.no_grad():
        y = dec(z)
        z_rec = enc(y)
    assert y.shape == (1, 3, 5, 16, 16)
    assert z_rec.shape == (1, 16, 3, 8, 8)
    assert torch.isfinite(y).all() and torch.isfinite(z_rec).all()

    # Check ResnetBlockCausal3D and UpsampleCausal3D direct parity against CPU/FP32 reference (TF32 disabled)
    prev_tf32 = _disable_tf32()
    try:
        blk = ResnetBlockCausal3D(32, 32, groups=8).to(device="cuda", dtype=torch.float32)
        bx = _make_tensor((1, 32, 3, 8, 8), dtype=torch.float32, device="cuda", channels_last=channels_last)
        with torch.no_grad():
            got = blk(bx)
            ref_blk = ResnetBlockCausal3D(32, 32, groups=8)
            ref_blk.load_state_dict(blk.state_dict())
            expected = ref_blk(bx.cpu()).cuda()
        _assert_lossless(got, expected, torch.float32)
    finally:
        _restore_tf32(prev_tf32)

    up = UpsampleCausal3D(32, use_conv=False, upsample_factor=(2, 2, 2)).cuda()
    with torch.no_grad():
        assert torch.equal(up(bx), _upsample_causal_3d_torch(bx, 2, 2, 2))


# =============================================================================
# 3. MiniMax-H3 VideoVAE Triton Norm (CUDA)
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("activation", ["identity", "silu", "relu"])
@pytest.mark.parametrize("temporal_isolated", [True, False])
@pytest.mark.parametrize(
    "shape,num_groups,dyn_shape",
    [
        ((1, 128, 4, 16, 16), 32, None),
        ((1, 128, 3, 40, 48), 32, (1, 128, 2, 10, 12)),  # coarse nearest-upsampled dynamic affine
        ((2, 96, 3, 13, 19), 32, (2, 96, 1, 1, 1)),  # non-pow2 CG=3 + broadcast dynamic affine
        ((1, 64, 5, 19, 23), 16, (1, 64, 3, 7, 11)),  # odd prime nearest-upsample ratios
    ],
)
def test_minimax_h3_temporal_isolated_gn_parity(
    dtype: torch.dtype,
    channels_last: bool,
    activation: str,
    temporal_isolated: bool,
    shape: tuple[int, int, int, int, int],
    num_groups: int,
    dyn_shape: tuple[int, int, int, int, int] | None,
) -> None:
    torch.manual_seed(77)
    x = _make_tensor(shape, dtype=dtype, device="cuda", channels_last=channels_last)
    c = shape[1]
    w = (1.0 + 0.1 * torch.randn(c, device="cuda", dtype=torch.float32)).to(dtype)
    b = (0.1 * torch.randn(c, device="cuda", dtype=torch.float32)).to(dtype)
    ds = (
        (1.0 + 0.1 * _make_tensor(dyn_shape, dtype=torch.float32, device="cuda", channels_last=False)).to(dtype)
        if dyn_shape
        else None
    )
    db = (
        (0.1 * _make_tensor(dyn_shape, dtype=torch.float32, device="cuda", channels_last=channels_last)).to(dtype)
        if dyn_shape
        else None
    )

    with torch.no_grad():
        actual = triton_temporal_isolated_group_norm_act(
            x,
            num_groups,
            w,
            b,
            ds,
            db,
            1e-6,
            activation,
            temporal_isolated=temporal_isolated,
        )
        ref = _temporal_isolated_group_norm_act_torch(
            x.float(),
            num_groups,
            w.float(),
            b.float(),
            ds.float() if ds is not None else None,
            db.float() if db is not None else None,
            1e-6,
            _ACT_TO_CODE[activation],
            temporal_isolated,
        ).to(dtype)

    _assert_lossless(actual, ref, dtype)
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_minimax_h3_norm_modules_parity(channels_last: bool) -> None:
    torch.manual_seed(99)
    x = _make_tensor((1, 64, 4, 16, 16), dtype=torch.float32, device="cuda", channels_last=channels_last)
    cond = _make_tensor((1, 16, 2, 8, 8), dtype=torch.float32, device="cuda", channels_last=channels_last)

    prev_tf32 = _disable_tf32()
    try:
        with torch.no_grad():
            # TemporalIsolatedGroupNorm
            tign = TemporalIsolatedGroupNorm(32, 64, eps=1e-6, affine=True).cuda()
            tign_cpu = TemporalIsolatedGroupNorm(32, 64, eps=1e-6, affine=True)
            tign_cpu.load_state_dict(tign.state_dict())
            _assert_lossless(tign(x), tign_cpu(x.cpu()).cuda(), torch.float32)

            # FusedGroupNorm3D without spatial affine
            fgn = FusedGroupNorm3D(32, 64, eps=1e-6, activation="silu", use_t_isolated_gn=True).cuda()
            fgn_cpu = FusedGroupNorm3D(32, 64, eps=1e-6, activation="silu", use_t_isolated_gn=True)
            fgn_cpu.load_state_dict(fgn.state_dict())
            _assert_lossless(fgn(x), fgn_cpu(x.cpu()).cuda(), torch.float32)

            # FusedGroupNorm3D with spatial affine
            fgn_sp = FusedGroupNorm3D(
                32, 64, eps=1e-6, activation="silu", cond_channels=16, use_t_isolated_gn=True
            ).cuda()
            fgn_sp_cpu = FusedGroupNorm3D(32, 64, eps=1e-6, activation="silu", cond_channels=16, use_t_isolated_gn=True)
            fgn_sp_cpu.load_state_dict(fgn_sp.state_dict())
            _assert_lossless(fgn_sp(x, cond), fgn_sp_cpu(x.cpu(), cond.cpu()).cuda(), torch.float32)

            # SpatialNorm3D
            sp = SpatialNorm3D(64, 16, use_t_isolated_gn=True).cuda()
            sp_cpu = SpatialNorm3D(64, 16, use_t_isolated_gn=True)
            sp_cpu.load_state_dict(sp.state_dict())
            _assert_lossless(sp(x, cond), sp_cpu(x.cpu(), cond.cpu()).cuda(), torch.float32)

            # Helper functions
            w = torch.randn(64, device="cuda")
            b = torch.randn(64, device="cuda")
            _assert_lossless(
                fused_group_norm(x, 32, w, b, 1e-6, "relu"),
                F.relu(F.group_norm(x.cpu(), 32, w.cpu(), b.cpu(), 1e-6)).cuda(),
                torch.float32,
            )
            ds = torch.randn(1, 64, 4, 16, 16, device="cuda")
            db = torch.randn(1, 64, 4, 16, 16, device="cuda")
            _assert_lossless(
                fused_spatial_norm(x, 32, w, b, ds, db, 1e-6, "silu"),
                F.silu(F.group_norm(x.cpu(), 32, w.cpu(), b.cpu(), 1e-6) * ds.cpu() + db.cpu()).cuda(),
                torch.float32,
            )
    finally:
        _restore_tf32(prev_tf32)


# =============================================================================
# 4. LTX-2.3 / LTX-2.5 VideoVAE Triton Ops (CUDA)
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_adaln", [False, True])
@pytest.mark.parametrize("apply_silu", [False, True])
@pytest.mark.parametrize(
    "shape,spatial_adaln",
    [
        ((1, 128, 5, 16, 16), False),
        ((2, 256, 3, 13, 19), False),
        ((1, 96, 3, 11, 15), True),  # non-pow2 C + spatially varying scale/shift
        ((1, 2560, 2, 4, 4), False),  # C > 2048 two-pass NDHWC
    ],
)
def test_ltx_pixel_norm_adaln_silu_parity(
    dtype: torch.dtype,
    channels_last: bool,
    with_adaln: bool,
    apply_silu: bool,
    shape: tuple[int, int, int, int, int],
    spatial_adaln: bool,
) -> None:
    torch.manual_seed(55)
    x = _make_tensor(shape, dtype=dtype, device="cuda", channels_last=channels_last)
    b, c, d, h, w = shape
    ada_shape = (b, c, d, h, w) if spatial_adaln else (b, c, 1, 1, 1)
    sc = (
        (0.1 * _make_tensor(ada_shape, dtype=torch.float32, device="cuda", channels_last=False)).to(dtype)
        if with_adaln
        else None
    )
    sh = (
        (0.1 * _make_tensor(ada_shape, dtype=torch.float32, device="cuda", channels_last=False)).to(dtype)
        if with_adaln
        else None
    )

    with torch.no_grad():
        actual = triton_pixel_norm_adaln_silu(x, sc, sh, 1e-8, apply_silu=apply_silu)
        ref = _pixel_norm_adaln_silu_torch(
            x.float(),
            sc.float() if sc is not None else None,
            sh.float() if sh is not None else None,
            1e-8,
            apply_silu,
        ).to(dtype)

    _assert_lossless(actual, ref, dtype)
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
def test_ltx_pixel_norm_fp16_no_overflow_and_4d() -> None:
    torch.manual_seed(17)
    # Values ~500 would overflow FP16 x**2 (250000 > 65504) if not computed in FP32
    x = (500.0 * torch.randn(1, 128, 3, 8, 8, device="cuda", dtype=torch.float16)).contiguous(
        memory_format=torch.channels_last_3d
    )
    pn = PixelNorm(dim=1, eps=1e-8)
    with torch.no_grad():
        out = pn(x)
        ref = _pixel_norm_adaln_silu_torch(x.float(), None, None, 1e-8, False).to(torch.float16)
    assert torch.isfinite(out).all()
    _assert_lossless(out, ref, torch.float16)

    # 4-D audio NCHW / NHWC support
    x4 = _make_tensor((2, 64, 16, 20), dtype=torch.bfloat16, device="cuda", channels_last=True)
    with torch.no_grad():
        out4 = pn(x4)
        ref4 = (x4.float() / torch.sqrt(torch.mean(x4.float() ** 2, dim=1, keepdim=True) + 1e-8)).to(torch.bfloat16)
    _assert_lossless(out4, ref4, torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize(
    "in_c,stride,red,dhw",
    [
        (128, (2, 2, 2), 2, (3, 7, 9)),  # C_out=64, C_res_unpacked=16, num_repeat=4
        (128, (2, 2, 2), 1, (2, 8, 8)),  # C_out=128, C_res_unpacked=16, num_repeat=8
        (128, (1, 2, 2), 2, (4, 9, 11)),  # spatial-only 2x
        (64, (2, 1, 1), 2, (5, 7, 13)),  # temporal-only 2x
        (96, (2, 2, 2), 2, (2, 5, 7)),  # odd C_res_unpacked=12 (non-pow2!)
        (36, (1, 3, 3), 3, (2, 5, 7)),  # non-pow2 stride general fallback
    ],
)
def test_ltx_depth_to_space_3d_parity(
    dtype: torch.dtype,
    channels_last: bool,
    with_residual: bool,
    in_c: int,
    stride: tuple[int, int, int],
    red: int,
    dhw: tuple[int, int, int],
) -> None:
    torch.manual_seed(66)
    p = math.prod(stride)
    conv_c = p * in_c // red
    d, h, w = dhw
    x_conv = _make_tensor((2, conv_c, d, h, w), dtype=dtype, device="cuda", channels_last=channels_last)
    x_res = (
        _make_tensor((2, in_c, d, h, w), dtype=dtype, device="cuda", channels_last=channels_last)
        if with_residual
        else None
    )

    with torch.no_grad():
        actual = triton_depth_to_space_3d(x_conv, stride, residual=x_res, out_channels_reduction_factor=red)
        ref = _depth_to_space_3d_torch(x_conv, *stride, x_res, red)

    if not with_residual:
        assert torch.equal(actual, ref), "DepthToSpace3D without residual must be bit-exact"
    else:
        assert torch.equal(actual, ref) or (actual.float() - ref.float()).abs().max().item() <= (
            1e-5 if dtype == torch.float32 else 1e-2
        )
    assert _is_channels_last_3d(actual) == channels_last


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("timestep_conditioning", [False, True])
def test_ltx_resnet_and_d2s_modules_parity(channels_last: bool, timestep_conditioning: bool) -> None:
    torch.manual_seed(88)
    prev_tf32 = _disable_tf32()
    try:
        blk = ResnetBlock3D(
            dims=3,
            in_channels=64,
            out_channels=64,
            norm_layer=NormLayerType.PIXEL_NORM,
            timestep_conditioning=timestep_conditioning,
        ).cuda()
        ref_blk = ResnetBlock3D(
            dims=3,
            in_channels=64,
            out_channels=64,
            norm_layer=NormLayerType.PIXEL_NORM,
            timestep_conditioning=timestep_conditioning,
        )
        ref_blk.load_state_dict(blk.state_dict())

        x = _make_tensor((1, 64, 3, 8, 8), dtype=torch.float32, device="cuda", channels_last=channels_last)
        t = torch.randn(1, 256, 1, 1, 1, device="cuda") if timestep_conditioning else None
        with torch.no_grad():
            got = blk(x, causal=True, timestep=t)
            expected = ref_blk(x.cpu(), causal=True, timestep=t.cpu() if t is not None else None).cuda()
        _assert_lossless(got, expected, torch.float32)

        for residual in (False, True):
            up = DepthToSpaceUpsample(
                dims=3, in_channels=64, stride=(2, 2, 2), residual=residual, out_channels_reduction_factor=2
            ).cuda()
            ref_up = DepthToSpaceUpsample(
                dims=3, in_channels=64, stride=(2, 2, 2), residual=residual, out_channels_reduction_factor=2
            )
            ref_up.load_state_dict(up.state_dict())
            with torch.no_grad():
                got_u = up(x, causal=True)
                exp_u = ref_up(x.cpu(), causal=True).cuda()
            _assert_lossless(got_u, exp_u, torch.float32)
    finally:
        _restore_tf32(prev_tf32)


# =============================================================================
# 5. CUDA Graph Capture Safety & torch.compile Fullgraph Safety
# =============================================================================


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_all_five_ops_cuda_graph_capture_and_replay(channels_last: bool) -> None:
    torch.manual_seed(2026)
    x = _make_tensor((1, 128, 3, 16, 16), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    ds = _make_tensor((1, 128, 2, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    db = _make_tensor((1, 128, 2, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    sc = torch.randn(1, 128, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    sh = torch.randn(1, 128, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    x_d2s = _make_tensor((1, 512, 3, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    x_res = _make_tensor((1, 128, 3, 8, 8), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)

    def run_all():
        y1 = triton_group_norm_3d_silu(x, 32, w, b, 1e-6, apply_silu=True)
        y2 = triton_upsample_causal_3d(x, (2, 2, 2))
        y3 = triton_temporal_isolated_group_norm_act(x, 32, w, b, ds, db, 1e-6, "silu", temporal_isolated=True)
        y4 = triton_pixel_norm_adaln_silu(x, sc, sh, 1e-8, apply_silu=True)
        y5 = triton_depth_to_space_3d(x_d2s, (2, 2, 2), residual=x_res, out_channels_reduction_factor=2)
        return y1, y2, y3, y4, y5

    # Warmup on side stream before CUDA Graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(2):
            _ = run_all()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        outs = run_all()

    # Mutate inputs, replay, and compare with eager
    x.normal_()
    x_d2s.normal_()
    x_res.normal_()
    g.replay()
    torch.cuda.synchronize()

    with torch.no_grad():
        eager = run_all()

    for got, exp in zip(outs, eager, strict=True):
        assert torch.equal(got, exp)


@requires_cuda
@pytest.mark.parametrize("channels_last", [False, True])
def test_all_five_ops_torch_compile_fullgraph(channels_last: bool) -> None:
    torch.manual_seed(2027)
    x = _make_tensor((1, 128, 3, 12, 12), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    ds = _make_tensor((1, 128, 2, 6, 6), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    db = _make_tensor((1, 128, 2, 6, 6), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    sc = torch.randn(1, 128, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    sh = torch.randn(1, 128, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    x_d2s = _make_tensor((1, 512, 3, 6, 6), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)
    x_res = _make_tensor((1, 128, 3, 6, 6), dtype=torch.bfloat16, device="cuda", channels_last=channels_last)

    def fn(x_, w_, b_, ds_, db_, sc_, sh_, x_d2s_, x_res_):
        y1 = triton_group_norm_3d_silu(x_, 32, w_, b_, 1e-6, apply_silu=True)
        y2 = triton_upsample_causal_3d(x_, (2, 2, 2))
        y3 = triton_temporal_isolated_group_norm_act(x_, 32, w_, b_, ds_, db_, 1e-6, "silu", temporal_isolated=True)
        y4 = triton_pixel_norm_adaln_silu(x_, sc_, sh_, 1e-8, apply_silu=True)
        y5 = triton_depth_to_space_3d(x_d2s_, (2, 2, 2), residual=x_res_, out_channels_reduction_factor=2)
        return y1, y2, y3, y4, y5

    compiled = torch.compile(fn, fullgraph=True)
    with torch.no_grad():
        got = compiled(x, w, b, ds, db, sc, sh, x_d2s, x_res)
        exp = fn(x, w, b, ds, db, sc, sh, x_d2s, x_res)

    for g_out, e_out in zip(got, exp, strict=True):
        assert torch.equal(g_out, e_out)
        assert _is_channels_last_3d(g_out) == channels_last
