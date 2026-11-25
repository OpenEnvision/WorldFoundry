# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and parity tests for the Self-Forcing 1.3B lossless Hopper Triton operator suite."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing import (
    self_forcing as sf_mod,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.fast_kernels import (
    set_sf_exact_norms,
    set_sf_fast_kernels_enabled,
    sf_apply_rope_bshd,
    sf_exact_norms_enabled,
    sf_fast_kernels_enabled,
    sf_layernorm_adaln,
    sf_qk_rmsnorm_rope_write_kv,
    sf_residual_add_layernorm_adaln,
    sf_residual_gate_add,
    sf_residual_gate_add_layernorm,
    sf_rmsnorm,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.model import rope_params
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
    CausalHead,
    CausalWanAttentionBlock,
    CausalWanModel,
    _build_3d_rope_cos_sin,
    causal_rope_apply,
)
from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import fuse_qkv_projections

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())


def _assert_lossless_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    min_exact_frac: float = 0.999,
) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert bool(torch.isfinite(actual).all())
    max_diff = float((actual.float() - expected.float()).abs().max().item())
    cos = _cosine_similarity(actual, expected)
    if dtype == torch.float32:
        assert max_diff <= 1e-5, f"FP32 max_abs_diff={max_diff}"
        assert cos > 0.999999, f"FP32 cos_sim={cos}"
    elif dtype == torch.float16:
        assert max_diff <= 5e-3, f"FP16 max_abs_diff={max_diff}"
        assert cos > 0.99999, f"FP16 cos_sim={cos}"
    else:
        # BF16 at magnitude ~4 has 1 ULP = 0.03125; typical normal values have <= 1e-2 (1 ULP).
        assert max_diff <= 3.2e-2, f"BF16 max_abs_diff={max_diff}"
        assert cos > 0.99999, f"BF16 cos_sim={cos}"
        exact_frac = float((actual == expected).float().mean().item())
        assert exact_frac >= min_exact_frac, f"BF16 exact_frac={exact_frac}"


def test_lazy_triton_import_and_no_cuda_init_on_cpu_import() -> None:
    """Importing fast_kernels directly MUST NOT import triton or initialize CUDA, and runs cleanly without triton."""
    fk_path = os.path.abspath(
        "worldfoundry/base_models/diffusion_model/models/networks/wan/variants/forcing/fast_kernels.py"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    # 1. Isolated fast_kernels import + CPU op call must never import triton or initialize CUDA
    code_isolated = (
        "import importlib.util, sys, torch; "
        f"spec = importlib.util.spec_from_file_location('sf_fast_kernels_isolated', {fk_path!r}); "
        "fk = importlib.util.module_from_spec(spec); "
        "sys.modules['sf_fast_kernels_isolated'] = fk; "
        "spec.loader.exec_module(fk); "
        "assert 'triton' not in sys.modules, 'fast_kernels eagerly imported triton!'; "
        "assert not torch.cuda.is_initialized(), 'fast_kernels initialized CUDA on import!'; "
        "x = torch.randn(1, 12, 24); "
        "sc = torch.randn(1, 3, 1, 24); "
        "sh = torch.randn(1, 3, 1, 24); "
        "out = fk.sf_layernorm_adaln(x, sc, sh, num_frames=3, frame_seq=4); "
        "assert out.shape == x.shape and 'triton' not in sys.modules and not torch.cuda.is_initialized()"
    )
    proc1 = subprocess.run([sys.executable, "-c", code_isolated], env=env, capture_output=True, text=True, check=False)
    assert proc1.returncode == 0, f"stdout={proc1.stdout}\nstderr={proc1.stderr}"

    # 2. Full package import with triton absent (CPU-only CI simulation) must succeed cleanly
    code_no_triton = (
        "import sys, torch; "
        "sys.modules['triton'] = None; "
        "import worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing as sf; "
        "assert not torch.cuda.is_initialized(); "
        "head = sf.CausalHead(dim=24, out_dim=4, patch_size=(1, 2, 2)).eval(); "
        "out = head(torch.randn(1, 12, 24), torch.randn(1, 3, 1, 24)); "
        "assert out.shape == (1, 3, 4, 16)"
    )
    proc2 = subprocess.run([sys.executable, "-c", code_no_triton], env=env, capture_output=True, text=True, check=False)
    assert proc2.returncode == 0, f"stdout={proc2.stdout}\nstderr={proc2.stderr}"


def test_cpu_fallback_and_autograd_fallback() -> None:
    """All operators run cleanly on CPU and with autograd enabled."""
    torch.manual_seed(7)
    b, f_frames, hw, n_heads, head_dim = 2, 3, 4, 2, 12
    s, c = f_frames * hw, n_heads * head_dim
    x = torch.randn(b, s, c, requires_grad=True)
    y = torch.randn(b, s, c, requires_grad=True)
    mod6 = torch.randn(b, f_frames, 6, c, requires_grad=True)
    e = mod6.chunk(6, dim=2)
    w = torch.randn(c, requires_grad=True)
    bias = torch.randn(c, requires_grad=True)

    # 1. LayerNorm + AdaLN
    out_ln = sf_layernorm_adaln(x, e[1], e[0], num_frames=f_frames, frame_seq=hw)
    ref_ln = (F.layer_norm(x, (c,), eps=1e-6).unflatten(1, (f_frames, hw)) * (1 + e[1]) + e[0]).flatten(1, 2)
    torch.testing.assert_close(out_ln, ref_ln, rtol=0, atol=0)
    out_ln.sum().backward()
    assert x.grad is not None and mod6.grad is not None

    # 2. Residual gate add + LayerNorm
    x2 = x.detach().clone().requires_grad_(True)
    y2 = y.detach().clone().requires_grad_(True)
    g2 = e[2].detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = bias.detach().clone().requires_grad_(True)
    x_new, norm_out = sf_residual_gate_add_layernorm(x2, y2, g2, w2, b2, num_frames=f_frames, frame_seq=hw)
    ref_x = x2 + (y2.unflatten(1, (f_frames, hw)) * g2).flatten(1, 2)
    ref_n = F.layer_norm(ref_x, (c,), weight=w2, bias=b2, eps=1e-6)
    torch.testing.assert_close(x_new, ref_x, rtol=0, atol=0)
    torch.testing.assert_close(norm_out, ref_n, rtol=0, atol=0)
    (x_new.sum() + norm_out.sum()).backward()
    assert x2.grad is not None and w2.grad is not None

    # 3. Residual add + LayerNorm + AdaLN
    x_new3, ffn_in = sf_residual_add_layernorm_adaln(
        x.detach(), y.detach(), e[4].detach(), e[3].detach(), num_frames=f_frames, frame_seq=hw
    )
    ref_x3 = x.detach() + y.detach()
    ref_ffn = (
        F.layer_norm(ref_x3, (c,), eps=1e-6).unflatten(1, (f_frames, hw)) * (1 + e[4].detach()) + e[3].detach()
    ).flatten(1, 2)
    torch.testing.assert_close(x_new3, ref_x3, rtol=0, atol=0)
    torch.testing.assert_close(ffn_in, ref_ffn, rtol=0, atol=0)

    # 4. Residual gate add & RMSNorm
    rg = sf_residual_gate_add(x.detach(), y.detach(), e[5].detach(), num_frames=f_frames, frame_seq=hw)
    torch.testing.assert_close(
        rg, x.detach() + (y.detach().unflatten(1, (f_frames, hw)) * e[5].detach()).flatten(1, 2), rtol=0, atol=0
    )
    rms = sf_rmsnorm(x.detach(), w.detach(), eps=1e-6)
    ref_rms = (x.detach().float() * torch.rsqrt(x.detach().float().pow(2).mean(-1, keepdim=True) + 1e-6)) * w.detach()
    torch.testing.assert_close(rms, ref_rms, rtol=0, atol=0)

    # 5. QK RMSNorm + RoPE + KV write on CPU
    with torch.no_grad():
        qkv = torch.randn(b, s, 3 * c)
        qp, kp, vp = qkv.chunk(3, dim=-1)
        freqs = torch.cat(
            [
                rope_params(32, head_dim - 4 * (head_dim // 6)),
                rope_params(32, 2 * (head_dim // 6)),
                rope_params(32, 2 * (head_dim // 6)),
            ],
            dim=1,
        )
        cos, sin = _build_3d_rope_cos_sin(freqs, f_frames, 2, 2, start_frame=1, dtype=torch.float64)
        k_cache = torch.zeros(b, s + 4, n_heads, head_dim)
        v_cache = torch.zeros(b, s + 4, n_heads, head_dim)
        q_rot = sf_qk_rmsnorm_rope_write_kv(
            qp,
            kp,
            vp,
            w.detach(),
            bias.detach(),
            cos,
            sin,
            k_cache,
            v_cache,
            dst_start=4,
            num_heads=n_heads,
            head_dim=head_dim,
            rotate_k=True,
        )
        assert q_rot.shape == (b, s, n_heads, head_dim)
        torch.testing.assert_close(v_cache[:, 4 : 4 + s], vp.view(b, s, n_heads, head_dim), rtol=0, atol=0)


def test_env_toggle_controls_fast_kernels_and_exact_norms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLDFOUNDRY_SF_FAST_KERNELS", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_KERNEL_BACKEND", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_SF_EXACT_NORMS", raising=False)
    assert sf_fast_kernels_enabled() is True
    assert sf_exact_norms_enabled(None) is True
    assert sf_exact_norms_enabled("torch") is True
    assert sf_exact_norms_enabled("flash_attention_3") is False

    monkeypatch.setenv("WORLDFOUNDRY_SF_FAST_KERNELS", "0")
    assert sf_fast_kernels_enabled() is False
    assert sf_fast_kernels_enabled(True) is True

    monkeypatch.setenv("WORLDFOUNDRY_SF_FAST_KERNELS", "1")
    monkeypatch.setenv("WORLDFOUNDRY_KERNEL_BACKEND", "torch")
    assert sf_fast_kernels_enabled() is False

    monkeypatch.setenv("WORLDFOUNDRY_SF_EXACT_NORMS", "0")
    assert sf_exact_norms_enabled("torch") is False
    assert sf_exact_norms_enabled("torch", override=True) is True

    monkeypatch.setenv("WORLDFOUNDRY_SF_EXACT_NORMS", "1")
    assert sf_exact_norms_enabled("flash_attention_3") is True
    assert sf_exact_norms_enabled("flash_attention_3", override=False) is False


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("batch,hidden", [(1, 1536), (2, 192)])
@pytest.mark.parametrize("exact", [True, False])
def test_cuda_elementwise_and_superfusion_kernels_parity(
    dtype: torch.dtype,
    batch: int,
    hidden: int,
    exact: bool,
) -> None:
    """Verify every fused Hopper Triton kernel matches exact PyTorch reference (0-ULP when exact=True)."""
    device = torch.device("cuda")
    torch.manual_seed(42 + hidden)
    num_frames, frame_seq = 3, 16
    seq_len = num_frames * frame_seq

    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    y = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    mod6 = torch.randn(batch, num_frames, 6, hidden, device=device, dtype=dtype)
    # Deliberately pass non-contiguous strided chunk views directly!
    e = mod6.chunk(6, dim=2)
    w = torch.randn(hidden, device=device, dtype=dtype)
    b = torch.randn(hidden, device=device, dtype=dtype)

    def _check(got: torch.Tensor, ref: torch.Tensor) -> None:
        if exact:
            assert torch.equal(got, ref), f"exact=True mismatch max_diff={(got.float() - ref.float()).abs().max()}"
        else:
            _assert_lossless_parity(got, ref, dtype=dtype)

    with torch.inference_mode():
        # 1. sf_layernorm_adaln
        got_ln = sf_layernorm_adaln(
            x, e[1], e[0], num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, exact=exact, enabled=True
        )
        ref_ln = sf_layernorm_adaln(x, e[1], e[0], num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, enabled=False)
        _check(got_ln, ref_ln)

        # Broadcast modulation [1, 1, 1, C] and mixed-dtype FP32 scale/shift
        sc_bcast = torch.randn(1, 1, 1, hidden, device=device, dtype=torch.float32)
        sh_bcast = torch.randn(1, num_frames, hidden, device=device, dtype=dtype)
        got_ln_m = sf_layernorm_adaln(
            x, sc_bcast, sh_bcast, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, exact=exact, enabled=True
        )
        ref_ln_m = sf_layernorm_adaln(
            x, sc_bcast, sh_bcast, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, enabled=False
        )
        assert got_ln_m.dtype == torch.float32
        _check(got_ln_m, ref_ln_m)

        # 2. sf_residual_gate_add (always 0-ULP bit-exact!)
        got_rg = sf_residual_gate_add(x, y, e[5], num_frames=num_frames, frame_seq=frame_seq, enabled=True)
        ref_rg = sf_residual_gate_add(x, y, e[5], num_frames=num_frames, frame_seq=frame_seq, enabled=False)
        assert torch.equal(got_rg, ref_rg)

        # 3. sf_residual_gate_add_layernorm (with weight+bias and affine-free)
        got_x1, got_n3 = sf_residual_gate_add_layernorm(
            x, y, e[2], w, b, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, exact=exact, enabled=True
        )
        ref_x1, ref_n3 = sf_residual_gate_add_layernorm(
            x, y, e[2], w, b, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, enabled=False
        )
        assert torch.equal(got_x1, ref_x1)
        _check(got_n3, ref_n3)

        got_x1b, got_n3b = sf_residual_gate_add_layernorm(
            x, y, e[2], None, None, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, exact=exact, enabled=True
        )
        ref_x1b, ref_n3b = sf_residual_gate_add_layernorm(
            x, y, e[2], None, None, num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, enabled=False
        )
        assert torch.equal(got_x1b, ref_x1b)
        _check(got_n3b, ref_n3b)

        # 4. sf_residual_add_layernorm_adaln
        got_x2, got_ffn = sf_residual_add_layernorm_adaln(
            x, y, e[4], e[3], num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, exact=exact, enabled=True
        )
        ref_x2, ref_ffn = sf_residual_add_layernorm_adaln(
            x, y, e[4], e[3], num_frames=num_frames, frame_seq=frame_seq, eps=1e-6, enabled=False
        )
        assert torch.equal(got_x2, ref_x2)
        _check(got_ffn, ref_ffn)

        # 5. sf_rmsnorm (strided QKV view)
        qkv = torch.randn(batch, seq_len, 3 * hidden, device=device, dtype=dtype)
        q_strided = qkv.chunk(3, dim=-1)[0]
        got_rms = sf_rmsnorm(q_strided, w, eps=1e-6, exact=exact, enabled=True)
        ref_rms = sf_rmsnorm(q_strided, w, eps=1e-6, enabled=False)
        _check(got_rms, ref_rms)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rope_dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("rotate_k", [True, False])
@pytest.mark.parametrize("fused_qkv", [True, False])
@pytest.mark.parametrize("exact", [True, False])
def test_cuda_qk_rmsnorm_rope_write_kv_and_apply_rope_parity(
    dtype: torch.dtype,
    rope_dtype: torch.dtype,
    rotate_k: bool,
    fused_qkv: bool,
    exact: bool,
) -> None:
    """Verify fused QK RMSNorm + 3D RoPE + KV cache write and standalone RoPE (0-ULP when exact=True)."""
    device = torch.device("cuda")
    torch.manual_seed(123)
    batch, num_frames, h_grid, w_grid = 2, 3, 4, 4
    seq_len = num_frames * h_grid * w_grid
    num_heads, head_dim = 12, 128
    hidden = num_heads * head_dim
    dst_start = 16
    kv_cap = dst_start + seq_len + 8

    if fused_qkv:
        qkv = torch.randn(batch, seq_len, 3 * hidden, device=device, dtype=dtype)
        q_proj, k_proj, v_proj = qkv.chunk(3, dim=-1)
    else:
        q_proj = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
        k_proj = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
        v_proj = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)

    q_w = torch.randn(hidden, device=device, dtype=dtype)
    k_w = torch.randn(hidden, device=device, dtype=dtype)
    freqs = torch.cat(
        [
            rope_params(64, head_dim - 4 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)
    cos, sin = _build_3d_rope_cos_sin(freqs, num_frames, h_grid, w_grid, start_frame=2, dtype=rope_dtype)

    k_cache_fast = torch.zeros(batch, kv_cap, num_heads, head_dim, device=device, dtype=dtype)
    v_cache_fast = torch.zeros_like(k_cache_fast)
    k_cache_ref = torch.zeros_like(k_cache_fast)
    v_cache_ref = torch.zeros_like(k_cache_fast)

    with torch.inference_mode():
        q_fast = sf_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            v_proj,
            q_w,
            k_w,
            cos,
            sin,
            k_cache_fast,
            v_cache_fast,
            dst_start=dst_start,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            rotate_k=rotate_k,
            exact=exact,
            enabled=True,
        )
        q_ref = sf_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            v_proj,
            q_w,
            k_w,
            cos,
            sin,
            k_cache_ref,
            v_cache_ref,
            dst_start=dst_start,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            rotate_k=rotate_k,
            exact=exact,
            enabled=False,
        )

        if exact:
            assert torch.equal(q_fast, q_ref)
            assert torch.equal(
                k_cache_fast[:, dst_start : dst_start + seq_len],
                k_cache_ref[:, dst_start : dst_start + seq_len],
            )
        else:
            _assert_lossless_parity(q_fast, q_ref, dtype=dtype)
            _assert_lossless_parity(
                k_cache_fast[:, dst_start : dst_start + seq_len],
                k_cache_ref[:, dst_start : dst_start + seq_len],
                dtype=dtype,
            )
        assert torch.equal(v_cache_fast, v_cache_ref)
        # Ensure untouched slots remain zero.
        assert torch.count_nonzero(k_cache_fast[:, :dst_start]) == 0
        assert torch.count_nonzero(k_cache_fast[:, dst_start + seq_len :]) == 0

        # Standalone sf_apply_rope_bshd on strided B=2 slice of k_cache is ALWAYS 0-ULP bit-exact!
        k_slice = k_cache_ref[:, dst_start : dst_start + seq_len]
        rot_fast = sf_apply_rope_bshd(k_slice, cos, sin, enabled=True)
        rot_ref = sf_apply_rope_bshd(k_slice, cos, sin, enabled=False)
        assert torch.equal(rot_fast, rot_ref)
        if rope_dtype == torch.float64:
            grid = torch.tensor([[num_frames, h_grid, w_grid]] * batch, dtype=torch.long)
            rot_complex = causal_rope_apply(k_slice, grid, freqs, start_frame=2).type_as(k_slice)
            assert torch.equal(rot_fast, rot_complex)


@requires_cuda
@pytest.mark.parametrize("kv_cache_relative_rope", [False, True])
@pytest.mark.parametrize("fused_qkv", [False, True])
@pytest.mark.parametrize("cross_attn_norm", [True, False])
@pytest.mark.parametrize("exact_norms", [True, False])
def test_causal_wan_attention_block_and_head_parity(
    kv_cache_relative_rope: bool,
    fused_qkv: bool,
    cross_attn_norm: bool,
    exact_norms: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full CausalWanAttentionBlock + CausalHead forward parity vs PyTorch reference."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(2026)

    dim, ffn_dim, num_heads = 1536, 4096, 12
    head_dim = dim // num_heads
    num_frames, h_grid, w_grid = 3, 6, 8
    frame_seq = h_grid * w_grid
    seq_len = num_frames * frame_seq
    kv_frames = 12

    block = (
        CausalWanAttentionBlock(
            cross_attn_type="t2v_cross_attn",
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            local_attn_size=kv_frames if kv_cache_relative_rope else -1,
            sink_size=5 if kv_cache_relative_rope else 0,
            qk_norm=True,
            cross_attn_norm=cross_attn_norm,
            eps=1e-6,
            kv_cache_relative_rope=kv_cache_relative_rope,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    if fused_qkv:
        assert fuse_qkv_projections(block) == 1

    head = CausalHead(dim=dim, out_dim=16, patch_size=(1, 2, 2), eps=1e-6).to(device=device, dtype=dtype).eval()
    torch.nn.init.normal_(head.head.weight, std=0.02)

    freqs = torch.cat(
        [
            rope_params(64, head_dim - 4 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
            rope_params(64, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)

    x_in = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    e_in = torch.randn(1, num_frames, 6, dim, device=device, dtype=dtype) * 0.1
    e_head = torch.randn(1, num_frames, 1, dim, device=device, dtype=dtype) * 0.1
    context = torch.randn(1, 32, dim, device=device, dtype=dtype)
    grid_sizes = torch.tensor([[num_frames, h_grid, w_grid]], dtype=torch.long)

    def _make_caches():
        kv = {
            "k": torch.zeros(1, kv_frames * frame_seq, num_heads, head_dim, device=device, dtype=dtype),
            "v": torch.zeros(1, kv_frames * frame_seq, num_heads, head_dim, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
        cross = {"is_init": False}
        return kv, cross

    kv_fast, cross_fast = _make_caches()
    kv_ref, cross_ref = _make_caches()

    with torch.inference_mode():
        # Run 2 chunks (chunk 0 initializes crossattn_cache, chunk 1 hits cached fast cross-attn + prefix KV)
        for chunk in range(2):
            cur_start = chunk * seq_len
            local_start = cur_start
            local_end = cur_start + seq_len
            kv_plan = (local_start, local_end, (5 * frame_seq) if kv_cache_relative_rope else 0, 0, 0, 1)
            if kv_cache_relative_rope:
                pre_rope = None
                pre_rel = (
                    _build_3d_rope_cos_sin(freqs, num_frames, h_grid, w_grid, start_frame=chunk * num_frames),
                    _build_3d_rope_cos_sin(freqs, (chunk + 1) * num_frames, h_grid, w_grid, start_frame=0),
                )
            else:
                pre_rope = _build_3d_rope_cos_sin(freqs, num_frames, h_grid, w_grid, start_frame=chunk * num_frames)
                pre_rel = None

            set_sf_fast_kernels_enabled(block, True)
            set_sf_exact_norms(block, exact_norms)
            out_fast = block(
                x_in,
                e_in,
                None,
                grid_sizes,
                freqs,
                context,
                None,
                None,
                kv_cache=kv_fast,
                crossattn_cache=cross_fast,
                current_start=cur_start,
                precomputed_rope=pre_rope,
                precomputed_relative_rope=pre_rel,
                kv_plan=kv_plan,
                frame_seqlen=frame_seq,
            )

            # Reference uses exact PyTorch RoPE (matching complex128 causal_rope_apply) with fast kernels disabled.
            with monkeypatch.context() as mp:
                mp.setattr(sf_mod, "_apply_real_rope", lambda t, c, s: sf_apply_rope_bshd(t, c, s, enabled=False))
                set_sf_fast_kernels_enabled(block, False)
                out_ref = block(
                    x_in,
                    e_in,
                    None,
                    grid_sizes,
                    freqs,
                    context,
                    None,
                    None,
                    kv_cache=kv_ref,
                    crossattn_cache=cross_ref,
                    current_start=cur_start,
                    precomputed_rope=pre_rope,
                    precomputed_relative_rope=pre_rel,
                    kv_plan=kv_plan,
                    frame_seqlen=frame_seq,
                )

            if exact_norms:
                assert torch.equal(out_fast, out_ref)
                assert torch.equal(kv_fast["k"][:, :local_end], kv_ref["k"][:, :local_end])
                assert torch.equal(kv_fast["v"][:, :local_end], kv_ref["v"][:, :local_end])
            else:
                _assert_lossless_parity(out_fast, out_ref, dtype=dtype, min_exact_frac=0.95)
                _assert_lossless_parity(
                    kv_fast["k"][:, :local_end], kv_ref["k"][:, :local_end], dtype=dtype, min_exact_frac=0.95
                )
                _assert_lossless_parity(
                    kv_fast["v"][:, :local_end], kv_ref["v"][:, :local_end], dtype=dtype, min_exact_frac=0.95
                )

        # CausalHead parity (preserving exact 4D output shape [B, F, HW, out_dim])
        set_sf_fast_kernels_enabled(head, True)
        set_sf_exact_norms(head, exact_norms)
        h_fast = head(x_in, e_head)
        set_sf_fast_kernels_enabled(head, False)
        h_ref = head(x_in, e_head)
        assert h_fast.shape == (1, num_frames, frame_seq, 64)
        if exact_norms:
            assert torch.equal(h_fast, h_ref)
        else:
            _assert_lossless_parity(h_fast, h_ref, dtype=dtype)


@requires_cuda
@pytest.mark.parametrize("kv_cache_relative_rope", [False, True])
@pytest.mark.parametrize("exact_norms", [True, False])
def test_cuda_graph_capture_and_replay_full_model(kv_cache_relative_rope: bool, exact_norms: bool) -> None:
    """Verify CUDA Graph capture & replay on CausalWanModel with fused Triton kernels."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(77)

    model = (
        CausalWanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=16,
            dim=256,
            ffn_dim=512,
            freq_dim=64,
            text_dim=32,
            out_dim=16,
            num_heads=4,
            num_layers=2,
            local_attn_size=12 if kv_cache_relative_rope else -1,
            sink_size=5 if kv_cache_relative_rope else 0,
            qk_norm=True,
            cross_attn_norm=True,
            kv_cache_relative_rope=kv_cache_relative_rope,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    fuse_qkv_projections(model)
    set_sf_fast_kernels_enabled(model, True)
    set_sf_exact_norms(model, exact_norms)

    frame_seq = 16  # 8x8 latent with 2x2 patch
    kv_cap = 12 * frame_seq
    kv_cache = [
        {
            "k": torch.zeros(1, kv_cap, 4, 64, device=device, dtype=dtype),
            "v": torch.zeros(1, kv_cap, 4, 64, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
        for _ in range(2)
    ]
    cross_cache = [{"is_init": False} for _ in range(2)]

    static_x = torch.randn(1, 16, 3, 8, 8, device=device, dtype=dtype)
    static_t = torch.full((1, 3), 500.0, device=device, dtype=torch.float32)
    static_ctx = torch.randn(1, 8, 32, device=device, dtype=dtype)

    with torch.inference_mode():
        # Warmup eager pass initializes crossattn_cache and compiles Triton kernels
        eager_out = model(
            static_x,
            t=static_t,
            context=static_ctx,
            seq_len=3 * frame_seq,
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=0,
            cache_start=0,
        )
        # Second eager pass on same slot (steady cross_ready=True)
        model.reset_causal_host_cache()
        eager_steady = model(
            static_x,
            t=static_t,
            context=static_ctx,
            seq_len=3 * frame_seq,
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=0,
            cache_start=0,
        )
        assert torch.equal(eager_out, eager_steady)

        # Capture CUDA Graph
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        model.reset_causal_host_cache()
        with torch.cuda.graph(graph):
            graph_out = model(
                static_x,
                t=static_t,
                context=static_ctx,
                seq_len=3 * frame_seq,
                kv_cache=kv_cache,
                crossattn_cache=cross_cache,
                current_start=0,
                cache_start=0,
            )

        # Replay with new input and verify exact match vs eager
        new_x = torch.randn_like(static_x)
        static_x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        replayed = graph_out.clone()

        model.reset_causal_host_cache()
        expected = model(
            new_x,
            t=static_t,
            context=static_ctx,
            seq_len=3 * frame_seq,
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=0,
            cache_start=0,
        )
        assert torch.equal(replayed, expected)


@requires_cuda
@pytest.mark.parametrize("exact", [True, False])
def test_torch_compile_fullgraph_no_graph_breaks(exact: bool) -> None:
    """Verify every custom op (exact & super-fused) composes with torch.compile(fullgraph=True)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(99)

    dim, num_heads, head_dim = 256, 4, 64
    num_frames, h_grid, w_grid = 3, 4, 4
    frame_seq = h_grid * w_grid
    seq_len = num_frames * frame_seq

    freqs = torch.cat(
        [
            rope_params(32, head_dim - 4 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
        ],
        dim=1,
    ).to(device)
    cos, sin = _build_3d_rope_cos_sin(freqs, num_frames, h_grid, w_grid, start_frame=0, dtype=torch.float64)

    def _fused_pipeline(
        x: torch.Tensor,
        qkv: torch.Tensor,
        y_sa: torch.Tensor,
        y_ca: torch.Tensor,
        y_ffn: torch.Tensor,
        e: torch.Tensor,
        norm3_w: torch.Tensor,
        norm3_b: torch.Tensor,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mods = e.chunk(6, dim=2)
        h1 = sf_layernorm_adaln(
            x, mods[1], mods[0], num_frames=num_frames, frame_seq=frame_seq, exact=exact, enabled=True
        )
        qp, kp, vp = qkv.chunk(3, dim=-1)
        q_rot = sf_qk_rmsnorm_rope_write_kv(
            qp,
            kp,
            vp,
            q_w,
            k_w,
            cos,
            sin,
            k_cache,
            v_cache,
            dst_start=0,
            num_heads=num_heads,
            head_dim=head_dim,
            rotate_k=True,
            exact=exact,
            enabled=True,
        )
        x1, cross_in = sf_residual_gate_add_layernorm(
            x, y_sa, mods[2], norm3_w, norm3_b, num_frames=num_frames, frame_seq=frame_seq, exact=exact, enabled=True
        )
        q_ca = sf_rmsnorm(cross_in, q_w, exact=exact, enabled=True)
        x2, ffn_in = sf_residual_add_layernorm_adaln(
            x1, y_ca, mods[4], mods[3], num_frames=num_frames, frame_seq=frame_seq, exact=exact, enabled=True
        )
        x_out = sf_residual_gate_add(x2, y_ffn, mods[5], num_frames=num_frames, frame_seq=frame_seq, enabled=True)
        k_rerot = sf_apply_rope_bshd(k_cache[:, :seq_len], cos, sin, enabled=True)
        return x_out, h1, ffn_in, q_ca, q_rot, k_rerot

    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    qkv = torch.randn(1, seq_len, 3 * dim, device=device, dtype=dtype)
    y_sa = torch.randn_like(x)
    y_ca = torch.randn_like(x)
    y_ffn = torch.randn_like(x)
    e = torch.randn(1, num_frames, 6, dim, device=device, dtype=dtype)
    norm3_w = torch.randn(dim, device=device, dtype=dtype)
    norm3_b = torch.randn(dim, device=device, dtype=dtype)
    q_w = torch.randn(dim, device=device, dtype=dtype)
    k_w = torch.randn(dim, device=device, dtype=dtype)
    k_eager = torch.zeros(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v_eager = torch.zeros_like(k_eager)
    k_comp = torch.zeros_like(k_eager)
    v_comp = torch.zeros_like(k_eager)

    compiled = torch.compile(_fused_pipeline, fullgraph=True, backend="inductor")
    with torch.inference_mode():
        eager_outs = _fused_pipeline(x, qkv, y_sa, y_ca, y_ffn, e, norm3_w, norm3_b, q_w, k_w, k_eager, v_eager)
        comp_outs = compiled(x, qkv, y_sa, y_ca, y_ffn, e, norm3_w, norm3_b, q_w, k_w, k_comp, v_comp)
    for got, ref in zip(comp_outs, eager_outs, strict=True):
        if exact:
            # Inductor rewrites PyTorch's F.layer_norm / mean reductions (1 BF16 ULP amplified by randn weights).
            torch.testing.assert_close(got, ref, rtol=5e-2, atol=7e-2)
            assert _cosine_similarity(got, ref) > 0.9999
        else:
            assert torch.equal(got, ref)
    if exact:
        torch.testing.assert_close(k_comp, k_eager, rtol=5e-2, atol=7e-2)
        assert _cosine_similarity(k_comp, k_eager) > 0.9999
    else:
        assert torch.equal(k_comp, k_eager)
    assert torch.equal(v_comp, v_eager)
