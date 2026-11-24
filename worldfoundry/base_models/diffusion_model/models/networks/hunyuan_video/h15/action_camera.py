# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Inference-only HY15 Action2V PRoPE transformer; common layers are shared."""

from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange, repeat

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import prope_qkv
from worldfoundry.core.attention import apply_nd_rotary_embedding as apply_rotary_emb
from worldfoundry.core.attention import get_nd_rotary_pos_embed
from worldfoundry.core.attention.backends.probe import normalize_attention_backend
from worldfoundry.core.attention.sequence.multimodal_sequence import parallel_attention
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel.communication_op import sequence_model_parallel_all_gather
from worldfoundry.core.nn import ConditioningProjection as TextProjection
from worldfoundry.core.nn import DiTModulation as ModulateDiT
from worldfoundry.core.nn import MLPEmbedder
from worldfoundry.core.nn import SinusoidalTimestepEmbedder as TimestepEmbedder
from worldfoundry.core.nn import TransformerMLP as MLP
from worldfoundry.core.nn import activation_layer as get_activation_layer
from worldfoundry.core.nn import normalization_layer as get_norm_layer

from .camera_attention import (
    hopper_fa3_usable,
    prepare_static_fp8_kv_arena,
    prepare_static_kv_arena,
    prepare_static_kv_arena_bshd,
    sequence_parallel_attention_bi_fa3,
    sequence_parallel_attention_bi_fp8,
    sequence_parallel_attention_txt,
    sequence_parallel_attention_vision,
)
from .camera_modulation import (
    FinalLayer,
    apply_gate,
    fused_layernorm_modulate,
    fused_residual_dual_gate,
    fused_residual_gate,
    modulate,
)
from .hunyuanvideo_1_5_transformer import _fuse_three_linears, _run_hy15_mlp
from .modules.embed_layers import ByT5Mapper, PatchEmbed, VisionProjection
from .modules.token_refiner import SingleTokenRefiner


def _precompute_prope_mats(
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(P, P_inv^T, P^T)`` once per forward pass over unique cameras ``[B, C, 4, 4]``."""
    # Invert SE(3) in float32 for precision then cast back to viewmats.dtype.
    vm = viewmats.float()
    r_inv = vm[..., :3, :3].transpose(-1, -2)
    vm_inv = torch.zeros_like(vm)
    vm_inv[..., :3, :3] = r_inv
    vm_inv[..., :3, 3] = -torch.matmul(r_inv, vm[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    vm_inv[..., 3, 3] = 1.0

    if Ks is not None:
        ks_f = Ks.float()
        fx = ks_f[..., 0, 0]
        fy = ks_f[..., 1, 1]
        ones = torch.ones_like(fx)
        zeros = torch.zeros_like(fx)
        lift_k = torch.zeros_like(vm)
        lift_k[..., 0, 0] = fx
        lift_k[..., 1, 1] = fy
        lift_k[..., 2, 2] = ones
        lift_k[..., 3, 3] = ones
        p = torch.matmul(lift_k, vm)

        lift_inv_k = torch.zeros_like(vm)
        lift_inv_k[..., 0, 0] = ones / fx
        lift_inv_k[..., 1, 1] = ones / fy
        lift_inv_k[..., 2, 2] = ones
        lift_inv_k[..., 3, 3] = ones
        del zeros
        p_inv = torch.matmul(vm_inv, lift_inv_k)
    else:
        p = vm
        p_inv = vm_inv

    p = p.contiguous()
    p_inv_t = p_inv.transpose(-1, -2).contiguous()
    p_t = p.transpose(-1, -2).contiguous()
    return p, p_inv_t, p_t


def _apply_fast_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved real-valued 3D RoPE using core's fused Triton kernel on CUDA."""
    cos, sin = freqs_cis
    if (
        q.is_cuda
        and q.ndim == 4
        and cos.ndim == 2
        and cos.shape[0] == q.shape[1]
        and cos.shape[1] == q.shape[-1]
        and not torch.is_grad_enabled()
    ):
        from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd

        cos_half = cos[:, 0::2].unsqueeze(0).unsqueeze(2)
        sin_half = sin[:, 0::2].unsqueeze(0).unsqueeze(2)
        return apply_real_rope_bshd(q, cos_half, sin_half), apply_real_rope_bshd(k, cos_half, sin_half)
    return apply_rotary_emb(q, k, freqs_cis, head_first=False)


try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as _libdevice

    @triton.jit
    def _hy_qkv_rmsnorm_rope_prope_kernel(
        qkv_ptr,
        q_w_ptr,
        k_w_ptr,
        cos_ptr,
        sin_ptr,
        p_ptr,
        pinvt_ptr,
        q_rope_ptr,
        k_rope_ptr,
        v_out_ptr,
        q_prope_ptr,
        k_prope_ptr,
        v_prope_ptr,
        seq_len,
        tokens_per_cam,
        num_cams,
        qkv_row_stride,
        cos_row_stride,
        eps,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        NUM_QUADS: tl.constexpr,
    ):
        """Fuse QKV split + per-head RMSNorm + interleaved FP32 RoPE + 4x4 SE(3) PRoPE in ONE pass.

        Grid: ``(B * S, NUM_HEADS)``. Collapses 8 separate HBM-bound kernels
        (``q_norm``, ``k_norm``, 3x ``_apply_prope_mat``, 2x RoPE, ``v.contiguous``)
        into a single register-resident pass using 4-element PRoPE quads.
        """
        row = tl.program_id(0)
        head = tl.program_id(1)
        b_idx = row // seq_len
        tok_idx = row % seq_len
        cam_id = b_idx * num_cams + (tok_idx // tokens_per_cam)

        quad = tl.arange(0, NUM_QUADS)
        c0_col = quad * 4 + 0
        c1_col = quad * 4 + 1
        c2_col = quad * 4 + 2
        c3_col = quad * 4 + 3

        hidden = NUM_HEADS * HEAD_DIM
        row_base = row.to(tl.int64) * qkv_row_stride
        q_base = row_base + head.to(tl.int64) * HEAD_DIM
        k_base = row_base + hidden + head.to(tl.int64) * HEAD_DIM
        v_base = row_base + 2 * hidden + head.to(tl.int64) * HEAD_DIM

        q0_raw = tl.load(qkv_ptr + q_base + c0_col)
        q1_raw = tl.load(qkv_ptr + q_base + c1_col)
        q2_raw = tl.load(qkv_ptr + q_base + c2_col)
        q3_raw = tl.load(qkv_ptr + q_base + c3_col)

        k0_raw = tl.load(qkv_ptr + k_base + c0_col)
        k1_raw = tl.load(qkv_ptr + k_base + c1_col)
        k2_raw = tl.load(qkv_ptr + k_base + c2_col)
        k3_raw = tl.load(qkv_ptr + k_base + c3_col)

        v0_raw = tl.load(qkv_ptr + v_base + c0_col)
        v1_raw = tl.load(qkv_ptr + v_base + c1_col)
        v2_raw = tl.load(qkv_ptr + v_base + c2_col)
        v3_raw = tl.load(qkv_ptr + v_base + c3_col)

        # Per-head FP32 RMSNorm + weight -> cast to v_dtype (exact HY15 RMSNorm semantics)
        q0_f, q1_f, q2_f, q3_f = (
            q0_raw.to(tl.float32),
            q1_raw.to(tl.float32),
            q2_raw.to(tl.float32),
            q3_raw.to(tl.float32),
        )
        k0_f, k1_f, k2_f, k3_f = (
            k0_raw.to(tl.float32),
            k1_raw.to(tl.float32),
            k2_raw.to(tl.float32),
            k3_raw.to(tl.float32),
        )

        q_rstd = _libdevice.rsqrt(
            tl.sum(q0_f * q0_f + q1_f * q1_f + q2_f * q2_f + q3_f * q3_f, axis=0) / HEAD_DIM + eps
        )
        k_rstd = _libdevice.rsqrt(
            tl.sum(k0_f * k0_f + k1_f * k1_f + k2_f * k2_f + k3_f * k3_f, axis=0) / HEAD_DIM + eps
        )

        qw0 = tl.load(q_w_ptr + c0_col).to(tl.float32)
        qw1 = tl.load(q_w_ptr + c1_col).to(tl.float32)
        qw2 = tl.load(q_w_ptr + c2_col).to(tl.float32)
        qw3 = tl.load(q_w_ptr + c3_col).to(tl.float32)
        kw0 = tl.load(k_w_ptr + c0_col).to(tl.float32)
        kw1 = tl.load(k_w_ptr + c1_col).to(tl.float32)
        kw2 = tl.load(k_w_ptr + c2_col).to(tl.float32)
        kw3 = tl.load(k_w_ptr + c3_col).to(tl.float32)

        qn0 = (q0_f * q_rstd * qw0).to(v0_raw.dtype).to(tl.float32)
        qn1 = (q1_f * q_rstd * qw1).to(v0_raw.dtype).to(tl.float32)
        qn2 = (q2_f * q_rstd * qw2).to(v0_raw.dtype).to(tl.float32)
        qn3 = (q3_f * q_rstd * qw3).to(v0_raw.dtype).to(tl.float32)
        kn0 = (k0_f * k_rstd * kw0).to(v0_raw.dtype).to(tl.float32)
        kn1 = (k1_f * k_rstd * kw1).to(v0_raw.dtype).to(tl.float32)
        kn2 = (k2_f * k_rstd * kw2).to(v0_raw.dtype).to(tl.float32)
        kn3 = (k3_f * k_rstd * kw3).to(v0_raw.dtype).to(tl.float32)

        out_base = (row.to(tl.int64) * NUM_HEADS + head.to(tl.int64)) * HEAD_DIM

        # 1. RoPE stream: interleaved pairs (0,1) and (2,3)
        cs_base = tok_idx.to(tl.int64) * cos_row_stride
        cos0 = tl.load(cos_ptr + cs_base + c0_col).to(tl.float32)
        sin0 = tl.load(sin_ptr + cs_base + c0_col).to(tl.float32)
        cos2 = tl.load(cos_ptr + cs_base + c2_col).to(tl.float32)
        sin2 = tl.load(sin_ptr + cs_base + c2_col).to(tl.float32)

        tl.store(q_rope_ptr + out_base + c0_col, (qn0 * cos0 - qn1 * sin0).to(v0_raw.dtype))
        tl.store(q_rope_ptr + out_base + c1_col, (qn0 * sin0 + qn1 * cos0).to(v0_raw.dtype))
        tl.store(q_rope_ptr + out_base + c2_col, (qn2 * cos2 - qn3 * sin2).to(v0_raw.dtype))
        tl.store(q_rope_ptr + out_base + c3_col, (qn2 * sin2 + qn3 * cos2).to(v0_raw.dtype))

        tl.store(k_rope_ptr + out_base + c0_col, (kn0 * cos0 - kn1 * sin0).to(v0_raw.dtype))
        tl.store(k_rope_ptr + out_base + c1_col, (kn0 * sin0 + kn1 * cos0).to(v0_raw.dtype))
        tl.store(k_rope_ptr + out_base + c2_col, (kn2 * cos2 - kn3 * sin2).to(v0_raw.dtype))
        tl.store(k_rope_ptr + out_base + c3_col, (kn2 * sin2 + kn3 * cos2).to(v0_raw.dtype))

        tl.store(v_out_ptr + out_base + c0_col, v0_raw)
        tl.store(v_out_ptr + out_base + c1_col, v1_raw)
        tl.store(v_out_ptr + out_base + c2_col, v2_raw)
        tl.store(v_out_ptr + out_base + c3_col, v3_raw)

        # 2. PRoPE stream: 4x4 SE(3) camera matrices p_mat (Q) and p_inv_t (K, V)
        bm = cam_id.to(tl.int64) * 16
        p00 = tl.load(p_ptr + bm + 0)
        p01 = tl.load(p_ptr + bm + 1)
        p02 = tl.load(p_ptr + bm + 2)
        p03 = tl.load(p_ptr + bm + 3)
        p10 = tl.load(p_ptr + bm + 4)
        p11 = tl.load(p_ptr + bm + 5)
        p12 = tl.load(p_ptr + bm + 6)
        p13 = tl.load(p_ptr + bm + 7)
        p20 = tl.load(p_ptr + bm + 8)
        p21 = tl.load(p_ptr + bm + 9)
        p22 = tl.load(p_ptr + bm + 10)
        p23 = tl.load(p_ptr + bm + 11)
        p30 = tl.load(p_ptr + bm + 12)
        p31 = tl.load(p_ptr + bm + 13)
        p32 = tl.load(p_ptr + bm + 14)
        p33 = tl.load(p_ptr + bm + 15)

        tl.store(q_prope_ptr + out_base + c0_col, (qn0 * p00 + qn1 * p10 + qn2 * p20 + qn3 * p30).to(v0_raw.dtype))
        tl.store(q_prope_ptr + out_base + c1_col, (qn0 * p01 + qn1 * p11 + qn2 * p21 + qn3 * p31).to(v0_raw.dtype))
        tl.store(q_prope_ptr + out_base + c2_col, (qn0 * p02 + qn1 * p12 + qn2 * p22 + qn3 * p32).to(v0_raw.dtype))
        tl.store(q_prope_ptr + out_base + c3_col, (qn0 * p03 + qn1 * p13 + qn2 * p23 + qn3 * p33).to(v0_raw.dtype))

        m00 = tl.load(pinvt_ptr + bm + 0)
        m01 = tl.load(pinvt_ptr + bm + 1)
        m02 = tl.load(pinvt_ptr + bm + 2)
        m03 = tl.load(pinvt_ptr + bm + 3)
        m10 = tl.load(pinvt_ptr + bm + 4)
        m11 = tl.load(pinvt_ptr + bm + 5)
        m12 = tl.load(pinvt_ptr + bm + 6)
        m13 = tl.load(pinvt_ptr + bm + 7)
        m20 = tl.load(pinvt_ptr + bm + 8)
        m21 = tl.load(pinvt_ptr + bm + 9)
        m22 = tl.load(pinvt_ptr + bm + 10)
        m23 = tl.load(pinvt_ptr + bm + 11)
        m30 = tl.load(pinvt_ptr + bm + 12)
        m31 = tl.load(pinvt_ptr + bm + 13)
        m32 = tl.load(pinvt_ptr + bm + 14)
        m33 = tl.load(pinvt_ptr + bm + 15)

        tl.store(k_prope_ptr + out_base + c0_col, (kn0 * m00 + kn1 * m10 + kn2 * m20 + kn3 * m30).to(v0_raw.dtype))
        tl.store(k_prope_ptr + out_base + c1_col, (kn0 * m01 + kn1 * m11 + kn2 * m21 + kn3 * m31).to(v0_raw.dtype))
        tl.store(k_prope_ptr + out_base + c2_col, (kn0 * m02 + kn1 * m12 + kn2 * m22 + kn3 * m32).to(v0_raw.dtype))
        tl.store(k_prope_ptr + out_base + c3_col, (kn0 * m03 + kn1 * m13 + kn2 * m23 + kn3 * m33).to(v0_raw.dtype))

        v0_f, v1_f, v2_f, v3_f = (
            v0_raw.to(tl.float32),
            v1_raw.to(tl.float32),
            v2_raw.to(tl.float32),
            v3_raw.to(tl.float32),
        )
        tl.store(v_prope_ptr + out_base + c0_col, (v0_f * m00 + v1_f * m10 + v2_f * m20 + v3_f * m30).to(v0_raw.dtype))
        tl.store(v_prope_ptr + out_base + c1_col, (v0_f * m01 + v1_f * m11 + v2_f * m21 + v3_f * m31).to(v0_raw.dtype))
        tl.store(v_prope_ptr + out_base + c2_col, (v0_f * m02 + v1_f * m12 + v2_f * m22 + v3_f * m32).to(v0_raw.dtype))
        tl.store(v_prope_ptr + out_base + c3_col, (v0_f * m03 + v1_f * m13 + v2_f * m23 + v3_f * m33).to(v0_raw.dtype))

    @triton.jit
    def _prope_mat4_kernel(
        x_ptr,
        m_ptr,
        out_ptr,
        vecs_per_cam,
        BLOCK_VECS: tl.constexpr,
    ):
        cam_id = tl.program_id(0)
        pid_v = tl.program_id(1)
        vec_off = pid_v * BLOCK_VECS + tl.arange(0, BLOCK_VECS)
        mask = vec_off < vecs_per_cam
        base = (cam_id.to(tl.int64) * vecs_per_cam + vec_off.to(tl.int64)) * 4
        x0 = tl.load(x_ptr + base + 0, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + base + 3, mask=mask, other=0.0).to(tl.float32)

        bm = cam_id.to(tl.int64) * 16
        m00 = tl.load(m_ptr + bm + 0)
        m01 = tl.load(m_ptr + bm + 1)
        m02 = tl.load(m_ptr + bm + 2)
        m03 = tl.load(m_ptr + bm + 3)
        m10 = tl.load(m_ptr + bm + 4)
        m11 = tl.load(m_ptr + bm + 5)
        m12 = tl.load(m_ptr + bm + 6)
        m13 = tl.load(m_ptr + bm + 7)
        m20 = tl.load(m_ptr + bm + 8)
        m21 = tl.load(m_ptr + bm + 9)
        m22 = tl.load(m_ptr + bm + 10)
        m23 = tl.load(m_ptr + bm + 11)
        m30 = tl.load(m_ptr + bm + 12)
        m31 = tl.load(m_ptr + bm + 13)
        m32 = tl.load(m_ptr + bm + 14)
        m33 = tl.load(m_ptr + bm + 15)

        y0 = x0 * m00 + x1 * m10 + x2 * m20 + x3 * m30
        y1 = x0 * m01 + x1 * m11 + x2 * m21 + x3 * m31
        y2 = x0 * m02 + x1 * m12 + x2 * m22 + x3 * m32
        y3 = x0 * m03 + x1 * m13 + x2 * m23 + x3 * m33

        tl.store(out_ptr + base + 0, y0, mask=mask)
        tl.store(out_ptr + base + 1, y1, mask=mask)
        tl.store(out_ptr + base + 2, y2, mask=mask)
        tl.store(out_ptr + base + 3, y3, mask=mask)

    _TRITON_PROPE_AVAILABLE = True
except Exception:
    _TRITON_PROPE_AVAILABLE = False


def _apply_prope_mat(tensor: torch.Tensor, mat_t: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal 4x4 camera transform ``mat_t`` to ``[B, L, ...]`` at full HBM bandwidth."""
    batch = tensor.shape[0]
    cams = mat_t.shape[1]
    total_cams = batch * cams
    if (
        _TRITON_PROPE_AVAILABLE
        and tensor.is_cuda
        and not torch.is_grad_enabled()
        and mat_t.ndim == 4
        and mat_t.shape[0] == batch
        and mat_t.shape[-2:] == (4, 4)
        and total_cams > 0
        and tensor.numel() % (total_cams * 4) == 0
    ):
        x_flat = tensor if tensor.is_contiguous() else tensor.contiguous()
        m_f32 = (
            mat_t
            if (mat_t.dtype == torch.float32 and mat_t.device == tensor.device and mat_t.is_contiguous())
            else mat_t.to(device=tensor.device, dtype=torch.float32).contiguous()
        )
        out = torch.empty_like(x_flat)
        vecs_per_cam = x_flat.numel() // (total_cams * 4)
        block_vecs = 256
        grid = (total_cams, triton.cdiv(vecs_per_cam, block_vecs))
        _prope_mat4_kernel[grid](x_flat, m_f32, out, vecs_per_cam, BLOCK_VECS=block_vecs, num_warps=4)
        return out
    reshaped = tensor.reshape(batch, cams, -1, 4)
    out = torch.matmul(reshaped, mat_t.to(dtype=tensor.dtype))
    return out.reshape(tensor.shape)


def _fused_qkv_rmsnorm_rope_prope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    p_mat: torch.Tensor,
    p_inv_t: torch.Tensor,
    *,
    heads_num: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Single-pass Triton QKV split + per-head RMSNorm + interleaved RoPE + 4x4 SE(3) PRoPE."""
    if not (
        _TRITON_PROPE_AVAILABLE
        and qkv.is_cuda
        and not torch.is_grad_enabled()
        and qkv.ndim == 3
        and qkv.is_contiguous()
    ):
        return None
    batch, seq_len, total_dim = int(qkv.shape[0]), int(qkv.shape[1]), int(qkv.shape[2])
    if total_dim % 3 != 0 or heads_num <= 0:
        return None
    hidden = total_dim // 3
    if hidden % heads_num != 0:
        return None
    head_dim = hidden // heads_num
    if p_mat.ndim != 4 or p_inv_t.ndim != 4 or p_mat.shape[0] != batch or p_inv_t.shape[0] != batch:
        return None
    num_cams = int(p_mat.shape[1])
    cos, sin = freqs_cis
    if (
        head_dim not in {32, 64, 128}
        or num_cams <= 0
        or p_inv_t.shape[1] != num_cams
        or p_mat.shape[-2:] != (4, 4)
        or p_inv_t.shape[-2:] != (4, 4)
        or seq_len <= 0
        or seq_len % num_cams != 0
        or not q_weight.is_cuda
        or not k_weight.is_cuda
        or q_weight.device != qkv.device
        or k_weight.device != qkv.device
        or q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or cos.device != qkv.device
        or sin.device != qkv.device
        or cos.ndim != 2
        or sin.shape != cos.shape
        or cos.shape[0] != seq_len
        or cos.shape[1] != head_dim
    ):
        return None
    tokens_per_cam = seq_len // num_cams
    out_shape = (batch, seq_len, heads_num, head_dim)
    q_rope = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    k_rope = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    q_prope = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    k_prope = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    v_prope = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)

    p_f32 = (
        p_mat
        if (p_mat.dtype == torch.float32 and p_mat.device == qkv.device and p_mat.is_contiguous())
        else p_mat.to(device=qkv.device, dtype=torch.float32).contiguous()
    )
    pinvt_f32 = (
        p_inv_t
        if (p_inv_t.dtype == torch.float32 and p_inv_t.device == qkv.device and p_inv_t.is_contiguous())
        else p_inv_t.to(device=qkv.device, dtype=torch.float32).contiguous()
    )
    qw_c = q_weight if q_weight.is_contiguous() else q_weight.contiguous()
    kw_c = k_weight if k_weight.is_contiguous() else k_weight.contiguous()
    cos_c = cos if cos.is_contiguous() else cos.contiguous()
    sin_c = sin if sin.is_contiguous() else sin.contiguous()
    _hy_qkv_rmsnorm_rope_prope_kernel[(batch * seq_len, heads_num)](
        qkv,
        qw_c,
        kw_c,
        cos_c,
        sin_c,
        p_f32,
        pinvt_f32,
        q_rope,
        k_rope,
        v_out,
        q_prope,
        k_prope,
        v_prope,
        seq_len,
        tokens_per_cam,
        num_cams,
        int(qkv.stride(1)),
        int(cos_c.stride(0)),
        float(eps),
        NUM_HEADS=heads_num,
        HEAD_DIM=head_dim,
        NUM_QUADS=head_dim // 4,
        num_warps=2 if head_dim <= 64 else 4,
    )
    return q_rope, k_rope, v_out, q_prope, k_prope, v_prope


class MMDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str | None = None,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self.heads_num = heads_num
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.factory_kwargs = factory_kwargs
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-06, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )
        self.hybrid_seq_parallel_attn = None
        self.attention_backend: str | None = attn_mode or "torch"

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind the request-scoped attention provider on this action-camera block."""
        resolved = "torch" if backend is None else normalize_attention_backend(str(backend))
        self.attention_backend = resolved
        self.attn_mode = resolved

    def fuse_qkv(self) -> int:
        """Fuse separate image and text Q/K/V linear projections in place."""
        fused_count = 0
        if not hasattr(self, "img_attn_qkv") and all(
            isinstance(getattr(self, name, None), nn.Linear) for name in ("img_attn_q", "img_attn_k", "img_attn_v")
        ):
            self.img_attn_qkv = _fuse_three_linears(self.img_attn_q, self.img_attn_k, self.img_attn_v)
            del self.img_attn_q, self.img_attn_k, self.img_attn_v
            fused_count += 1
        if not hasattr(self, "txt_attn_qkv") and all(
            isinstance(getattr(self, name, None), nn.Linear) for name in ("txt_attn_q", "txt_attn_k", "txt_attn_v")
        ):
            self.txt_attn_qkv = _fuse_three_linears(self.txt_attn_q, self.txt_attn_k, self.txt_attn_v)
            del self.txt_attn_q, self.txt_attn_k, self.txt_attn_v
            fused_count += 1
        return fused_count

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def modulate_txt(self, vec_txt, txt):
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(
            vec_txt
        ).chunk(6, dim=-1)
        txt_modulated = fused_layernorm_modulate(
            txt, shift=txt_mod1_shift, scale=txt_mod1_scale, eps=self.txt_norm1.eps
        )
        if hasattr(self, "txt_attn_qkv"):
            txt_q, txt_k, txt_v = self.txt_attn_qkv(txt_modulated).chunk(3, dim=-1)
        else:
            txt_q = self.txt_attn_q(txt_modulated)
            txt_k = self.txt_attn_k(txt_modulated)
            txt_v = self.txt_attn_v(txt_modulated)
        head_dim = txt_q.shape[-1] // self.heads_num
        txt_q = txt_q.view(txt_q.shape[0], txt_q.shape[1], self.heads_num, head_dim)
        txt_k = txt_k.view(txt_k.shape[0], txt_k.shape[1], self.heads_num, head_dim)
        txt_v = txt_v.view(txt_v.shape[0], txt_v.shape[1], self.heads_num, head_dim)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        return (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)

    def modulate_img(self, vec, img):
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(
            vec
        ).chunk(6, dim=-1)
        img_modulated = fused_layernorm_modulate(
            img, shift=img_mod1_shift, scale=img_mod1_scale, eps=self.img_norm1.eps
        )
        if hasattr(self, "img_attn_qkv"):
            img_q, img_k, img_v = self.img_attn_qkv(img_modulated).chunk(3, dim=-1)
        else:
            img_q = self.img_attn_q(img_modulated)
            img_k = self.img_attn_k(img_modulated)
            img_v = self.img_attn_v(img_modulated)
        head_dim = img_q.shape[-1] // self.heads_num
        img_q = img_q.view(img_q.shape[0], img_q.shape[1], self.heads_num, head_dim)
        img_k = img_k.view(img_k.shape[0], img_k.shape[1], self.heads_num, head_dim)
        img_v = img_v.view(img_v.shape[0], img_v.shape[1], self.heads_num, head_dim)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        return (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate)

    def _img_qkv_rope_prope(
        self,
        vec: torch.Tensor,
        img: torch.Tensor,
        freqs_cis: tuple | None,
        prope_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        viewmats: torch.Tensor | None,
        Ks: torch.Tensor | None,
    ):
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(
            vec
        ).chunk(6, dim=-1)
        img_modulated = fused_layernorm_modulate(
            img, shift=img_mod1_shift, scale=img_mod1_scale, eps=self.img_norm1.eps
        )
        if hasattr(self, "img_attn_qkv"):
            qkv = self.img_attn_qkv(img_modulated)
            q_weight = getattr(self.img_attn_q_norm, "weight", None)
            k_weight = getattr(self.img_attn_k_norm, "weight", None)
            q_eps = float(getattr(self.img_attn_q_norm, "eps", 1e-6))
            k_eps = float(getattr(self.img_attn_k_norm, "eps", 1e-6))
            is_rms_qk = (
                "rms" in type(self.img_attn_q_norm).__name__.casefold()
                and "rms" in type(self.img_attn_k_norm).__name__.casefold()
                and getattr(self.img_attn_q_norm, "bias", None) is None
                and getattr(self.img_attn_k_norm, "bias", None) is None
            )
            if (
                prope_mats is not None
                and freqs_cis is not None
                and is_rms_qk
                and q_weight is not None
                and k_weight is not None
                and q_eps == k_eps
                and getattr(self.img_attn_q_norm, "compute_mode", "fp32") == "fp32"
                and getattr(self.img_attn_k_norm, "compute_mode", "fp32") == "fp32"
            ):
                p_mat, p_inv_t, p_t = prope_mats
                fused = _fused_qkv_rmsnorm_rope_prope(
                    qkv,
                    q_weight,
                    k_weight,
                    freqs_cis,
                    p_mat,
                    p_inv_t,
                    heads_num=self.heads_num,
                    eps=q_eps,
                )
                if fused is not None:
                    img_q, img_k, img_v, img_q_prope, img_k_prope, img_v_prope = fused
                    return (
                        img_q,
                        img_k,
                        img_v,
                        img_q_prope,
                        img_k_prope,
                        img_v_prope,
                        p_t,
                        None,
                        img_mod1_gate,
                        img_mod2_shift,
                        img_mod2_scale,
                        img_mod2_gate,
                    )
            img_q, img_k, img_v = qkv.chunk(3, dim=-1)
        else:
            img_q = self.img_attn_q(img_modulated)
            img_k = self.img_attn_k(img_modulated)
            img_v = self.img_attn_v(img_modulated)
        head_dim = img_q.shape[-1] // self.heads_num
        img_q = img_q.view(img_q.shape[0], img_q.shape[1], self.heads_num, head_dim)
        img_k = img_k.view(img_k.shape[0], img_k.shape[1], self.heads_num, head_dim)
        img_v = img_v.view(img_v.shape[0], img_v.shape[1], self.heads_num, head_dim)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        if prope_mats is not None:
            p_mat, p_inv_t, p_t = prope_mats
            img_q_prope = _apply_prope_mat(img_q, p_mat)
            img_k_prope = _apply_prope_mat(img_k, p_inv_t)
            img_v_prope = _apply_prope_mat(img_v, p_inv_t)
            apply_fn_o = None
        else:
            (img_q_prope, img_k_prope, img_v_prope, apply_fn_o) = prope_qkv(
                img_q.permute(0, 2, 1, 3),
                img_k.permute(0, 2, 1, 3),
                img_v.permute(0, 2, 1, 3),
                viewmats=viewmats,
                Ks=Ks,
            )
            img_q_prope = img_q_prope.permute(0, 2, 1, 3)
            img_k_prope = img_k_prope.permute(0, 2, 1, 3)
            img_v_prope = img_v_prope.permute(0, 2, 1, 3)
            p_t = None
        if freqs_cis is not None:
            (img_q, img_k) = _apply_fast_interleaved_rope(img_q, img_k, freqs_cis)
        return (
            img_q,
            img_k,
            img_v,
            img_q_prope,
            img_k_prope,
            img_v_prope,
            p_t,
            apply_fn_o,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        )

    def forward_txt(
        self,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        kv_cache: dict | None = None,
        cache_txt: bool = False,
    ) -> tuple[torch.Tensor]:
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        attn_mode = self.attention_backend or "torch_causal"
        (txt_attn, t_kv) = sequence_parallel_attention_txt(
            txt_q,
            txt_k,
            txt_v,
            img_q_len=txt_q.shape[1],
            img_kv_len=txt_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_txt=cache_txt,
        )
        txt = fused_residual_gate(txt, self.txt_attn_proj(txt_attn), txt_mod1_gate)
        txt = fused_residual_gate(
            txt,
            _run_hy15_mlp(
                self.txt_mlp,
                fused_layernorm_modulate(txt, shift=txt_mod2_shift, scale=txt_mod2_scale, eps=self.txt_norm2.eps),
            ),
            txt_mod2_gate,
        )
        return (txt, t_kv)

    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        attn_param=None,
        block_idx=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        cache_vision: bool = False,
        prope_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        static_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        static_bshd_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        static_prefix_len: int | None = None,
        static_fp8_rope_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        static_fp8_prope_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
        num_blocks: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            img_q,
            img_k,
            img_v,
            img_q_prope,
            img_k_prope,
            img_v_prope,
            p_t,
            apply_fn_o,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self._img_qkv_rope_prope(vec, img, freqs_cis, prope_mats, viewmats, Ks)
        attn_mode = self.attention_backend or self.attn_mode
        (img_attn, vision_kv) = sequence_parallel_attention_vision(
            img_q,
            img_k,
            img_v,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_vision=cache_vision,
            attn_mode=attn_mode,
            static_kv_buffer=static_kv_buffer,
            static_bshd_kv_buffer=static_bshd_kv_buffer,
            static_prefix_len=static_prefix_len,
            static_fp8_kv_buffer=static_fp8_rope_buffer,
            stream_tag="rope",
            num_blocks=num_blocks,
        )
        (img_attn_prope, _) = sequence_parallel_attention_vision(
            img_q_prope,
            img_k_prope,
            img_v_prope,
            block_idx=block_idx,
            kv_cache=kv_cache,
            cache_vision=False,
            attn_mode=attn_mode,
            static_kv_buffer=static_kv_buffer,
            static_bshd_kv_buffer=static_bshd_kv_buffer,
            static_prefix_len=static_prefix_len,
            static_fp8_kv_buffer=static_fp8_prope_buffer,
            stream_tag="prope",
            num_blocks=num_blocks,
        )
        if p_t is not None:
            img_attn_prope = _apply_prope_mat(img_attn_prope, p_t)
        else:
            img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
            img_attn_prope = apply_fn_o(img_attn_prope)
            img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        img = fused_residual_dual_gate(
            img, self.img_attn_proj(img_attn), self.img_attn_prope_proj(img_attn_prope), img_mod1_gate
        )
        img = fused_residual_gate(
            img,
            _run_hy15_mlp(
                self.img_mlp,
                fused_layernorm_modulate(img, shift=img_mod2_shift, scale=img_mod2_scale, eps=self.img_norm2.eps),
            ),
            img_mod2_gate,
        )
        return (img, vision_kv)

    def forward_bi(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        prope_mats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        num_blocks: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        (
            img_q,
            img_k,
            img_v,
            img_q_prope,
            img_k_prope,
            img_v_prope,
            p_t,
            apply_fn_o,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self._img_qkv_rope_prope(vec, img, freqs_cis, prope_mats, viewmats, Ks)
        attn_mode = "flash" if is_flash else (self.attention_backend or self.attn_mode)
        if attn_mode == "cudnn_fp8" and text_mask is None:
            attn = sequence_parallel_attention_bi_fp8(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                block_idx=block_idx,
                num_blocks=num_blocks,
                stream_tag="rope",
            )
            (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
            attn_prope = sequence_parallel_attention_bi_fp8(
                (img_q_prope, txt_q),
                (img_k_prope, txt_k),
                (img_v_prope, txt_v),
                block_idx=block_idx,
                num_blocks=num_blocks,
                stream_tag="prope",
            )
        elif attn_mode == "flash_attention_3" and text_mask is None:
            attn = sequence_parallel_attention_bi_fa3(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
            )
            (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
            attn_prope = sequence_parallel_attention_bi_fa3(
                (img_q_prope, txt_q),
                (img_k_prope, txt_k),
                (img_v_prope, txt_v),
            )
        else:
            eff_attn_mode = "torch" if attn_mode == "cudnn_fp8" else attn_mode
            attn = parallel_attention(
                (img_q, txt_q),
                (img_k, txt_k),
                (img_v, txt_v),
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                text_mask=text_mask,
                attn_mode=eff_attn_mode,
                attn_param=attn_param,
                block_idx=block_idx,
            )
            (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
            attn_prope = parallel_attention(
                (img_q_prope, txt_q),
                (img_k_prope, txt_k),
                (img_v_prope, txt_v),
                img_q_len=img_q_prope.shape[1],
                img_kv_len=img_k_prope.shape[1],
                text_mask=text_mask,
                attn_mode=eff_attn_mode,
                attn_param=attn_param,
                block_idx=block_idx,
            )
        img_attn_prope = attn_prope[:, : img_q_prope.shape[1]].contiguous()
        if p_t is not None:
            img_attn_prope = _apply_prope_mat(img_attn_prope, p_t)
        else:
            img_attn_prope = rearrange(img_attn_prope, "B L (H D) -> B H L D", H=self.heads_num)
            img_attn_prope = apply_fn_o(img_attn_prope)
            img_attn_prope = rearrange(img_attn_prope, "B H L D -> B L (H D)")
        img = fused_residual_dual_gate(
            img, self.img_attn_proj(img_attn), self.img_attn_prope_proj(img_attn_prope), img_mod1_gate
        )
        img = fused_residual_gate(
            img,
            _run_hy15_mlp(
                self.img_mlp,
                fused_layernorm_modulate(img, shift=img_mod2_shift, scale=img_mod2_scale, eps=self.img_norm2.eps),
            ),
            img_mod2_gate,
        )
        txt = fused_residual_gate(txt, self.txt_attn_proj(txt_attn), txt_mod1_gate)
        txt = fused_residual_gate(
            txt,
            _run_hy15_mlp(
                self.txt_mlp,
                fused_layernorm_modulate(txt, shift=txt_mod2_shift, scale=txt_mod2_scale, eps=self.txt_norm2.eps),
            ),
            txt_mod2_gate,
        )
        return (img, txt)

    def forward_sr(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple | None = None,
        text_mask=None,
        attn_param=None,
        is_flash=False,
        block_idx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (img_q, img_k, img_v, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.modulate_img(
            vec, img
        )
        (txt_q, txt_k, txt_v, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.modulate_txt(
            vec_txt, txt
        )
        if freqs_cis is not None:
            (img_qq, img_kk) = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            (img_q, img_k) = (img_qq, img_kk)
        attn_mode = "flash" if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )
        (img_attn, txt_attn) = (attn[:, : img_q.shape[1]].contiguous(), attn[:, img_q.shape[1] :].contiguous())
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate
        )
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), gate=txt_mod2_gate
        )
        return (img, txt)

    def forward(self, bi_inference=True, ar_txt_inference=False, ar_vision_inference=False, **kwargs):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            return self.forward_sr(**kwargs)


class ARHunyuanVideo_1_5_DiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone.

    Args:
        patch_size (list): The size of the patch.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        hidden_size (int): The hidden size of the transformer backbone.
        heads_num (int): The number of attention heads.
        mlp_width_ratio (float): Width ratio for the transformer MLPs.
        mlp_act_type (str): Activation type for the transformer MLPs.
        mm_double_blocks_depth (int): Number of double-stream transformer blocks.
        mm_single_blocks_depth (int): Number of single-stream transformer blocks.
        rope_dim_list (list): Rotary embedding dim for t, h, w.
        qkv_bias (bool): Use bias in qkv projection.
        qk_norm (bool): Whether to use qk norm.
        qk_norm_type (str): Type of qk norm.
        guidance_embed (bool): Use guidance embedding for distillation.
        text_projection (str): Text input projection. Default is "single_refiner".
        use_attention_mask (bool): If to use attention mask.
        text_states_dim (int): Text encoder output dim.
        text_states_dim_2 (int): Secondary text encoder output dim.
        text_pool_type (str): Type for text pooling.
        rope_theta (int): Rotary embedding theta parameter.
        attn_mode (str): Attention mode identifier.
        attn_param (dict): Attention parameter dictionary.
        glyph_byT5_v2 (bool): Use ByT5 glyph module.
        vision_projection (str): Vision condition embedding mode.
        vision_states_dim (int): Vision encoder states input dim.
        is_reshape_temporal_channels (bool): For video VAE adaptation.
        use_cond_type_embedding (bool): Use condition type embedding.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],  # noqa: B006 - Keeps the upstream default object and signature.
        in_channels: int = 4,
        concat_condition: bool = True,
        out_channels: int | None = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: list = [16, 56, 56],  # noqa: B006 - Keeps the upstream default object and signature.
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        text_pool_type: str | None = None,
        rope_theta: int = 256,
        attn_mode: str = "flash",
        attn_param: dict | None = None,
        glyph_byT5_v2: bool = False,
        vision_projection: str = "none",
        vision_states_dim: int = 1280,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = False,
        ideal_resolution: str | None = None,
        ideal_task: str | None = None,
    ):
        super().__init__()
        factory_kwargs = {}
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=hidden_size, use_residual=False
            )
        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )
        if vision_projection == "linear":
            self.vision_in = VisionProjection(input_dim=self.vision_states_dim, output_dim=self.hidden_size)
        else:
            self.vision_in = None
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim, self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        self.vector_in = (
            MLPEmbedder(self.config.text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = None
        self.time_r_in = None
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )
        assert mm_single_blocks_depth == 0, "No single block in HunyuanVideo 1.5 architecture"
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels, get_activation_layer("silu"), **factory_kwargs
        )
        if attn_param is None:
            self.attn_param = {
                "win_size": [[3, 3, 3]],
                "win_type": "fixed",
                "win_ratio": 10,
                "tile_size": [6, 8, 8],
                "ssta_topk": 64,
                "ssta_threshold": 0.0,
                "ssta_lambda": 0.7,
                "ssta_sampling_type": "importance",
                "ssta_adaptive_pool": None,
                "attn_sparse_type": "ssta",
                "attn_pad_type": "zero",
                "attn_use_text_mask": 1,
                "attn_mask_share_within_head": 0,
            }
        else:
            self.attn_param = attn_param
        if attn_mode == "flex-block-attn":
            self.register_to_config(attn_param=self.attn_param)
        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
        else:
            self.cond_type_embedding = None
        self.attention_backend: str | None = normalize_attention_backend(attn_mode) if attn_mode else "torch"
        self._rope_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._static_kv_arena: dict[tuple[Any, ...], list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._static_fp8_kv_arena: dict[tuple[Any, ...], dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]]] = {}
        self._last_arena_source_key: tuple[Any, ...] | None = None
        self._last_arena_src_refs: tuple[Any, ...] | None = None
        self._last_arena_src_vers: tuple[Any, ...] | None = None
        self._last_arena_entry: tuple[list[tuple[torch.Tensor, torch.Tensor]], int, tuple[Any, ...]] | None = None
        self._last_fp8_arena_entry: (
            tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]], int, tuple[Any, ...]] | None
        ) = None
        self._vision_graph_enabled: bool = False
        self._vision_graph_warmup: int = 1
        self._vision_graph_max_graphs: int = 8
        self._vision_graph_runners: dict[tuple[Any, ...], Any] = {}

    def fuse_qkv(self) -> int:
        """Fuse separate Q/K/V projections across all 54 double-stream blocks."""
        total = 0
        for block in self.double_blocks:
            total += block.fuse_qkv()
        self.invalidate_cuda_graphs()
        return total

    def enable_fp8(
        self,
        *,
        exclude_edge_blocks: bool = True,
        scaling: str = "rowwise",
        use_fast_accum: bool = True,
        min_features: int = 1024,
        keep_dense_fallback: bool = False,
    ) -> int:
        """Apply Hopper/Blackwell FP8 W8A8 quantization to eligible transformer linears."""
        from worldfoundry.core.acceleration.quantization.linear import replace_linear_with_float8

        excludes = [
            "img_in",
            "txt_in",
            "time_in",
            "action_in",
            "vector_in",
            "byt5_in",
            "vision_in",
            "final_layer",
            "img_mod",
            "txt_mod",
        ]
        if exclude_edge_blocks and len(self.double_blocks) >= 2:
            excludes.extend(["double_blocks.0.", f"double_blocks.{len(self.double_blocks) - 1}."])
        replaced = replace_linear_with_float8(
            self,
            exclude=tuple(excludes),
            min_features=min_features,
            scaling=scaling,
            use_fast_accum=use_fast_accum,
            keep_dense_fallback=keep_dense_fallback,
        )
        self.invalidate_cuda_graphs()
        return replaced

    def enable_cuda_graph(self, *, warmup: int = 1, max_graphs: int = 8) -> None:
        """Enable persistent-arena chunk-step CUDA Graph replay for ``forward_vision``."""
        self._vision_graph_enabled = True
        self._vision_graph_warmup = int(warmup)
        self._vision_graph_max_graphs = int(max_graphs)

    def disable_cuda_graph(self) -> None:
        """Disable ``forward_vision`` CUDA Graph replay and release captured graphs."""
        self._vision_graph_enabled = False
        self.invalidate_cuda_graphs()

    def invalidate_cuda_graphs(self) -> None:
        """Drop captured vision graphs and static KV arena bindings."""
        for runner in self._vision_graph_runners.values():
            runner.invalidate()
        self._vision_graph_runners.clear()
        self._static_kv_arena.clear()
        self._static_fp8_kv_arena.clear()
        self._last_arena_source_key = None
        self._last_arena_src_refs = None
        self._last_arena_src_vers = None
        self._last_arena_entry = None
        self._last_fp8_arena_entry = None

    def graph_report(self) -> dict[str, Any] | None:
        """Return aggregate capture/replay telemetry for ``forward_vision`` CUDA Graphs."""
        if not self._vision_graph_enabled and not self._vision_graph_runners:
            return None
        captures = replays = eager = failures = 0
        for runner in self._vision_graph_runners.values():
            rep = runner.report()
            captures += int(rep.get("capture", 0))
            replays += int(rep.get("replay", 0))
            eager += int(rep.get("eager", 0))
            failures += int(rep.get("capture_failed", 0))
        return {
            "enabled": self._vision_graph_enabled,
            "runner_count": len(self._vision_graph_runners),
            "capture": captures,
            "replay": replays,
            "eager": eager,
            "capture_failed": failures,
        }

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind the request-scoped attention backend on all double-stream blocks."""
        resolved = "torch" if backend is None else normalize_attention_backend(str(backend))
        changed = resolved != self.attention_backend
        self.attention_backend = resolved
        self.attn_mode = resolved
        for block in self.double_blocks:
            block.set_attention_backend(resolved)
        if changed:
            self.invalidate_cuda_graphs()

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        (freqs_cos, freqs_sin) = get_nd_rotary_pos_embed(
            rope_dim_list, rope_sizes, theta=self.rope_theta, use_real=True, theta_rescale_factor=1
        )
        return (freqs_cos, freqs_sin)

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()
                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )
                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)
            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)
        return (reorder_txt, reorder_mask)

    def add_action_parameters(self):
        pass

    def add_prope_parameters(self):
        for block in self.double_blocks:
            if hasattr(block, "img_attn_prope_proj"):
                print("Trying to add prope parameters, but img_attn_prope_proj already exists. Skipping...")
                continue
            ref_param = block.img_attn_proj.weight
            block.img_attn_prope_proj = nn.Linear(
                block.hidden_size,
                block.hidden_size,
                bias=block.qkv_bias,
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
            nn.init.zeros_(block.img_attn_prope_proj.weight)
            if block.img_attn_prope_proj.bias is not None:
                nn.init.zeros_(block.img_attn_prope_proj.bias)

    def add_discrete_action_parameters(self):
        if hasattr(self, "action_in"):
            print("Trying to add action_in param that exists. Skipping...")
            return
        self.action_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"))
        nn.init.zeros_(self.action_in.mlp[2].weight)
        if self.action_in.mlp[2].bias is not None:
            nn.init.zeros_(self.action_in.mlp[2].bias)

    def get_text_and_mask(
        self, encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
    ):
        text_mask = encoder_attention_mask
        txt = text_states
        bs = txt.shape[0]
        vec_txt = self.time_in(timestep_txt)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, timestep_txt, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb
        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            (txt, text_mask) = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True)
        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones(
                    (bs, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2
                    * torch.ones_like(
                        extra_encoder_hidden_states[:, :, 0],
                        dtype=torch.long,
                        device=extra_encoder_hidden_states.device,
                    )
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
            (txt, text_mask) = self.reorder_txt_token(extra_encoder_hidden_states, txt, extra_attention_mask, text_mask)
        return (txt, text_mask, vec_txt)

    def forward_txt(
        self,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        vision_states: torch.Tensor = None,
        mask_type="t2v",
        extra_kwargs=None,
        kv_cache: dict | None = None,
        cache_txt: bool | None = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if cache_txt:
            _kv_cache_new = []
            transformer_num_layers = len(self.double_blocks)
            for _ in range(transformer_num_layers):
                _kv_cache_new.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
        (txt, text_mask, vec_txt) = self.get_text_and_mask(
            encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
        )
        txt = txt[text_mask.bool().to(txt.device)].unsqueeze(0)
        for index, block in enumerate(self.double_blocks):
            (txt, t_kv) = block(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                txt=txt,
                vec_txt=vec_txt,
                text_mask=None,
                attn_param=None,
                is_flash=False,
                block_idx=index,
                kv_cache=kv_cache,
                cache_txt=cache_txt,
            )
            if cache_txt:
                _kv_cache_new[index]["k_txt"] = t_kv["k_txt"]
                _kv_cache_new[index]["v_txt"] = t_kv["v_txt"]
        if cache_txt:
            return _kv_cache_new

    def _get_cached_rope(
        self,
        rope_temporal_size: int,
        th: int,
        tw: int,
        start_rope_start_idx: int,
        tt: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (rope_temporal_size, th, tw, start_rope_start_idx, tt, device)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        base_key = (rope_temporal_size, th, tw, "full", device)
        full = self._rope_cache.get(base_key)
        if full is None:
            f_cos, f_sin = self.get_rotary_pos_embed((rope_temporal_size, th, tw))
            full = (f_cos.to(device=device), f_sin.to(device=device))
            if len(self._rope_cache) >= 32:
                self._rope_cache.clear()
            self._rope_cache[base_key] = full
        per_latent_size = th * tw
        start_index = start_rope_start_idx * per_latent_size
        end_index = (start_rope_start_idx + tt) * per_latent_size
        sliced = (
            full[0][start_index:end_index, ...].contiguous(),
            full[1][start_index:end_index, ...].contiguous(),
        )
        self._rope_cache[key] = sliced
        return sliced

    def _run_vision_core(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        viewmats: torch.Tensor | None,
        Ks: torch.Tensor | None,
        freqs_cos: torch.Tensor | None,
        freqs_sin: torch.Tensor | None,
        *,
        kv_cache: list[dict[str, Any]] | None,
        static_buffers: list[tuple[torch.Tensor, torch.Tensor]] | None,
        static_bshd_buffers: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        static_prefix_len: int | None,
        static_fp8_buffers: dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None,
        cache_vision: bool,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
        img = hidden_states
        t = timestep
        (bs, _, ot, oh, ow) = img.shape
        (tt, th, tw) = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        img = self.img_in(img)
        t = t.reshape(-1)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        prope_mats = None
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            if freqs_cos is not None and freqs_sin is not None:
                freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
                freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
            if tt % sp_size == 0:
                vec = torch.chunk(vec.reshape(bs, tt, -1), sp_size, dim=1)[sp_rank].reshape(-1, vec.shape[-1])
                if viewmats is not None:
                    vm_local = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                    ks_local = None if Ks is None else torch.chunk(Ks, sp_size, dim=1)[sp_rank]
                    prope_mats = _precompute_prope_mats(vm_local, ks_local)
            else:
                vec = repeat(vec, "(B T) C->B (T H W) C", B=bs, H=th, W=tw)
                vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
                vec = rearrange(vec, "B S C->(B S) C")
                if viewmats is not None:
                    vm_rep = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
                    ks_rep = None if Ks is None else repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)
                    vm_local = torch.chunk(vm_rep, sp_size, dim=1)[sp_rank]
                    ks_local = None if ks_rep is None else torch.chunk(ks_rep, sp_size, dim=1)[sp_rank]
                    prope_mats = _precompute_prope_mats(vm_local, ks_local)
        else:
            if viewmats is not None:
                prope_mats = _precompute_prope_mats(viewmats, Ks)

        kv_cache_new = None
        if cache_vision:
            assert kv_cache is not None
            kv_cache_new = [
                {"k_vision": None, "v_vision": None, "k_txt": kv_cache[i]["k_txt"], "v_txt": kv_cache[i]["v_txt"]}
                for i in range(len(self.double_blocks))
            ]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        num_blocks = len(self.double_blocks)
        fp8_rope_list = None if static_fp8_buffers is None else static_fp8_buffers.get("rope")
        fp8_prope_list = None if static_fp8_buffers is None else static_fp8_buffers.get("prope")
        for index, block in enumerate(self.double_blocks):
            static_pair = None if static_buffers is None else static_buffers[index]
            static_bshd_pair = None if static_bshd_buffers is None else static_bshd_buffers[index]
            fp8_rope_pair = None if fp8_rope_list is None else fp8_rope_list[index]
            fp8_prope_pair = None if fp8_prope_list is None else fp8_prope_list[index]
            (img, vision_kv) = block.forward_vision(
                img=img,
                vec=vec,
                freqs_cis=freqs_cis,
                attn_param=self.attn_param,
                block_idx=index,
                viewmats=None if prope_mats is not None else viewmats,
                Ks=None if prope_mats is not None else Ks,
                kv_cache=kv_cache,
                cache_vision=cache_vision,
                prope_mats=prope_mats,
                static_kv_buffer=static_pair,
                static_bshd_kv_buffer=static_bshd_pair,
                static_prefix_len=static_prefix_len,
                static_fp8_rope_buffer=fp8_rope_pair,
                static_fp8_prope_buffer=fp8_prope_pair,
                num_blocks=num_blocks,
            )
            if kv_cache_new is not None:
                kv_cache_new[index]["k_vision"] = vision_kv["k_vision"]
                kv_cache_new[index]["v_vision"] = vision_kv["v_vision"]

        if kv_cache_new is not None:
            return img, kv_cache_new
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        return img, None

    @staticmethod
    def _kv_cache_identity(kv_cache: list[dict[str, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Capture source tensor object references and mutation versions for arena freshness."""
        first = kv_cache[0]
        last = kv_cache[-1]
        refs = (
            kv_cache,
            first["k_txt"],
            first["v_txt"],
            first.get("k_vision"),
            first.get("v_vision"),
            last["k_txt"],
            last["v_txt"],
            last.get("k_vision"),
            last.get("v_vision"),
        )
        vers = tuple(None if (t is None or t.is_inference()) else t._version for t in refs[1:])
        return refs, vers

    def forward_vision(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_r=None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        mask_type="t2v",
        extra_kwargs=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        cache_vision: bool = False,
        rope_temporal_size=4,
        start_rope_start_idx=0,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del timestep_r, mask_type, extra_kwargs
        (_, _, ot, oh, ow) = hidden_states.shape
        (tt, th, tw) = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        self.attn_param["thw"] = [tt, th, tw]
        rope_temporal_size = rope_temporal_size // self.patch_size[0]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self._get_cached_rope(
                rope_temporal_size, th, tw, start_rope_start_idx, tt, hidden_states.device
            )

        sp_enabled = get_parallel_state().sp_enabled
        if (
            self._vision_graph_enabled
            and not cache_vision
            and not sp_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and kv_cache is not None
        ):
            chunk_tokens = tt * th * tw
            head_dim = self.hidden_size // self.heads_num
            want_fp8_arena = (
                self.attention_backend == "cudnn_fp8" and len(self.double_blocks) > 2 and head_dim in {64, 128}
            )
            want_bshd_arena = (
                self.attention_backend == "flash_attention_3"
                and hidden_states.dtype in {torch.bfloat16, torch.float16}
                and hopper_fa3_usable(hidden_states.device)
            )
            k_vis0 = kv_cache[0].get("k_vision")
            src_refs, src_vers = self._kv_cache_identity(kv_cache)
            source_key = (
                tuple(kv_cache[0]["k_txt"].shape),
                None if k_vis0 is None else tuple(k_vis0.shape),
                chunk_tokens,
                hidden_states.dtype,
                self.attention_backend,
                want_bshd_arena,
            )
            same_sources = (
                self._last_arena_source_key == source_key
                and self._last_arena_src_refs is not None
                and self._last_arena_src_vers == src_vers
                and all(a is b for a, b in zip(self._last_arena_src_refs, src_refs, strict=True))
            )

            fp8_buffers: dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None
            fp8_key: tuple[Any, ...] | None = None
            if same_sources and self._last_arena_entry is not None:
                buffers, prefix_len, arena_key = self._last_arena_entry
                if want_fp8_arena:
                    if self._last_fp8_arena_entry is not None:
                        fp8_buffers, _, fp8_key = self._last_fp8_arena_entry
                    else:
                        fp8_buffers, _, fp8_key = prepare_static_fp8_kv_arena(
                            kv_cache, chunk_tokens, arena=self._static_fp8_kv_arena
                        )
                        if fp8_buffers is not None and fp8_key is not None:
                            self._last_fp8_arena_entry = (fp8_buffers, prefix_len, fp8_key)
            else:
                if want_bshd_arena:
                    buffers, prefix_len, arena_key = prepare_static_kv_arena_bshd(
                        kv_cache,
                        chunk_tokens,
                        arena=self._static_kv_arena,
                        dtype=hidden_states.dtype,
                    )
                else:
                    buffers, prefix_len, arena_key = prepare_static_kv_arena(
                        kv_cache,
                        chunk_tokens,
                        arena=self._static_kv_arena,
                        dtype=hidden_states.dtype,
                    )
                self._last_arena_source_key = source_key
                self._last_arena_src_refs = src_refs
                self._last_arena_src_vers = src_vers
                self._last_arena_entry = (buffers, prefix_len, arena_key)
                self._last_fp8_arena_entry = None
                if want_fp8_arena:
                    fp8_buffers, _, fp8_key = prepare_static_fp8_kv_arena(
                        kv_cache, chunk_tokens, arena=self._static_fp8_kv_arena
                    )
                    if fp8_buffers is not None and fp8_key is not None:
                        self._last_fp8_arena_entry = (fp8_buffers, prefix_len, fp8_key)

            # On the very first cudnn_fp8 call before scales are calibrated, run one eager
            # calibration pass so CUDA Graph captures with the pre-quantized FP8 KV arena.
            if want_fp8_arena and fp8_buffers is None:
                img, _ = self._run_vision_core(
                    hidden_states,
                    timestep,
                    action,
                    viewmats,
                    Ks,
                    freqs_cos,
                    freqs_sin,
                    kv_cache=None,
                    static_buffers=buffers,
                    static_bshd_buffers=None,
                    static_prefix_len=prefix_len,
                    static_fp8_buffers=None,
                    cache_vision=False,
                )
                assert return_dict is False, "return_dict is not supported."
                return (img, None)

            runner_key = (
                arena_key,
                fp8_key,
                self.attention_backend,
                action is not None,
                viewmats is not None,
                Ks is not None,
            )
            runner = self._vision_graph_runners.get(runner_key)
            if runner is None:
                from worldfoundry.core.execution.graphs.inference_graph import InferenceCUDAGraphRunner

                has_action = action is not None
                has_vm = viewmats is not None
                has_ks = Ks is not None
                bound_bhsd_buffers = None if want_bshd_arena else buffers
                bound_bshd_buffers = buffers if want_bshd_arena else None
                bound_prefix_len = prefix_len
                bound_fp8_buffers = fp8_buffers

                def _graphed_step(h_in, t_in, act_in, vm_in, ks_in, cos_in, sin_in):
                    out_img, _ = self._run_vision_core(
                        h_in,
                        t_in,
                        act_in if has_action else None,
                        vm_in if has_vm else None,
                        ks_in if has_ks else None,
                        cos_in,
                        sin_in,
                        kv_cache=None,
                        static_buffers=bound_bhsd_buffers,
                        static_bshd_buffers=bound_bshd_buffers,
                        static_prefix_len=bound_prefix_len,
                        static_fp8_buffers=bound_fp8_buffers,
                        cache_vision=False,
                    )
                    return out_img

                runner = InferenceCUDAGraphRunner(
                    _graphed_step,
                    warmup=self._vision_graph_warmup,
                    max_graphs=self._vision_graph_max_graphs,
                    clone_outputs=True,
                    extra_key=f"hy_worldplay_vision:{runner_key}",
                )
                self._vision_graph_runners[runner_key] = runner

            dummy = hidden_states.new_empty(0)
            img = runner(
                hidden_states,
                timestep,
                action if action is not None else dummy,
                viewmats if viewmats is not None else dummy,
                Ks if Ks is not None else dummy,
                freqs_cos,
                freqs_sin,
            )
            assert return_dict is False, "return_dict is not supported."
            return (img, None)

        img, kv_cache_new = self._run_vision_core(
            hidden_states,
            timestep,
            action,
            viewmats,
            Ks,
            freqs_cos,
            freqs_sin,
            kv_cache=kv_cache,
            static_buffers=None,
            static_prefix_len=None,
            static_fp8_buffers=None,
            cache_vision=cache_vision,
        )
        if cache_vision:
            return kv_cache_new
        assert return_dict is False, "return_dict is not supported."
        return (img, None)

    def forward_bi(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_txt: torch.Tensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        output_features=False,
        output_features_stride=8,
        attention_kwargs: dict[str, Any] | None = None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        extra_kwargs=None,
        viewmats: torch.Tensor | None = None,
        Ks: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        t = timestep
        (bs, _, ot, oh, ow) = x.shape
        (tt, th, tw) = (ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2])
        self.attn_param["thw"] = [tt, th, tw]
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self._get_cached_rope(tt, th, tw, 0, tt, hidden_states.device)
        img = self.img_in(img)
        vec = self.time_in(t)
        if action is not None and hasattr(self, "action_in"):
            vec = vec + self.action_in(action.reshape(-1))
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled
        prope_mats = None
        if sp_enabled:
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f"Too short context length for SP {sp_size}"
            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]
            if tt % sp_size == 0:
                vec = torch.chunk(vec.reshape(bs, tt, -1), sp_size, dim=1)[sp_rank].reshape(-1, vec.shape[-1])
                if viewmats is not None:
                    vm_local = torch.chunk(viewmats, sp_size, dim=1)[sp_rank]
                    ks_local = None if Ks is None else torch.chunk(Ks, sp_size, dim=1)[sp_rank]
                    prope_mats = _precompute_prope_mats(vm_local, ks_local)
            else:
                vec = repeat(vec, "(B T) C->B (T H W) C", B=bs, H=th, W=tw)
                vec = torch.chunk(vec, sp_size, dim=1)[sp_rank]
                vec = rearrange(vec, "B S C->(B S) C")
                if viewmats is not None:
                    vm_rep = repeat(viewmats, "B T M N->B (T H W) M N", H=th, W=tw)
                    ks_rep = None if Ks is None else repeat(Ks, "B T M N->B (T H W) M N", H=th, W=tw)
                    vm_local = torch.chunk(vm_rep, sp_size, dim=1)[sp_rank]
                    ks_local = None if ks_rep is None else torch.chunk(ks_rep, sp_size, dim=1)[sp_rank]
                    prope_mats = _precompute_prope_mats(vm_local, ks_local)
        else:
            if viewmats is not None:
                prope_mats = _precompute_prope_mats(viewmats, Ks)
        assert text_states_2 is None, (
            "text_states_2 is not None, but handling of text_states_2 is not implemented in forward_bi yet."
        )
        assert self.guidance_embed is False, (
            "guidance_embed is True, but handling of guidance is not implemented in forward_bi yet."
        )
        assert timestep_r is None, (
            "timestep_r is not None, but handling of timestep_r is not implemented in forward_bi yet."
        )
        (txt, text_mask, vec_txt) = self.get_text_and_mask(
            encoder_attention_mask, text_states, timestep_txt, extra_kwargs, vision_states, mask_type
        )
        block_text_mask = text_mask
        resolved_bi_mode = self.attention_backend or self.attn_mode
        if (
            block_text_mask is not None
            and resolved_bi_mode
            in {None, "torch", "flash", "sdpa", "cudnn", "flash_attention_3", "sol_attn", "cudnn_fp8"}
            and not torch.cuda.is_current_stream_capturing()
        ):
            if block_text_mask.dtype in {torch.int64, torch.int32}:
                assert block_text_mask.max() <= 1 and block_text_mask.min() >= 0, (
                    "Integer attention mask must be between 0 and 1 for torch attention."
                )
                block_text_mask = block_text_mask.to(torch.bool)
            if block_text_mask.dtype == torch.bool:
                if bool(block_text_mask.all()):
                    block_text_mask = None
                elif (
                    block_text_mask.ndim == 2
                    and block_text_mask.shape[1] == txt.shape[1]
                    and (block_text_mask.shape[0] == 1 or bool((block_text_mask == block_text_mask[:1]).all()))
                ):
                    # Compact uniform padded text tokens once before the block stack so SDPA
                    # takes the maskless Flash/cuDNN kernel instead of materializing 108 dense
                    # (B, 1, L_img+L_txt, L_img+L_txt) pair_mask tensors. Masked txt positions
                    # never affect img or valid txt tokens and txt is not returned.
                    valid_idx = block_text_mask[0].nonzero(as_tuple=False).squeeze(-1)
                    if valid_idx.numel() > 0:
                        txt = txt.index_select(1, valid_idx).contiguous()
                        block_text_mask = None
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        num_bi_blocks = len(self.double_blocks)
        for index, block in enumerate(self.double_blocks):
            force_full_attn = (
                self.attn_mode in ["flex-block-attn"]
                and self.attn_param["win_type"] == "hybrid"
                and (self.attn_param["win_ratio"] > 0)
                and ((index + 1) % self.attn_param["win_ratio"] == 0 or index + 1 == num_bi_blocks)
            )
            self.attn_param["layer-name"] = f"double_block_{index + 1}"
            (img, txt) = block.forward_bi(
                img=img,
                txt=txt,
                vec_txt=vec_txt,
                vec=vec,
                freqs_cis=freqs_cis,
                text_mask=block_text_mask,
                attn_param=self.attn_param,
                is_flash=force_full_attn,
                block_idx=index,
                viewmats=None if prope_mats is not None else viewmats,
                Ks=None if prope_mats is not None else Ks,
                prope_mats=prope_mats,
                num_blocks=num_bi_blocks,
            )
        img = self.final_layer(img, vec)
        if sp_enabled:
            img = sequence_model_parallel_all_gather(img, dim=1)
        img = self.unpatchify(img, tt, th, tw)
        assert return_dict is False, "return_dict is not supported."
        assert output_features is False, "output_features is not supported in bi-inference mode currently."
        features_list = None
        return (img, features_list)

    def forward(self, bi_inference=True, ar_txt_inference=False, ar_vision_inference=False, **kwargs):
        if bi_inference:
            return self.forward_bi(**kwargs)
        elif ar_txt_inference:
            return self.forward_txt(**kwargs)
        elif ar_vision_inference:
            return self.forward_vision(**kwargs)
        else:
            raise NotImplementedError

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        (pt, ph, pw) = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        self.set_attention_backend(attn_mode)
