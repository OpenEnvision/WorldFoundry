# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for LingBot-Video Dense & MoE transformers.

Composes four orthogonal inference acceleration pillars:

1. **Kernel Fusion & Broadcast AdaLN**:
   - Lossless projection fusion: collapses ``to_q/to_k/to_v`` into one
     contiguous ``qkv_proj`` (eliminating per-forward weight concatenation) and
     collapses ``gate_proj/up_proj`` into ``gate_up_proj`` + Triton ``silu_and_mul``.
   - Compact broadcast AdaLN modulation: runs ``time_modulation`` on ``(B, D)``
     instead of expanding ``(B, S, 6D)`` across 30k+ tokens (eliminating tens
     of gigabytes of HBM traffic per step).
   - Triton fused per-head QK-RMSNorm + interleaved 3D RoPE
     (``worldfoundry::lingbot_qk_rmsnorm_rope``), Triton ``residual_gate_add``,
     and Triton ``layer_norm_scale_shift`` for the output norm.
   - Zero-sync cached 3D RoPE tables ``(cos, sin)`` on unpadded ``B=1`` paths.
2. **Hopper FlashAttention-3 + In-Tree PISA Sparse Attention**:
   - Native Hopper FA3 dispatch (``worldfoundry::lingbot_fa3_attn``) for dense
     self-attention blocks and dense warmup/boundary steps.
   - In-tree PISA (``worldfoundry.core.attention.sparse.piecewise.piecewise_attention``)
     for refiner / long-sequence interior blocks with configurable density,
     block size, dense edge-block guards (e.g. ``(0, 1, -1)``), and dense
     warmup/tail step guards.
3. **Calibrated EasyCache Cross-Step Residual Reuse**:
   - Request/branch-isolated online strided EasyCache over the transformer
     block stack with warmup, dense-last, and max-skip boundary protection.
4. **Piecewise Block-Stack CUDA Graph**:
   - Captures pure-tensor dense and sparse block-stack segments separately in a
     :class:`PiecewiseBlockGraphRunner` outside Python EasyCache gating.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import torch
from torch import nn

from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.attention.sparse.piecewise import (
    piecewise_attention,
    piecewise_attention_available,
)
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _lingbot_qk_rmsnorm_rope_kernel(
        Q_ptr,
        K_ptr,
        QW_ptr,
        KW_ptr,
        COS_ptr,
        SIN_ptr,
        Q_out_ptr,
        K_out_ptr,
        stride_q_b,
        stride_q_s,
        stride_q_h,
        stride_k_b,
        stride_k_s,
        stride_k_h,
        stride_qo_b,
        stride_qo_s,
        stride_qo_h,
        stride_ko_b,
        stride_ko_s,
        stride_ko_h,
        stride_cos_b,
        stride_cos_s,
        stride_sin_b,
        stride_sin_s,
        S,
        H: tl.constexpr,
        PAIRS: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        eps,
    ):
        """Fuse per-head FP32 RMSNorm + interleaved complex RoPE for [B, S, H, D]."""
        bs_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        b_idx = bs_idx // S
        s_idx = bs_idx % S

        pair_offs = tl.arange(0, BLOCK_PAIRS)
        mask = pair_offs < PAIRS
        even_offs = pair_offs * 2
        odd_offs = even_offs + 1

        cos = tl.load(COS_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s + pair_offs, mask=mask, other=1.0).to(
            tl.float32
        )
        sin = tl.load(SIN_ptr + b_idx * stride_sin_b + s_idx * stride_sin_s + pair_offs, mask=mask, other=0.0).to(
            tl.float32
        )

        is_q = head_idx < H
        if is_q:
            h = head_idx
            base_in = Q_ptr + b_idx * stride_q_b + s_idx * stride_q_s + h * stride_q_h
            base_out = Q_out_ptr + b_idx * stride_qo_b + s_idx * stride_qo_s + h * stride_qo_h
            w_even = tl.load(QW_ptr + even_offs, mask=mask, other=0.0).to(tl.float32)
            w_odd = tl.load(QW_ptr + odd_offs, mask=mask, other=0.0).to(tl.float32)
        else:
            h = head_idx - H
            base_in = K_ptr + b_idx * stride_k_b + s_idx * stride_k_s + h * stride_k_h
            base_out = K_out_ptr + b_idx * stride_ko_b + s_idx * stride_ko_s + h * stride_ko_h
            w_even = tl.load(KW_ptr + even_offs, mask=mask, other=0.0).to(tl.float32)
            w_odd = tl.load(KW_ptr + odd_offs, mask=mask, other=0.0).to(tl.float32)

        x_even = tl.load(base_in + even_offs, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(base_in + odd_offs, mask=mask, other=0.0).to(tl.float32)

        # Match LingBotVideoRMSNorm: fp32 variance & weight multiply, then interleaved RoPE.
        sq_sum = tl.sum(x_even * x_even + x_odd * x_odd, axis=0)
        inv_rms = tl.rsqrt(sq_sum / (2.0 * PAIRS) + eps)
        n_even = (x_even * inv_rms) * w_even
        n_odd = (x_odd * inv_rms) * w_odd

        r_even = n_even * cos - n_odd * sin
        r_odd = n_even * sin + n_odd * cos

        tl.store(base_out + even_offs, r_even, mask=mask)
        tl.store(base_out + odd_offs, r_odd, mask=mask)


@torch.library.custom_op("worldfoundry::lingbot_qk_rmsnorm_rope", mutates_args=(), device_types="cuda")
def _lingbot_qk_rmsnorm_rope_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    output_bhsd: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    pairs = d // 2
    block_pairs = triton.next_power_of_2(pairs)
    if output_bhsd:
        q_out = torch.empty((b, h, s, d), device=q.device, dtype=q.dtype)
        k_out = torch.empty((b, h, s, d), device=k.device, dtype=k.dtype)
        qo_sb, qo_ss, qo_sh = q_out.stride(0), q_out.stride(2), q_out.stride(1)
        ko_sb, ko_ss, ko_sh = k_out.stride(0), k_out.stride(2), k_out.stride(1)
    else:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        qo_sb, qo_ss, qo_sh = q_out.stride(0), q_out.stride(1), q_out.stride(2)
        ko_sb, ko_ss, ko_sh = k_out.stride(0), k_out.stride(1), k_out.stride(2)
    grid = (b * s, 2 * h)
    _lingbot_qk_rmsnorm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        qo_sb,
        qo_ss,
        qo_sh,
        ko_sb,
        ko_ss,
        ko_sh,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        s,
        H=h,
        PAIRS=pairs,
        BLOCK_PAIRS=block_pairs,
        eps=float(eps),
        num_warps=2 if block_pairs <= 64 else 4,
    )
    return q_out, k_out


@_lingbot_qk_rmsnorm_rope_op.register_fake
def _lingbot_qk_rmsnorm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    output_bhsd: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos, sin, eps
    if output_bhsd:
        b, s, h, d = q.shape
        return q.new_empty((b, h, s, d)), k.new_empty((b, h, s, d))
    return torch.empty_like(q), torch.empty_like(k)


def fused_lingbot_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    eps: float = 1e-6,
    *,
    output_bhsd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse per-head FP32 QK-RMSNorm + interleaved 3D RoPE for ``[B, S, H, D]``."""
    if isinstance(rotary_emb, tuple):
        cos, sin = rotary_emb
    else:
        cos = rotary_emb.real.contiguous()
        sin = rotary_emb.imag.contiguous()

    if (
        _TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and q.is_cuda
        and k.is_cuda
        and q.ndim == 4
        and k.shape == q.shape
        and q.shape[-1] % 2 == 0
        and q.numel() > 0
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and q.dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        cos_2d = cos if cos.ndim == 3 else cos.reshape(q.shape[0], q.shape[1], -1)
        sin_2d = sin if sin.ndim == 3 else sin.reshape(q.shape[0], q.shape[1], -1)
        return _lingbot_qk_rmsnorm_rope_op(
            q,
            k,
            q_weight.float().contiguous(),
            k_weight.float().contiguous(),
            cos_2d.float().contiguous(),
            sin_2d.float().contiguous(),
            float(eps),
            bool(output_bhsd),
        )

    # Exact portable fallback matching LingBotVideoRMSNorm + apply_rotary_emb.
    def _norm(val: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        vf = val.float()
        return (weight.float() * (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps))).to(val.dtype)

    qn = _norm(q, q_weight)
    kn = _norm(k, k_weight)
    cos_u = cos.float().unsqueeze(2)
    sin_u = sin.float().unsqueeze(2)

    def _rot(val: torch.Tensor) -> torch.Tensor:
        pairs = val.float().reshape(*val.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        out = torch.stack((even * cos_u - odd * sin_u, even * sin_u + odd * cos_u), dim=-1).flatten(-2)
        return out.to(val.dtype)

    q_res, k_res = _rot(qn), _rot(kn)
    if output_bhsd:
        return q_res.transpose(1, 2).contiguous(), k_res.transpose(1, 2).contiguous()
    return q_res, k_res


@torch.library.custom_op("worldfoundry::lingbot_fa3_attn", mutates_args=(), device_types="cuda")
def _lingbot_fa3_attn_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import flash_attn_interface as fa3

    out = fa3.flash_attn_func(q, k, v, causal=False)
    return out[0] if isinstance(out, tuple) else out


@_lingbot_fa3_attn_op.register_fake
def _lingbot_fa3_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    del k
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def lingbot_fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run Hopper FlashAttention-3 on ``[B, S, H, D]`` tensors."""
    return _lingbot_fa3_attn_op(q.contiguous(), k.contiguous(), v.contiguous())


def lingbot_pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    density: float = 0.25,
    block_size: int = 64,
    min_sequence_length: int = 1024,
    strict: bool = False,
    q_k_bhsd: bool = False,
) -> torch.Tensor:
    """Run in-tree PISA piecewise sparse attention returning ``[B, S, H, D]``."""
    q_in = q if q_k_bhsd else q.transpose(1, 2).contiguous()
    k_in = k if q_k_bhsd else k.transpose(1, 2).contiguous()
    v_in = v.transpose(1, 2).contiguous()
    out_bhsd = piecewise_attention(
        q_in,
        k_in,
        v_in,
        density=float(density),
        block_size=int(block_size),
        min_sequence_length=int(min_sequence_length),
        strict=bool(strict),
    )
    return out_bhsd.transpose(1, 2).contiguous()


@dataclass(frozen=True, slots=True)
class LingBotAccelerationConfig:
    """Declarative SOL-Engine acceleration recipe for LingBot-Video."""

    fuse_qkv: bool = True
    fuse_swiglu: bool = True
    compact_adaln: bool = True
    fused_kernels: bool = True
    attention_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3"
    # In-tree PISA sparse attention for refiner / long-sequence interior blocks.
    pisa_enabled: bool = False
    pisa_density: float = 0.25
    pisa_block_size: int = 64
    pisa_min_sequence_length: int = 1024
    pisa_dense_steps: int = 6
    pisa_dense_last_steps: int = 2
    pisa_dense_blocks: tuple[int, ...] = (0, 1, -1)
    pisa_strict: bool = False
    # Calibrated EasyCache cross-step residual reuse.
    easycache: bool = False
    easycache_threshold: float = 0.12
    easycache_warmup_steps: int = 4
    easycache_dense_last: int = 2
    easycache_max_skip_steps: int = 3
    easycache_subsample_stride: int = 64
    # Piecewise CUDA Graph over the block stack.
    cuda_graph: Literal["off", "piecewise"] = "off"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 8
    cuda_graph_clone_outputs: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pisa_dense_blocks"] = list(self.pisa_dense_blocks)
        payload["notes"] = list(self.notes)
        return payload


class LingBotEasyCache:
    """Branch-isolated online strided EasyCache over LingBot's transformer block stack."""

    algorithm = "easycache"

    def __init__(
        self,
        *,
        threshold: float = 0.12,
        total_steps: int = 40,
        warmup_steps: int = 4,
        dense_last: int = 2,
        max_skip_steps: int = 3,
        subsample_stride: int = 64,
        eps: float = 1e-8,
    ) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("LingBotEasyCache threshold must be finite and non-negative")
        self.threshold = float(threshold)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.max_skip_steps = int(max_skip_steps)
        self.subsample_stride = int(subsample_stride)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._previous_input: torch.Tensor | None = None
        self._dense_input: torch.Tensor | None = None
        self._dense_output: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._factor: torch.Tensor | None = None
        self._accumulator: float = 0.0
        self._skips: int = 0
        self._layout: tuple[tuple[int, ...], torch.dtype, torch.device, int] | None = None
        self.events: list[BlockCacheEvent] = []

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.detach().reshape(value.shape[0], -1)
        if self.subsample_stride > 1 and flat.shape[1] >= self.subsample_stride:
            flat = flat[:, :: self.subsample_stride]
        return flat.float().clone()

    def run_stack(
        self,
        *,
        step: int,
        total_steps: int | None,
        value: torch.Tensor,
        block_count: int,
        run_stack_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        count = self.total_steps if total_steps is None else int(total_steps)
        layout = (tuple(value.shape), value.dtype, value.device, int(block_count))
        probe = self._probe(value)
        reason: str | None = None
        if torch.is_grad_enabled():
            self._residual = None
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._residual = None
            self._factor = None
            self._dense_input = None
            self._dense_output = None
            reason = "layout-change"
        elif step < self.warmup_steps:
            reason = "warmup"
        elif step >= max(count - self.dense_last, 0):
            reason = "dense-last"
        elif (
            self._residual is None or self._factor is None or self._previous_input is None or self._dense_output is None
        ):
            reason = "seed"
        elif self._skips >= self.max_skip_steps:
            reason = "max-skip"

        if reason is None:
            delta = (probe - cast(torch.Tensor, self._previous_input)).abs().mean(dim=1)
            norm = cast(torch.Tensor, self._dense_output).abs().mean(dim=1).clamp_min(self.eps)
            estimate = float((cast(torch.Tensor, self._factor) * delta / norm).max().item())
            self._accumulator += estimate
            if not math.isfinite(self._accumulator) or self._accumulator >= self.threshold:
                reason = "threshold"

        hit = reason is None
        if hit:
            output = value + cast(torch.Tensor, self._residual)
            self._skips += 1
        else:
            output = run_stack_fn(value)
            if not torch.is_grad_enabled():
                out_probe = self._probe(output)
                if self._dense_input is not None and self._dense_output is not None:
                    in_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.eps)
                    self._factor = (out_probe - self._dense_output).abs().mean(dim=1) / in_delta
                self._dense_input = probe
                self._dense_output = out_probe
                self._residual = (output - value).detach().clone()
                self._layout = layout
            self._accumulator = 0.0
            self._skips = 0

        if not torch.is_grad_enabled():
            self._previous_input = probe
        indices = tuple(range(int(block_count)))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                int(step),
                hit,
                reason or "below-threshold",
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return output, hit

    @property
    def hits(self) -> int:
        return sum(1 for ev in self.events if ev.hit)

    @property
    def dense_block_calls(self) -> int:
        return sum(ev.dense_block_calls for ev in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(ev.skipped_block_calls for ev in self.events)

    def receipt(self) -> dict[str, Any]:
        total = len(self.events)
        hits = self.hits
        return {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "max_skip_steps": self.max_skip_steps,
            "subsample_stride": self.subsample_stride,
            "events": total,
            "hits": hits,
            "misses": total - hits,
            "hit_rate": hits / total if total else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "fallbacks": [],
        }


def _fuse_mlp_gate_up(mlp: nn.Module) -> bool:
    if (
        getattr(mlp, "gate_up_proj", None) is None
        and isinstance(getattr(mlp, "gate_proj", None), nn.Linear)
        and isinstance(getattr(mlp, "up_proj", None), nn.Linear)
    ):
        gate_proj = cast(nn.Linear, mlp.gate_proj)
        up_proj = cast(nn.Linear, mlp.up_proj)
        with torch.no_grad():
            w = torch.cat((gate_proj.weight.data, up_proj.weight.data), dim=0).contiguous()
        proj = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
        proj.weight = nn.Parameter(w, requires_grad=False)
        mlp.gate_up_proj = proj
        return True
    return False


def fuse_lingbot_projections(
    model: nn.Module,
    *,
    fuse_qkv: bool = True,
    fuse_swiglu: bool = True,
) -> dict[str, int]:
    """Losslessly pre-fuse LingBot attention QKV and dense/shared-expert MLP gate+up projections."""
    qkv_fused = 0
    swiglu_fused = 0
    for block in getattr(model, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None:
            if not fuse_qkv:
                attn.qkv_proj = None
            elif (
                getattr(attn, "qkv_proj", None) is None
                and isinstance(getattr(attn, "to_q", None), nn.Linear)
                and isinstance(getattr(attn, "to_k", None), nn.Linear)
                and isinstance(getattr(attn, "to_v", None), nn.Linear)
            ):
                with torch.no_grad():
                    w = torch.cat(
                        (attn.to_q.weight.data, attn.to_k.weight.data, attn.to_v.weight.data), dim=0
                    ).contiguous()
                    b = (
                        torch.cat((attn.to_q.bias.data, attn.to_k.bias.data, attn.to_v.bias.data), dim=0).contiguous()
                        if attn.to_q.bias is not None and attn.to_k.bias is not None and attn.to_v.bias is not None
                        else None
                    )
                proj = nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device=w.device, dtype=w.dtype)
                proj.weight = nn.Parameter(w, requires_grad=False)
                if b is not None:
                    proj.bias = nn.Parameter(b, requires_grad=False)
                attn.qkv_proj = proj
                qkv_fused += 1
            elif getattr(attn, "qkv_proj", None) is not None:
                qkv_fused += 1

        ffn = getattr(block, "ffn", None)
        if ffn is not None:
            candidates = [ffn]
            shared = getattr(ffn, "shared_experts", None)
            if isinstance(shared, nn.Module):
                candidates.append(shared)
            for mlp in candidates:
                if not fuse_swiglu:
                    if hasattr(mlp, "gate_up_proj"):
                        mlp.gate_up_proj = None
                elif _fuse_mlp_gate_up(mlp) or getattr(mlp, "gate_up_proj", None) is not None:
                    swiglu_fused += 1

    return {"qkv_fused": qkv_fused, "swiglu_fused": swiglu_fused}


def _resolve_fa3_usable(sm_major: int) -> bool:
    if int(sm_major) < 9:
        return False
    from worldfoundry.core.attention.backends.probe import attention_backend_capability

    return bool(attention_backend_capability("flash_attention_3").usable)


def lingbot_sol_profile(
    tier: Literal["exact", "pisa", "sol_high_fidelity", "sol_balanced", "refiner_sol"] | str,
    *,
    sm_major: int = 9,
    cuda_graph: Literal["off", "piecewise"] | None = None,
) -> LingBotAccelerationConfig:
    """Return a validated SOL-Engine preset for LingBot-Video."""
    normalized = str(tier).strip().lower()
    fa3_ok = _resolve_fa3_usable(sm_major)
    attn_backend: Literal["torch", "flash_attention_3"] = "flash_attention_3" if fa3_ok else "torch"

    if normalized == "exact":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=False,
            easycache=False,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"lossless LingBot QKV/SwiGLU/AdaLN/QK-RoPE fusion + {attn_backend} + piecewise graph",),
        )
    if normalized == "pisa":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.25,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=6,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=False,
            cuda_graph=cuda_graph or "piecewise",
            notes=(
                f"LingBot kernel fusion + {attn_backend} + PISA(density=0.25, edge 0/1/-1 dense) + piecewise graph",
            ),
        )
    if normalized == "sol_high_fidelity":
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.35,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=6,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=True,
            easycache_threshold=0.06,
            easycache_warmup_steps=5,
            easycache_dense_last=3,
            easycache_max_skip_steps=2,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"LingBot kernel fusion + PISA(0.35) + EasyCache(0.06) + {attn_backend} + piecewise graph",),
        )
    if normalized in {"sol_balanced", "refiner_sol"}:
        return LingBotAccelerationConfig(
            attention_backend=attn_backend,
            pisa_enabled=True,
            pisa_density=0.25,
            pisa_block_size=64,
            pisa_min_sequence_length=1024,
            pisa_dense_steps=4 if normalized == "refiner_sol" else 5,
            pisa_dense_last_steps=2,
            pisa_dense_blocks=(0, 1, -1),
            easycache=True,
            easycache_threshold=0.18,
            easycache_warmup_steps=4,
            easycache_dense_last=2,
            easycache_max_skip_steps=3,
            cuda_graph=cuda_graph or "piecewise",
            notes=(f"LingBot kernel fusion + PISA(0.25) + EasyCache(0.18) + {attn_backend} + piecewise graph",),
        )
    raise ValueError(
        f"unsupported LingBot acceleration tier {tier!r}; "
        "expected one of: exact, pisa, sol_high_fidelity, sol_balanced, refiner_sol"
    )


def accelerate_lingbot_vae(
    vae: nn.Module,
    *,
    channels_last_3d: bool = True,
    cast_bf16: bool = True,
) -> dict[str, Any]:
    """Accelerate LingBot's Wan VideoVAE with BF16 decode and ``channels_last_3d`` weights."""
    param = next(vae.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    bf16_applied = False
    if cast_bf16 and is_cuda:
        vae.to(dtype=torch.bfloat16)
        bf16_applied = True
    converted_3d = 0
    if channels_last_3d:
        for module in vae.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.to(memory_format=torch.channels_last_3d)
                converted_3d += 1
    return {
        "channels_last_3d": bool(channels_last_3d),
        "conv3d_layers_converted": converted_3d,
        "vae_bf16": bf16_applied,
        "fallbacks": [],
    }


def accelerate_lingbot_video(
    model: nn.Module,
    config: LingBotAccelerationConfig | str,
    *,
    sm_major: int | None = None,
    vae: nn.Module | None = None,
) -> dict[str, Any]:
    """Install the LingBot-Video SOL-Engine acceleration stack and return an auditable receipt."""
    param = next(model.parameters(), None)
    is_cuda = param is not None and param.is_cuda
    detected_sm = (
        torch.cuda.get_device_capability(param.device)[0]
        if is_cuda and param is not None
        else (sm_major if sm_major is not None else 0)
    )
    resolved = lingbot_sol_profile(config, sm_major=detected_sm) if isinstance(config, str) else config
    fallbacks: list[str] = []

    # 1. Projection fusion (QKV + SwiGLU).
    fusion_receipt = fuse_lingbot_projections(
        model,
        fuse_qkv=resolved.fuse_qkv,
        fuse_swiglu=resolved.fuse_swiglu,
    )

    # 2. Resolve Hopper FA3 and in-tree PISA eligibility.
    effective_attn = resolved.attention_backend
    if effective_attn == "flash_attention_3" and not (is_cuda and _resolve_fa3_usable(detected_sm)):
        effective_attn = "torch"
        if is_cuda:
            fallbacks.append("attention:flash_attention_3->torch")

    pisa_ready = bool(
        resolved.pisa_enabled and is_cuda and param is not None and piecewise_attention_available(param.device)
    )
    if resolved.pisa_enabled and is_cuda and not pisa_ready:
        if resolved.pisa_strict:
            raise RuntimeError("PISA sparse attention requested with strict=True but TMA kernel is unavailable")
        fallbacks.append("sparse_attn:pisa->dense")

    blocks = getattr(model, "blocks", ())
    num_blocks = len(blocks)
    has_moe = any(
        type(getattr(block, "ffn", None)).__name__ == "LingBotVideoSparseMoeBlock"
        or hasattr(getattr(block, "ffn", None), "router")
        for block in blocks
    )
    resolved_dense_blocks: set[int] = set()
    for b in resolved.pisa_dense_blocks:
        idx = num_blocks + int(b) if int(b) < 0 else int(b)
        if 0 <= idx < num_blocks:
            resolved_dense_blocks.add(idx)

    # 3. Configure per-block kernel fusion, FA3, and PISA metadata.
    model._lingbot_has_moe = has_moe
    model._lingbot_compact_adaln = bool(resolved.compact_adaln)
    model._lingbot_fused_kernels = bool(resolved.fused_kernels)
    model._lingbot_pisa_policy = (
        {
            "enabled": bool(pisa_ready),
            "requested": True,
            "tma_available": pisa_ready,
            "density": float(resolved.pisa_density),
            "block_size": int(resolved.pisa_block_size),
            "min_sequence_length": int(resolved.pisa_min_sequence_length),
            "dense_steps": int(resolved.pisa_dense_steps),
            "dense_last_steps": int(resolved.pisa_dense_last_steps),
            "dense_blocks": sorted(resolved_dense_blocks),
            "strict": bool(resolved.pisa_strict),
        }
        if resolved.pisa_enabled
        else None
    )
    for idx, block in enumerate(blocks):
        block._use_fused_kernels = bool(resolved.fused_kernels)
        attn = getattr(block, "attn", None)
        if attn is not None:
            attn._attention_backend = effective_attn
            attn._use_fused_kernels = bool(resolved.fused_kernels)
            attn._pisa_block_eligible = bool(pisa_ready and idx not in resolved_dense_blocks)
            attn._pisa_density = float(resolved.pisa_density)
            attn._pisa_block_size = int(resolved.pisa_block_size)
            attn._pisa_min_sequence_length = int(resolved.pisa_min_sequence_length)
            attn._pisa_strict = bool(resolved.pisa_strict)
            attn._fa3_calls = 0
            attn._pisa_calls = 0
            attn._sdpa_calls = 0

    # 4. Calibrated EasyCache spec.
    if resolved.easycache:
        model._lingbot_easycache_spec = {
            "threshold": float(resolved.easycache_threshold),
            "warmup_steps": int(resolved.easycache_warmup_steps),
            "dense_last": int(resolved.easycache_dense_last),
            "max_skip_steps": int(resolved.easycache_max_skip_steps),
            "subsample_stride": int(resolved.easycache_subsample_stride),
        }
        model._lingbot_easycaches = {}
    else:
        model._lingbot_easycache_spec = None
        model._lingbot_easycaches = {}

    # 5. Piecewise CUDA Graph runner for dense/sparse block stack segments.
    # Sparse MoE blocks perform dynamic token routing that is not CUDA-graph safe,
    # so keep MoE models cleanly eager (zero capture failures / zero RNG corruption).
    effective_cuda_graph = "off" if has_moe else resolved.cuda_graph
    if effective_cuda_graph == "piecewise":
        model._lingbot_piecewise_graph = PiecewiseBlockGraphRunner(
            warmup=resolved.cuda_graph_warmup,
            max_graphs_per_segment=resolved.cuda_graph_max_graphs,
            clone_outputs=resolved.cuda_graph_clone_outputs,
            enabled=True,
        )
        graph_receipt: dict[str, Any] = {"mode": "piecewise"}
    else:
        model._lingbot_piecewise_graph = None
        graph_receipt = (
            {"mode": "off", "reason": "moe-dynamic-routing"}
            if has_moe and resolved.cuda_graph == "piecewise"
            else {"mode": "off"}
        )

    model._lingbot_fa3_replay_calls = 0
    model._lingbot_pisa_replay_calls = 0
    vae_receipt = accelerate_lingbot_vae(vae) if vae is not None else None

    receipt: dict[str, Any] = {
        "engine": "worldfoundry-sol-lingbot",
        "config": resolved.to_dict(),
        "projection_fusion": fusion_receipt,
        "attention": {
            "dense_backend": effective_attn,
            "fused_kernels": bool(resolved.fused_kernels),
            "compact_adaln": bool(resolved.compact_adaln),
            "pisa": model._lingbot_pisa_policy or {"enabled": False},
        },
        "easycache": dict(model._lingbot_easycache_spec) if model._lingbot_easycache_spec else {"enabled": False},
        "cuda_graph": graph_receipt,
        "vae": vae_receipt,
        "fallbacks": fallbacks,
    }
    model._lingbot_acceleration_receipt = receipt
    return receipt


def lingbot_runtime_receipt(model: nn.Module) -> dict[str, Any]:
    """Collect auditable live runtime receipts from an accelerated LingBot-Video transformer."""
    base = dict(
        getattr(model, "_lingbot_acceleration_receipt", {"engine": "worldfoundry-sol-lingbot", "fallbacks": []})
    )
    fallbacks = list(base.get("fallbacks", []))
    fa3_calls = 0
    pisa_calls = 0
    sdpa_calls = 0
    for block in getattr(model, "blocks", ()):
        attn = getattr(block, "attn", None)
        if attn is not None:
            fa3_calls += int(getattr(attn, "_fa3_calls", 0))
            pisa_calls += int(getattr(attn, "_pisa_calls", 0))
            sdpa_calls += int(getattr(attn, "_sdpa_calls", 0))

    caches = getattr(model, "_lingbot_easycaches", {}) or {}
    piecewise = getattr(model, "_lingbot_piecewise_graph", None)
    graph_report = piecewise.report() if piecewise is not None else None
    if graph_report is not None:
        failed = sum(int(seg.get("capture_failed", 0)) for seg in graph_report.get("segments", {}).values())
        if failed > 0:
            fallbacks.append(f"cuda_graph:capture_failed={failed}")

    base["fallbacks"] = fallbacks
    effective_fa3 = fa3_calls + int(getattr(model, "_lingbot_fa3_replay_calls", 0))
    effective_pisa = pisa_calls + int(getattr(model, "_lingbot_pisa_replay_calls", 0))
    base["runtime"] = {
        "fa3_calls": effective_fa3,
        "pisa_calls": effective_pisa,
        "sdpa_calls": sdpa_calls,
        "easycache_branches": {branch: cache.receipt() for branch, cache in caches.items()},
        "piecewise_graph": graph_report,
    }
    return base


__all__ = [
    "LingBotAccelerationConfig",
    "LingBotEasyCache",
    "accelerate_lingbot_vae",
    "accelerate_lingbot_video",
    "fuse_lingbot_projections",
    "fused_lingbot_qk_rmsnorm_rope",
    "lingbot_fa3_attention",
    "lingbot_pisa_attention",
    "lingbot_runtime_receipt",
    "lingbot_sol_profile",
]
