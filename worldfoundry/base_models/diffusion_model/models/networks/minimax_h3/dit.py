"""MiniMax H3 packed-token audio-video DiT (single-GPU PyTorch).

Ported from SGLang's Apache-2.0
``runtime/models/dits/minimax_h3.py``. This is a faithful single-GPU port:

* ``ColumnParallelLinear`` / ``RowParallelLinear`` / ``MergedColumnParallelLinear``
  collapse to plain ``nn.Linear`` (world_size = 1).
* Ulysses sequence-parallel all-to-all, tensor-parallel all-gather, breakable
  CUDA graphs, and the layerwise-offload mixin are removed — every path here is
  rank-local.
* Dense variable-length non-causal attention is computed with
  ``F.scaled_dot_product_attention`` per packed document (``cu_seqlens``).

The numerics and the mixed fp32/bf16 parameter split are preserved exactly:
patch projections, the timestep embedder, the output heads, and
``rope.inv_freq`` stay fp32; transformer blocks are bf16. The fused CUDA/Triton
operators are represented by the portable fallbacks in :mod:`.ops` (Stage 8
swaps in the fused kernels behind the ``can_use_fused_*`` probes).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from .ops import (
    apply_qk_norm,
    indexed_gate,
    make_rms_norm,
    qk_norm_rope,
    rms_norm_indexed_scale_shift,
    rope_cos_sin_cache,
    silu_mul,
)

_BF16 = torch.bfloat16
_FP32 = torch.float32

_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "force_dense_attention",
        "step_cache",
        "step_index",
        "context_parallel",
    }
)


# --------------------------------------------------------------------------- #
# Ulysses Context Parallel (sequence-parallel) config & differentiable hops.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MiniMaxH3ContextParallelConfig:
    """Ulysses Context Parallel (sequence-parallel) configuration for MiniMax H3 DiT.

    Shards packed sequence tokens ``[T, H]`` across ``sp_size`` ranks
    (``[T_local, H]``) so every linear / FP8 / FFN / AdaLN / QK-norm+RoPE
    operation scales by ``1 / sp_size``. Inside each attention block, a
    differentiable/inference ``all_to_all`` exchanges sequence <-> head shards
    so each rank computes full-sequence attention on ``num_heads / sp_size``
    heads (e.g. ``56 -> 28`` heads/GPU at ``sp_size=2``, ``14`` at
    ``sp_size=4``) across FA3 / torch SDPA / ``sol_attn`` / ``cudnn_fp8``,
    followed by a compact logit gather after ``final_layer``.

    Works identically with ``cfg_scale == 1.0`` (distilled 7-step / 20-step /
    FL2VA) and composes orthogonally with 2-rank ``cfg_parallel`` (e.g. 4 GPUs
    = ``CFG=2 x SP=2``), ``FirstBlockCache`` / ``EasyCache`` / ``TeaCache``,
    ``Sol-Attn``, and Hopper rowwise ``FP8``.
    """

    sp_size: int = 1
    process_group: Any = None
    rank: int | None = None
    all_to_all_fn: Callable[[torch.Tensor, torch.Tensor, Any], None] | None = None
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None
    allow_cuda_graph: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sp_size, bool) or not isinstance(self.sp_size, int) or self.sp_size < 1:
            raise ValueError(f"context_parallel sp_size must be a positive integer, got {self.sp_size!r}")
        if self.rank is not None:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int):
                raise ValueError(f"context_parallel rank must be an integer or None, got {self.rank!r}")
            if not (0 <= self.rank < self.sp_size):
                raise ValueError(f"context_parallel rank {self.rank} must be in [0, {self.sp_size})")
        if not isinstance(self.allow_cuda_graph, bool):
            raise ValueError(f"context_parallel allow_cuda_graph must be bool, got {self.allow_cuda_graph!r}")


@dataclass
class _ResolvedContextParallel:
    """Active per-request Ulysses Context Parallel execution state."""

    sp_size: int
    rank: int
    process_group: Any
    all_to_all_fn: Callable[[torch.Tensor, torch.Tensor, Any], None] | None
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None
    allow_cuda_graph: bool = False
    all_to_all_calls: int = 0
    gather_calls: int = 0
    stats_gather_calls: int = 0


_ACTIVE_CONTEXT_PARALLEL: ContextVar[_ResolvedContextParallel | None] = ContextVar(
    "minimax_h3_active_context_parallel",
    default=None,
)


def _coerce_context_parallel_config(
    config: MiniMaxH3ContextParallelConfig | _ResolvedContextParallel | Mapping[str, Any] | int | bool | None,
) -> MiniMaxH3ContextParallelConfig | None:
    """Normalize caller-supplied context-parallel specification."""

    if config is None or config is False:
        return None
    if config is True:
        raise ValueError(
            "context_parallel must be an integer sp_size >= 1 (e.g. 2 or 4), "
            "MiniMaxH3ContextParallelConfig, Mapping, False, or None — not True"
        )
    if isinstance(config, _ResolvedContextParallel):
        if config.sp_size <= 1:
            return None
        return MiniMaxH3ContextParallelConfig(
            sp_size=config.sp_size,
            process_group=config.process_group,
            rank=config.rank,
            all_to_all_fn=config.all_to_all_fn,
            all_gather_fn=config.all_gather_fn,
            allow_cuda_graph=config.allow_cuda_graph,
        )
    if isinstance(config, MiniMaxH3ContextParallelConfig):
        return config if config.sp_size > 1 else None
    if isinstance(config, int):
        if config < 1:
            raise ValueError(f"context_parallel sp_size must be >= 1, got {config}")
        return MiniMaxH3ContextParallelConfig(sp_size=int(config)) if config > 1 else None
    if isinstance(config, Mapping):
        raw = dict(config)
        if "context_parallel_size" in raw and "sp_size" not in raw:
            raw["sp_size"] = raw.pop("context_parallel_size")
        parsed = MiniMaxH3ContextParallelConfig(**raw)
        return parsed if parsed.sp_size > 1 else None
    raise TypeError(
        "context_parallel must be int, MiniMaxH3ContextParallelConfig, Mapping, False, or None, "
        f"got {type(config).__name__}"
    )


def resolve_minimax_h3_context_parallel(
    config: MiniMaxH3ContextParallelConfig | _ResolvedContextParallel | Mapping[str, Any] | int | bool | None,
    *,
    num_heads: int | None = None,
) -> _ResolvedContextParallel | None:
    """Validate and resolve a Ulysses Context Parallel runtime handle."""

    if isinstance(config, _ResolvedContextParallel):
        if config.sp_size <= 1:
            return None
        if num_heads is not None and int(num_heads) % config.sp_size != 0:
            raise ValueError(
                f"MiniMax H3 context parallel requires num_attention_heads ({num_heads}) "
                f"divisible by sp_size ({config.sp_size})"
            )
        return config

    parsed = _coerce_context_parallel_config(config)
    if parsed is None or parsed.sp_size <= 1:
        return None
    sp_size = int(parsed.sp_size)
    if num_heads is not None and int(num_heads) % sp_size != 0:
        raise ValueError(
            f"MiniMax H3 context parallel requires num_attention_heads ({num_heads}) divisible by sp_size ({sp_size})"
        )

    group = parsed.process_group
    rank = parsed.rank
    needs_dist = rank is None or parsed.all_to_all_fn is None or parsed.all_gather_fn is None
    if needs_dist:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "MiniMax H3 context parallel requires an initialized torch.distributed process group "
                "(or explicit rank/all_to_all_fn/all_gather_fn for testing)."
            )
        if group is None:
            from worldfoundry.core.distributed.sequence_parallel.runtime import get_sequence_parallel_group

            sp_grp = get_sequence_parallel_group()
            if sp_grp is not None and int(torch.distributed.get_world_size(group=sp_grp)) == sp_size:
                group = sp_grp
        actual_ws = int(torch.distributed.get_world_size(group=group))
        if actual_ws != sp_size:
            raise ValueError(
                f"MiniMax H3 context parallel sp_size={sp_size} does not match process group size {actual_ws}; "
                "pass an explicit process_group or initialize the sequence parallel group."
            )
        if rank is None:
            rank = int(torch.distributed.get_rank(group=group))
    assert rank is not None
    if not (0 <= int(rank) < sp_size):
        raise ValueError(f"MiniMax H3 context parallel rank {rank} must be in [0, {sp_size})")
    return _ResolvedContextParallel(
        sp_size=sp_size,
        rank=int(rank),
        process_group=group,
        all_to_all_fn=parsed.all_to_all_fn,
        all_gather_fn=parsed.all_gather_fn,
        allow_cuda_graph=bool(parsed.allow_cuda_graph),
    )


def _cp_exec_all_to_all_single(
    recv: torch.Tensor,
    send: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
    *,
    async_op: bool = False,
) -> Any:
    cp_ctx.all_to_all_calls += 1
    if cp_ctx.all_to_all_fn is not None:
        cp_ctx.all_to_all_fn(recv, send, cp_ctx.process_group)
        return None
    return torch.distributed.all_to_all_single(recv, send, group=cp_ctx.process_group, async_op=async_op)


def _cp_exec_all_gather_dim0(
    local_tensor: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
    *,
    is_stats: bool = False,
) -> torch.Tensor:
    if is_stats:
        cp_ctx.stats_gather_calls += 1
    else:
        cp_ctx.gather_calls += 1
    src = local_tensor.contiguous()
    sp_size = cp_ctx.sp_size
    if cp_ctx.all_gather_fn is not None:
        gathered_list = [torch.empty_like(src) for _ in range(sp_size)]
        cp_ctx.all_gather_fn(gathered_list, src, cp_ctx.process_group)
        return torch.cat(gathered_list, dim=0)
    out = torch.empty((sp_size * int(src.shape[0]), *src.shape[1:]), device=src.device, dtype=src.dtype)
    if hasattr(torch.distributed, "all_gather_into_tensor"):
        torch.distributed.all_gather_into_tensor(out, src, group=cp_ctx.process_group)
        return out
    gathered_list = list(out.chunk(sp_size, dim=0))
    torch.distributed.all_gather(gathered_list, src, group=cp_ctx.process_group)
    return out


def _cp_all_gather_stats(
    stats_local: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
) -> torch.Tensor:
    """Gather a 1-D FP64 stats vector from all SP ranks and sum in rank order."""

    gathered = _cp_exec_all_gather_dim0(stats_local.reshape(1, -1), cp_ctx, is_stats=True)
    return gathered.sum(dim=0)


def _cp_shard_dim0_impl(x: torch.Tensor, cp_ctx: _ResolvedContextParallel, pad_len: int) -> torch.Tensor:
    if pad_len > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, pad_len)
        x = F.pad(x, pad)
    local_len = int(x.shape[0]) // cp_ctx.sp_size
    start = cp_ctx.rank * local_len
    return x[start : start + local_len].contiguous()


def _cp_gather_dim0_impl(x_local: torch.Tensor, cp_ctx: _ResolvedContextParallel, orig_seq_len: int) -> torch.Tensor:
    if orig_seq_len == 0 or int(x_local.shape[0]) == 0:
        return torch.empty((0, *x_local.shape[1:]), device=x_local.device, dtype=x_local.dtype)
    gathered = _cp_exec_all_gather_dim0(x_local, cp_ctx)
    if orig_seq_len < int(gathered.shape[0]):
        return gathered[:orig_seq_len]
    return gathered


class _CPShardSequence(torch.autograd.Function):
    """Differentiable sequence shard (dim 0) with all-gather backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        cp_ctx: _ResolvedContextParallel,
        orig_seq_len: int,
        pad_len: int,
    ) -> torch.Tensor:
        ctx.cp_ctx = cp_ctx
        ctx.orig_seq_len = orig_seq_len
        return _cp_shard_dim0_impl(x, cp_ctx, pad_len)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_full = _cp_gather_dim0_impl(grad_outputs[0], ctx.cp_ctx, ctx.orig_seq_len)
        return grad_full, None, None, None


class _CPGatherSequence(torch.autograd.Function):
    """Differentiable sequence all-gather (dim 0) with shard backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x_local: torch.Tensor,
        cp_ctx: _ResolvedContextParallel,
        orig_seq_len: int,
        pad_len: int,
    ) -> torch.Tensor:
        ctx.cp_ctx = cp_ctx
        ctx.pad_len = pad_len
        return _cp_gather_dim0_impl(x_local, cp_ctx, orig_seq_len)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_local = _cp_shard_dim0_impl(grad_outputs[0], ctx.cp_ctx, ctx.pad_len)
        return grad_local, None, None, None


def _cp_shard_sequence(
    x: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
    *,
    orig_seq_len: int,
    pad_len: int,
) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        out: Any = _CPShardSequence.apply(x, cp_ctx, orig_seq_len, pad_len)
        return out
    return _cp_shard_dim0_impl(x, cp_ctx, pad_len)


def _cp_gather_sequence(
    x_local: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
    *,
    orig_seq_len: int,
    pad_len: int,
) -> torch.Tensor:
    if torch.is_grad_enabled() and x_local.requires_grad:
        out: Any = _CPGatherSequence.apply(x_local, cp_ctx, orig_seq_len, pad_len)
        return out
    return _cp_gather_dim0_impl(x_local, cp_ctx, orig_seq_len)


def _cp_a2a_qkv_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange ``[T_local, H_all, D]`` -> ``[T_padded, H_local, D]`` with zero post-A2A copy."""

    s_loc, h_all, head_dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    sp_size = cp_ctx.sp_size
    h_loc = h_all // sp_size
    if s_loc == 0:
        empty = torch.empty((0, h_loc, head_dim), device=q.device, dtype=q.dtype)
        return empty, empty.clone(), empty.clone()

    s_pad = sp_size * s_loc
    use_async = (
        cp_ctx.all_to_all_fn is None
        and not (q.is_cuda and torch.cuda.is_current_stream_capturing())
        and not torch.is_grad_enabled()
    )
    # Overlap K/V transpose copies on the compute stream with Q/K NCCL A2As.
    send_q = q.reshape(s_loc, sp_size, h_loc, head_dim).transpose(0, 1).contiguous()
    recv_q = torch.empty_like(send_q)
    work_q = _cp_exec_all_to_all_single(recv_q, send_q, cp_ctx, async_op=use_async)

    send_k = k.reshape(s_loc, sp_size, h_loc, head_dim).transpose(0, 1).contiguous()
    recv_k = torch.empty_like(send_k)
    work_k = _cp_exec_all_to_all_single(recv_k, send_k, cp_ctx, async_op=use_async)

    send_v = v.reshape(s_loc, sp_size, h_loc, head_dim).transpose(0, 1).contiguous()
    recv_v = torch.empty_like(send_v)
    work_v = _cp_exec_all_to_all_single(recv_v, send_v, cp_ctx, async_op=use_async)

    if work_q is not None:
        work_q.wait()
    if work_k is not None:
        work_k.wait()
    if work_v is not None:
        work_v.wait()
    return (
        recv_q.view(s_pad, h_loc, head_dim),
        recv_k.view(s_pad, h_loc, head_dim),
        recv_v.view(s_pad, h_loc, head_dim),
    )


def _cp_a2a_out_forward_impl(
    out_full: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
) -> torch.Tensor:
    """Exchange ``[T_padded, H_local, D]`` -> ``[T_local, H_all, D]`` with zero pre-A2A copy."""

    s_pad, h_loc, head_dim = int(out_full.shape[0]), int(out_full.shape[1]), int(out_full.shape[2])
    sp_size = cp_ctx.sp_size
    h_all = sp_size * h_loc
    if s_pad == 0:
        return torch.empty((0, h_all, head_dim), device=out_full.device, dtype=out_full.dtype)
    s_loc = s_pad // sp_size
    send = out_full.contiguous().view(sp_size, s_loc, h_loc, head_dim)
    recv = torch.empty_like(send)
    _cp_exec_all_to_all_single(recv, send, cp_ctx, async_op=False)
    return recv.transpose(0, 1).contiguous().view(s_loc, h_all, head_dim)


class _CPAllToAllQKV(torch.autograd.Function):
    """Differentiable Ulysses Q/K/V all-to-all ``[T_local, H, D] -> [T_pad, H/P, D]``."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cp_ctx: _ResolvedContextParallel,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.cp_ctx = cp_ctx
        return _cp_a2a_qkv_forward_impl(q, k, v, cp_ctx)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        gq, gk, gv = grad_outputs
        return (
            _cp_a2a_out_forward_impl(gq, ctx.cp_ctx),
            _cp_a2a_out_forward_impl(gk, ctx.cp_ctx),
            _cp_a2a_out_forward_impl(gv, ctx.cp_ctx),
            None,
        )


class _CPAllToAllOut(torch.autograd.Function):
    """Differentiable inverse Ulysses output all-to-all ``[T_pad, H/P, D] -> [T_local, H, D]``."""

    @staticmethod
    def forward(
        ctx: Any,
        out_full: torch.Tensor,
        cp_ctx: _ResolvedContextParallel,
    ) -> torch.Tensor:
        ctx.cp_ctx = cp_ctx
        return _cp_a2a_out_forward_impl(out_full, cp_ctx)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        g_loc = grad_outputs[0]
        s_loc, h_all, head_dim = int(g_loc.shape[0]), int(g_loc.shape[1]), int(g_loc.shape[2])
        sp_size = ctx.cp_ctx.sp_size
        h_loc = h_all // sp_size
        if s_loc == 0:
            return torch.empty((0, h_loc, head_dim), device=g_loc.device, dtype=g_loc.dtype), None
        send = g_loc.reshape(s_loc, sp_size, h_loc, head_dim).transpose(0, 1).contiguous()
        recv = torch.empty_like(send)
        _cp_exec_all_to_all_single(recv, send, ctx.cp_ctx, async_op=False)
        return recv.view(sp_size * s_loc, h_loc, head_dim), None


def _cp_all_to_all_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        out: Any = _CPAllToAllQKV.apply(q, k, v, cp_ctx)
        return out
    return _cp_a2a_qkv_forward_impl(q, k, v, cp_ctx)


def _cp_all_to_all_out(
    out_full: torch.Tensor,
    cp_ctx: _ResolvedContextParallel,
) -> torch.Tensor:
    if torch.is_grad_enabled() and out_full.requires_grad:
        out: Any = _CPAllToAllOut.apply(out_full, cp_ctx)
        return out
    return _cp_a2a_out_forward_impl(out_full, cp_ctx)


def _resolve_probe_bounds(
    *,
    active_tokens: int,
    total_tokens: int,
    probe_start: int,
    probe_stride: int,
    row_start: int = 0,
    row_stop: int | None = None,
) -> tuple[int, int, int, int]:
    """Compute ``(global_start, global_active, local_start, local_stop)`` for strided probes.

    Guarantees that taking ``[local_start : local_stop : probe_stride]`` on each
    Ulysses SP shard selects the exact same global token indices as taking
    ``[global_start : global_active : probe_stride]`` on a single GPU.
    """

    n_active = max(1, min(int(active_tokens), int(total_tokens))) if int(total_tokens) > 0 else 0
    start_idx = int(probe_start) if 0 <= int(probe_start) < n_active else 0
    stride = max(1, int(probe_stride))
    r_start = max(0, int(row_start))
    r_stop = int(total_tokens) if row_stop is None else int(row_stop)
    if n_active <= 0 or r_stop <= r_start:
        return start_idx, n_active, 0, 0
    lo_global = max(start_idx, r_start)
    rem = (lo_global - start_idx) % stride
    if rem != 0:
        lo_global += stride - rem
    hi_global = min(n_active, r_stop)
    if lo_global >= hi_global:
        return start_idx, n_active, 0, 0
    return start_idx, n_active, lo_global - r_start, hi_global - r_start


# --------------------------------------------------------------------------- #
# Grouped-QKV checkpoint reorder (dense MHA: per-head [q,k,v] -> [q,k,v] all).
# --------------------------------------------------------------------------- #
def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder a grouped ``[q,k,v]`` checkpoint tensor to ``[q_all,k_all,v_all]``."""

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


# --------------------------------------------------------------------------- #
# Variable-length non-causal attention over a packed sequence.
# --------------------------------------------------------------------------- #
# Safe non-clipping fallbacks if cudnn_fp8 is captured without a prior warmup
# calibration pass (covers |Q|,|K| <= 224 and |V| <= 896 in E4M3).
_DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK = 0.5
_DEFAULT_CUDNN_FP8_STATIC_DESCALE_V = 2.0
_DEFAULT_CUDNN_FP8_CALIBRATION_HEADROOM = 1.25


def _normalize_host_bounds(bounds: Sequence[int] | None) -> tuple[int, ...] | None:
    if bounds is None:
        return None
    if isinstance(bounds, tuple):
        return bounds
    return tuple(int(v) for v in bounds)


def _compute_prefix_sink_tokens(
    *,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    infer_out_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    refined_prompt_embeds_length: int | torch.Tensor | None = None,
    refiner_cu_seqlens_host: tuple[int, ...] | None = None,
) -> int:
    """Compute the live ``[text | cond/ref | audio]`` prefix length before video targets.

    In MiniMax H3 packed layouts (``t2va``, ``fl2va``, ``ref2va``), non-target
    conditioning tokens (text, keyframe/reference blocks, and audio) occupy the
    leading prefix strictly before the generated video target suffix. Protecting
    the full prefix under sparse ``sol_attn`` prevents compressing keyframe
    anchors and cross-modal audio tokens.
    """

    text_len = int(text_pos.shape[0])
    if refined_prompt_embeds_length is not None and not torch.is_tensor(refined_prompt_embeds_length):
        text_len = min(text_len, max(0, int(refined_prompt_embeds_length)))
    elif refiner_cu_seqlens_host is not None and len(refiner_cu_seqlens_host) >= 2:
        text_len = min(text_len, max(0, int(refiner_cu_seqlens_host[-1])))
    cond_rows = max(0, int(img_pos.shape[0]) - int(infer_out_pos.shape[0]))
    return text_len + cond_rows + int(audio_pos.shape[0])


def _cudnn_fp8_segment_attention(
    q_seg: torch.Tensor,
    k_seg: torch.Tensor,
    v_seg: torch.Tensor,
    *,
    softmax_scale: float,
    backend_options: Mapping[str, Any] | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
) -> torch.Tensor:
    """Explicit E4M3 quantization + native cuDNN9 FP8 SDPA for ``[L, nh, hd]``."""

    from worldfoundry.core.attention.backends.dispatch import _record_provider_runtime
    from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

    opts = backend_options or {}
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8_dtype).max)
    min_scale = 1e-6

    q_bhsd = q_seg.transpose(0, 1).unsqueeze(0).contiguous()
    k_bhsd = k_seg.transpose(0, 1).unsqueeze(0).contiguous()
    v_bhsd = v_seg.transpose(0, 1).unsqueeze(0).contiguous()

    capturing = q_bhsd.is_cuda and torch.cuda.is_current_stream_capturing()
    has_explicit_scale = any(k in opts for k in ("descale_q", "descale_k", "descale_v", "static_descale"))
    scale_o = opts.get("scale_o")
    if has_explicit_scale:
        default_qk = float(opts.get("static_descale", _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK))
        default_v = float(opts.get("static_descale", _DEFAULT_CUDNN_FP8_STATIC_DESCALE_V))
        dq = float(opts.get("descale_q", default_qk))
        dk = float(opts.get("descale_k", default_qk))
        dv = float(opts.get("descale_v", default_v))
    elif capturing:
        if calibrated_descales is not None:
            dq, dk, dv = calibrated_descales
        else:
            dq = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK
            dk = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_QK
            dv = _DEFAULT_CUDNN_FP8_STATIC_DESCALE_V
    else:
        # Batch all 3 reductions into one device-to-host sync (3x fewer syncs
        # than per-tensor .item()) and record calibrated per-layer descales for
        # subsequent CUDA Graph capture.
        amax_q, amax_k, amax_v = torch.stack((q_bhsd.abs().amax(), k_bhsd.abs().amax(), v_bhsd.abs().amax())).tolist()
        dq = max(float(amax_q) / fp8_max, min_scale)
        dk = max(float(amax_k) / fp8_max, min_scale)
        dv = max(float(amax_v) / fp8_max, min_scale)
        if on_calibrate is not None:
            headroom = max(1.0, float(opts.get("calibration_headroom", _DEFAULT_CUDNN_FP8_CALIBRATION_HEADROOM)))
            on_calibrate((dq * headroom, dk * headroom, dv * headroom))

    qf = (q_bhsd * (1.0 / dq)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    kf = (k_bhsd * (1.0 / dk)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    vf = (v_bhsd * (1.0 / dv)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    try:
        attended = native_cudnn_fp8_sdpa(
            qf,
            kf,
            vf,
            scale=softmax_scale,
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
            scale_o=None if scale_o is None else float(scale_o),
        )
    except Exception:
        _record_provider_runtime("cudnn_fp8", attempts=1, errors=1)
        raise
    _record_provider_runtime("cudnn_fp8", attempts=1, successes=1)
    return attended.squeeze(0).transpose(0, 1)


_FLASH_ATTN_3_UNSET = object()
_FLASH_ATTN_3_INTERFACE: Any = _FLASH_ATTN_3_UNSET
_FLASH_ATTN_3_HAS_BACKWARD: bool | None = None
_FLASH_ATTN_3_MAX_HEAD_DIM: int = 256


def _get_flash_attn_3_interface() -> Any | None:
    """Lazily import and cache Hopper ``flash_attn_interface`` once per process."""

    global _FLASH_ATTN_3_INTERFACE, _FLASH_ATTN_3_HAS_BACKWARD, _FLASH_ATTN_3_MAX_HEAD_DIM
    if _FLASH_ATTN_3_INTERFACE is _FLASH_ATTN_3_UNSET:
        try:
            import flash_attn_interface as _fa3
        except Exception:
            _FLASH_ATTN_3_INTERFACE = None
            _FLASH_ATTN_3_HAS_BACKWARD = False
            _FLASH_ATTN_3_MAX_HEAD_DIM = 0
        else:
            _FLASH_ATTN_3_INTERFACE = _fa3
            try:
                import flash_attn_config as _fa3_cfg

                flags = getattr(_fa3_cfg, "CONFIG", {}).get("build_flags", {})
                _FLASH_ATTN_3_HAS_BACKWARD = not bool(flags.get("FLASHATTENTION_DISABLE_BACKWARD", False))
                enabled_dims = [
                    dim
                    for dim in (64, 96, 128, 192, 256)
                    if not bool(flags.get(f"FLASHATTENTION_DISABLE_HDIM{dim}", False))
                ]
                _FLASH_ATTN_3_MAX_HEAD_DIM = max(enabled_dims) if enabled_dims else 0
            except Exception:
                _FLASH_ATTN_3_HAS_BACKWARD = True
                _FLASH_ATTN_3_MAX_HEAD_DIM = 256
    return _FLASH_ATTN_3_INTERFACE


def _fa3_needs_autograd_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return True if backward is needed on a forward-only FlashAttention-3 build."""

    if not torch.is_grad_enabled() or not (q.requires_grad or k.requires_grad or v.requires_grad):
        return False
    _get_flash_attn_3_interface()
    return not bool(_FLASH_ATTN_3_HAS_BACKWARD)


def _fa3_head_dim_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return whether ``q``/``k``/``v`` head dimensions are compiled into the FA3 wheel."""

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    head_dim = int(q.shape[-1])
    if head_dim <= 0 or int(k.shape[-1]) != head_dim or int(v.shape[-1]) != head_dim:
        return False
    _get_flash_attn_3_interface()
    return head_dim <= int(_FLASH_ATTN_3_MAX_HEAD_DIM)


def _can_use_direct_flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return whether Hopper ``flash_attn_3`` direct SM90 path is eligible."""

    if torch.compiler.is_compiling() or _fa3_needs_autograd_fallback(q, k, v):
        return False
    if getattr(torch.version, "hip", None):
        return False
    if not (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device == k.device == v.device
        and tuple(torch.cuda.get_device_capability(q.device)) == (9, 0)
    ):
        return False
    fa3 = _get_flash_attn_3_interface()
    if fa3 is None or not hasattr(fa3, "flash_attn_func"):
        return False
    return _fa3_head_dim_supported(q, k, v)


def _flash_attention_3_segment_attention(
    q_seg: torch.Tensor,
    k_seg: torch.Tensor,
    v_seg: torch.Tensor,
    *,
    softmax_scale: float,
    out_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct zero-overhead Hopper FlashAttention-3 for one ``[L, nh, hd]`` segment.

    Bypasses generic ``attention_forward`` signature/rearrange/lock overhead on
    every transformer block. When autograd is inactive on ``q``/``k``/``v``,
    invokes ``torch.ops.flash_attn_3.fwd`` directly and can write in-place into
    a contiguous-last-dim ``out_buf`` (eliminating the trailing-pad copy buffer).
    Under autograd, delegates to ``flash_attn_interface.flash_attn_func``.
    """

    from worldfoundry.core.attention.backends.dispatch import _record_provider_runtime

    fa3 = _get_flash_attn_3_interface()
    assert fa3 is not None
    q_bshd = (q_seg if q_seg.stride(-1) == 1 else q_seg.contiguous()).unsqueeze(0)
    k_bshd = (k_seg if k_seg.stride(-1) == 1 else k_seg.contiguous()).unsqueeze(0)
    v_bshd = (v_seg if v_seg.stride(-1) == 1 else v_seg.contiguous()).unsqueeze(0)
    needs_grad = torch.is_grad_enabled() and (q_seg.requires_grad or k_seg.requires_grad or v_seg.requires_grad)
    can_write_inplace = (
        out_buf is not None
        and out_buf.stride(-1) == 1
        and out_buf.shape == q_seg.shape
        and out_buf.dtype == q_seg.dtype
        and out_buf.device == q_seg.device
    )
    try:
        if not needs_grad and hasattr(torch.ops, "flash_attn_3") and hasattr(torch.ops.flash_attn_3, "fwd"):
            out_target = out_buf.unsqueeze(0) if can_write_inplace and out_buf is not None else None
            attended = torch.ops.flash_attn_3.fwd(
                q_bshd,
                k_bshd,
                v_bshd,
                None,
                None,
                None,
                out_target,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                float(softmax_scale),
                False,
                -1,
                -1,
                0,
                0.0,
                True,
                None,
                1,
                None,
                0,
            )[0]
            if out_buf is not None and not can_write_inplace:
                out_buf.copy_(attended.squeeze(0))
                attended = out_buf.unsqueeze(0)
        else:
            result = fa3.flash_attn_func(q_bshd, k_bshd, v_bshd, softmax_scale=float(softmax_scale))
            attended = result[0] if isinstance(result, tuple) else result
            if out_buf is not None:
                out_buf.copy_(attended.squeeze(0))
                attended = out_buf.unsqueeze(0)
    except Exception:
        _record_provider_runtime("flash_attention_3", attempts=1, errors=1)
        raise
    _record_provider_runtime("flash_attention_3", attempts=1, successes=1)
    return attended.squeeze(0)


def _flash_attention_3_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    zero_init_out: bool = False,
) -> torch.Tensor:
    """Single-launch zero-sync Hopper FlashAttention-3 packed varlen kernel."""

    from worldfoundry.core.attention.backends.dispatch import _record_provider_runtime

    fa3 = _get_flash_attn_3_interface()
    assert fa3 is not None
    cu_cont = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
    q_c = q if q.stride(-1) == 1 else q.contiguous()
    k_c = k if k.stride(-1) == 1 else k.contiguous()
    v_c = v if v.stride(-1) == 1 else v.contiguous()
    max_len = int(max_seqlen)
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    try:
        if not needs_grad and hasattr(torch.ops, "flash_attn_3") and hasattr(torch.ops.flash_attn_3, "fwd"):
            out_target = torch.zeros(q_c.shape, device=q_c.device, dtype=q_c.dtype) if zero_init_out else None
            attended = torch.ops.flash_attn_3.fwd(
                q_c,
                k_c,
                v_c,
                None,
                None,
                None,
                out_target,
                cu_cont,
                cu_cont,
                None,
                None,
                None,
                max_len,
                max_len,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                float(softmax_scale),
                False,
                -1,
                -1,
                0,
                0.0,
                True,
                None,
                1,
                None,
                0,
            )[0]
        else:
            result = fa3.flash_attn_varlen_func(
                q_c,
                k_c,
                v_c,
                cu_cont,
                cu_cont,
                max_len,
                max_len,
                softmax_scale=float(softmax_scale),
            )
            attended = result[0] if isinstance(result, tuple) else result
    except Exception:
        _record_provider_runtime("flash_attention_3", attempts=1, errors=1)
        raise
    _record_provider_runtime("flash_attention_3", attempts=1, successes=1)
    return attended


def _resolve_sm90_sol_kv_splits(
    options: dict[str, Any],
    *,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Clamp ``kv_splits`` to SM90 route-group bounds and unlock ``allow_multi_split``."""

    raw_splits = options.get("kv_splits", 1)
    if isinstance(raw_splits, bool) or not isinstance(raw_splits, int) or raw_splits <= 1:
        return options
    route_groups = ((seq_len + 63) // 64 + 63) // 64
    is_sm90 = device.type == "cuda" and tuple(torch.cuda.get_device_capability(device)) == (9, 0)
    if not is_sm90:
        options["kv_splits"] = 1
        options.pop("allow_multi_split", None)
        return options
    max_valid = min(raw_splits, route_groups)
    clamped = 4 if max_valid >= 4 else (2 if max_valid >= 2 else 1)
    options["kv_splits"] = clamped
    if clamped == 1:
        options.pop("allow_multi_split", None)
    else:
        # Unlock SM90 multi-split automatically unless caller explicitly passed False.
        options.setdefault("allow_multi_split", True)
    return options


def _dispatch_attention_segment(
    q_seg: torch.Tensor,
    k_seg: torch.Tensor,
    v_seg: torch.Tensor,
    *,
    softmax_scale: float,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    prefix_sink_tokens: int | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
    out_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute dense or sparse non-causal attention on one ``[L, nh, hd]`` segment."""

    if (
        backend is None
        or backend == "torch"
        or (
            backend == "flash_attention_3"
            and (_fa3_needs_autograd_fallback(q_seg, k_seg, v_seg) or not _fa3_head_dim_supported(q_seg, k_seg, v_seg))
        )
    ):
        q_bhsd = q_seg.transpose(0, 1).unsqueeze(0)
        k_bhsd = k_seg.transpose(0, 1).unsqueeze(0)
        v_bhsd = v_seg.transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=False, scale=softmax_scale)
        result = attended.squeeze(0).transpose(0, 1)
        if out_buf is not None:
            out_buf.copy_(result)
            return out_buf
        return result

    if backend == "flash_attention_3" and _can_use_direct_flash_attention_3(q_seg, k_seg, v_seg):
        return _flash_attention_3_segment_attention(
            q_seg,
            k_seg,
            v_seg,
            softmax_scale=softmax_scale,
            out_buf=out_buf,
        )

    if backend == "cudnn_fp8":
        result = _cudnn_fp8_segment_attention(
            q_seg,
            k_seg,
            v_seg,
            softmax_scale=softmax_scale,
            backend_options=backend_options,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )
        if out_buf is not None:
            out_buf.copy_(result)
            return out_buf
        return result

    from worldfoundry.core.attention.backends.dispatch import attention_forward

    resolved_options = dict(backend_options) if backend_options else None
    if backend == "sol_attn":
        seq_len = int(q_seg.shape[0])
        if prefix_sink_tokens is not None and 0 < int(prefix_sink_tokens) < seq_len:
            opts = {} if resolved_options is None else resolved_options
            if opts.get("compress_kv", True) and "sink_tokens" not in opts:
                opts["sink_tokens"] = int(prefix_sink_tokens)
                opts.setdefault("sink_start", 0)
                resolved_options = opts
        raw_kv_splits = resolved_options.get("kv_splits", 1) if resolved_options is not None else 1
        if (
            resolved_options is not None
            and not isinstance(raw_kv_splits, bool)
            and isinstance(raw_kv_splits, int)
            and raw_kv_splits > 1
        ):
            resolved_options = _resolve_sm90_sol_kv_splits(
                resolved_options,
                seq_len=seq_len,
                device=q_seg.device,
            )

    attended = attention_forward(
        q_seg.unsqueeze(0),
        k_seg.unsqueeze(0),
        v_seg.unsqueeze(0),
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        scale=softmax_scale,
        backend=backend,
        backend_options=resolved_options,
    )
    result = attended.squeeze(0)
    if out_buf is not None:
        out_buf.copy_(result)
        return out_buf
    return result


def _can_use_varlen_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return whether ``aten._flash_attention_forward`` can run zero-sync packed varlen."""

    return (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device == k.device == v.device
        and torch.backends.cuda.is_flash_attention_available()
        and torch.cuda.get_device_capability(q.device) >= (8, 0)
        and hasattr(torch.ops.aten, "_flash_attention_forward")
    )


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
    cu_seqlens_host: tuple[int, ...] | None = None,
    max_seqlen: int | None = None,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    ignore_trailing_pad: bool = False,
    prefix_sink_tokens: int | None = None,
    calibrated_descales: tuple[float, float, float] | None = None,
    on_calibrate: Callable[[tuple[float, float, float]], None] | None = None,
) -> torch.Tensor:
    """Dense or backend-routed non-causal attention per packed document.

    ``q``/``k``/``v``: ``[T, num_heads, head_dim]`` packed rows.
    ``cu_seqlens``: cumulative sequence boundaries ``[num_docs + 1]``.

    Execution tiers (zero GPU→CPU sync on all hot paths):
    1. Single document (or single document + ignored alignment pad) with
       ``cu_seqlens_host``: dispatches directly to Hopper FA3, cuDNN9 SDPA,
       ``sol_attn``, or ``cudnn_fp8`` without copying (FA3 writes in-place into
       ``out[:used]`` when trailing pad is present).
    2. Multi-document packed varlen on SM90 (``flash_attention_3``) or SM80+
       (``torch``) CUDA bf16/fp16 when ``max_seqlen`` is supplied: single fused
       varlen kernel launch with zero ``.tolist()`` sync and zero-filled
       uncovered tails.
    3. Portable per-segment fallback using ``cu_seqlens_host`` or ``cu_seqlens``.
    """

    total = int(q.shape[0])
    if total == 0:
        return torch.empty(q.shape, device=q.device, dtype=q.dtype)

    # Single-document unpadded fast path: Hopper FA3 / cuDNN9 SDPA / configured backend.
    if (
        cu_seqlens_host is not None
        and len(cu_seqlens_host) == 2
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[1] == total
    ):
        return _dispatch_attention_segment(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            prefix_sink_tokens=prefix_sink_tokens,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )

    # Single-document + alignment-padding tail fast path: skip wasted
    # attention launches over padding rows that are never read downstream.
    if (
        ignore_trailing_pad
        and cu_seqlens_host is not None
        and len(cu_seqlens_host) == 3
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[2] == total
    ):
        used = cu_seqlens_host[1]
        if used == total:
            return _dispatch_attention_segment(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
            )
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            if used == 0:
                return torch.zeros(q.shape, device=q.device, dtype=q.dtype)
            seg_out = _dispatch_attention_segment(
                q[:used],
                k[:used],
                v[:used],
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
            )
            return F.pad(seg_out, (0, 0, 0, 0, 0, total - used))
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        if used > 0:
            _dispatch_attention_segment(
                q[:used],
                k[:used],
                v[:used],
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
                out_buf=out[:used],
            )
        if used < total:
            out[used:].zero_()
        return out

    host_bounds_valid = cu_seqlens_host is None or (
        len(cu_seqlens_host) >= 2
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[-1] <= total
        and all(left <= right for left, right in itertools.pairwise(cu_seqlens_host))
    )

    # Multi-document packed varlen zero-sync fast path via Hopper FA3.
    if (
        backend == "flash_attention_3"
        and host_bounds_valid
        and max_seqlen is not None
        and int(max_seqlen) > 0
        and cu_seqlens.is_cuda
        and cu_seqlens.device == q.device
        and cu_seqlens.dtype == torch.int32
        and _can_use_direct_flash_attention_3(q, k, v)
    ):
        attended = _flash_attention_3_varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            softmax_scale=softmax_scale,
            zero_init_out=cu_seqlens_host is None,
        )
        if cu_seqlens_host is not None and cu_seqlens_host[-1] < total:
            attended[cu_seqlens_host[-1] :].zero_()
        return attended

    # Multi-document packed varlen zero-sync fast path via aten._flash_attention_forward.
    if (
        (backend is None or backend == "torch")
        and host_bounds_valid
        and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad))
        and max_seqlen is not None
        and int(max_seqlen) > 0
        and cu_seqlens.is_cuda
        and cu_seqlens.device == q.device
        and cu_seqlens.dtype == torch.int32
        and _can_use_varlen_flash_attention(q, k, v)
    ):
        cu_cont = cu_seqlens if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()
        max_len = int(max_seqlen)
        attended = torch.ops.aten._flash_attention_forward(
            q_c,
            k_c,
            v_c,
            cu_cont,
            cu_cont,
            max_len,
            max_len,
            0.0,
            False,
            False,
            scale=softmax_scale,
        )[0]
        if cu_seqlens_host is not None and cu_seqlens_host[-1] < total:
            attended[cu_seqlens_host[-1] :].zero_()
        return attended

    bounds = cu_seqlens_host if cu_seqlens_host is not None else tuple(int(v) for v in cu_seqlens.tolist())
    if len(bounds) == 2 and bounds[0] == 0 and bounds[1] == total:
        return _dispatch_attention_segment(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            prefix_sink_tokens=prefix_sink_tokens,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
        )

    if ignore_trailing_pad and len(bounds) == 3 and bounds[0] == 0 and bounds[2] == total:
        used = bounds[1]
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            if used == 0:
                return torch.zeros(q.shape, device=q.device, dtype=q.dtype)
            seg_out = _dispatch_attention_segment(
                q[:used],
                k[:used],
                v[:used],
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
            )
            return F.pad(seg_out, (0, 0, 0, 0, 0, total - used))
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        if used > 0:
            _dispatch_attention_segment(
                q[:used],
                k[:used],
                v[:used],
                softmax_scale=softmax_scale,
                backend=backend,
                backend_options=backend_options,
                prefix_sink_tokens=prefix_sink_tokens,
                calibrated_descales=calibrated_descales,
                on_calibrate=on_calibrate,
                out_buf=out[:used],
            )
        if used < total:
            out[used:].zero_()
        return out

    tight_coverage = bool(bounds) and bounds[0] == 0 and bounds[-1] == total
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        pieces: list[torch.Tensor] = []
        cursor = 0
        for start, stop in itertools.pairwise(bounds):
            start_i, stop_i = int(start), int(stop)
            if stop_i <= start_i:
                continue
            if start_i > cursor:
                pieces.append(q.new_zeros((start_i - cursor, *q.shape[1:])))
            pieces.append(
                _dispatch_attention_segment(
                    q[start_i:stop_i],
                    k[start_i:stop_i],
                    v[start_i:stop_i],
                    softmax_scale=softmax_scale,
                    backend=backend,
                    backend_options=backend_options,
                    calibrated_descales=calibrated_descales,
                    on_calibrate=on_calibrate,
                )
            )
            cursor = stop_i
        if cursor < total:
            pieces.append(q.new_zeros((total - cursor, *q.shape[1:])))
        return torch.cat(pieces, dim=0) if pieces else torch.zeros(q.shape, device=q.device, dtype=q.dtype)

    out = (
        torch.empty(q.shape, device=q.device, dtype=q.dtype)
        if tight_coverage
        else torch.zeros(q.shape, device=q.device, dtype=q.dtype)
    )
    for start, stop in itertools.pairwise(bounds):
        length = int(stop) - int(start)
        if length <= 0:
            continue
        _dispatch_attention_segment(
            q[start:stop],
            k[start:stop],
            v[start:stop],
            softmax_scale=softmax_scale,
            backend=backend,
            backend_options=backend_options,
            calibrated_descales=calibrated_descales,
            on_calibrate=on_calibrate,
            out_buf=out[start:stop],
        )
    return out


# --------------------------------------------------------------------------- #
# Timestep embedder (fp32 sinusoidal, cos-before-sin).
# --------------------------------------------------------------------------- #
class MiniMaxH3TimeEmbedder(nn.Module):
    """fp32 sinusoidal timestep MLP; concatenates ``cos`` before ``sin``."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = nn.Linear(arch.timestep_input_dim, arch.time_embed_hidden_size, bias=True, dtype=_FP32)
        self.proj_out = nn.Linear(arch.time_embed_hidden_size, arch.time_embed_dim, bias=True, dtype=_FP32)
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32 (cos concatenated before sin)."""

        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=_FP32, device=t.device) / half)
            self._frequency_cache = freqs
        args = t.to(_FP32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden = self.proj_in(t_freq)
        hidden = F.silu(hidden)
        return self.proj_out(hidden)


# --------------------------------------------------------------------------- #
# 3D RoPE frequency builder.
# --------------------------------------------------------------------------- #
class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary 0.75)."""

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, inv_freq_len, dtype=_FP32) / float(inv_freq_len)))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


# --------------------------------------------------------------------------- #
# Attention (fused qkv -> per-head qk RMSNorm -> RoPE -> varlen SDPA -> out).
# --------------------------------------------------------------------------- #
class MiniMaxH3Attention(nn.Module):
    """Fused-QKV self-attention with per-head RMSNorm, 3D RoPE, and varlen SDPA."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.attention_fn = _varlen_attention
        self.attention_backend: str | None = None
        self.attention_backend_options: dict[str, Any] | None = None
        self.dense_attention_backend: str = "torch"
        self._cudnn_fp8_calibrated_descales: tuple[float, float, float] | None = None
        # Fused qkv: three logical matrices stored as one tensor.
        self.qkv_proj = nn.Linear(arch.hidden_size, 3 * self.inner_dim, bias=False, dtype=_BF16)
        self.q_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = make_rms_norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, arch.hidden_size, bias=False, dtype=_BF16)

    def _update_cudnn_fp8_calibration(self, descales: tuple[float, float, float]) -> None:
        if self._cudnn_fp8_calibrated_descales is None:
            self._cudnn_fp8_calibrated_descales = descales
        else:
            prev_q, prev_k, prev_v = self._cudnn_fp8_calibrated_descales
            self._cudnn_fp8_calibrated_descales = (
                max(prev_q, descales[0]),
                max(prev_k, descales[1]),
                max(prev_v, descales[2]),
            )

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
        dense_backend: str | None = None,
    ) -> None:
        """Bind a request-scoped attention provider (``AttentionBackendConfigurable``)."""

        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        self._cudnn_fp8_calibrated_descales = None
        if dense_backend is not None:
            self.dense_attention_backend = normalize_attention_backend(str(dense_backend))
        if backend is None:
            self.attention_backend = None
            self.attention_backend_options = None
            if dense_backend is None:
                self.dense_attention_backend = "torch"
            return

        normalized = normalize_attention_backend(str(backend))
        self.attention_backend = normalized
        self.attention_backend_options = dict(backend_options) if backend_options else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cp_seq_len: int | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed rows -> [T, hidden]."""

        total = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        if rope_cache is None:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        else:
            cos_sin_cache, positions = rope_cache
            q, k = qk_norm_rope(q, k, self.q_norm, self.k_norm, cos_sin_cache, positions)
        cp_ctx = _ACTIVE_CONTEXT_PARALLEL.get() if cp_seq_len is not None else None
        cp_pad_len = 0
        if cp_ctx is not None and cp_ctx.sp_size > 1:
            if self.num_heads % cp_ctx.sp_size != 0:
                raise ValueError(
                    f"MiniMax H3 context parallel requires num_attention_heads ({self.num_heads}) "
                    f"divisible by sp_size ({cp_ctx.sp_size})"
                )
            q, k, v = _cp_all_to_all_qkv(q, k, v, cp_ctx)
            padded_len = int(q.shape[0])
            orig_len = int(cp_seq_len) if cp_seq_len is not None else padded_len
            cp_pad_len = max(0, padded_len - orig_len)
            if cp_pad_len > 0:
                q = q[:orig_len]
                k = k[:orig_len]
                v = v[:orig_len]
        if attention_groups is None:
            if self.attention_fn is _varlen_attention:
                eff_backend = self.dense_attention_backend if force_dense_attention else self.attention_backend
                eff_options = None if force_dense_attention else self.attention_backend_options
                out = _varlen_attention(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_seqlens,
                    softmax_scale=self.softmax_scale,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    backend=eff_backend,
                    backend_options=eff_options,
                    ignore_trailing_pad=ignore_trailing_pad,
                    prefix_sink_tokens=prefix_sink_tokens,
                    calibrated_descales=self._cudnn_fp8_calibrated_descales,
                    on_calibrate=self._update_cudnn_fp8_calibration,
                )
            else:
                out = self.attention_fn(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
        else:
            # Variants can expose global media with document-local text without
            # materializing a quadratic attention mask or duplicating projections.
            out = torch.zeros_like(q)
            for queries, keys in attention_groups:
                attended = F.scaled_dot_product_attention(
                    q.index_select(0, queries).transpose(0, 1).unsqueeze(0),
                    k.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    v.index_select(0, keys).transpose(0, 1).unsqueeze(0),
                    scale=self.softmax_scale,
                )
                out.index_copy_(0, queries, attended.squeeze(0).transpose(0, 1))
        if cp_ctx is not None and cp_ctx.sp_size > 1:
            if cp_pad_len > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, cp_pad_len))
            out = _cp_all_to_all_out(out, cp_ctx)
        out = out.reshape(total, self.num_heads * self.head_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU MLP.
# --------------------------------------------------------------------------- #
class MiniMaxH3MLP(nn.Module):
    """SwiGLU MLP with fused ``gate|up`` projection (``fc1``) and ``fc2`` down-proj."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        # gate and up fused into fc1.
        self.fc1 = nn.Linear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False, dtype=_BF16)
        self.fc2 = nn.Linear(arch.ffn_hidden_size, arch.hidden_size, bias=False, dtype=_BF16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        hidden = silu_mul(hidden)
        return self.fc2(hidden)


# --------------------------------------------------------------------------- #
# AdaLN projection (SiLU + zero-init linear over unique condition embeddings).
# --------------------------------------------------------------------------- #
class MiniMaxH3AdalnProj(nn.Module):
    """[M, t_dim] -> expand_ratio tensors of [M * modality_num, H]."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = nn.Linear(arch.time_embed_dim, out_features, bias=True, dtype=_BF16)

    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        return self.linear(adaln_input)

    def split_output(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.split_output(self.project_local(adaln_input))


# --------------------------------------------------------------------------- #
# Token refiner (pre-norm transformer, no AdaLN/RoPE).
# --------------------------------------------------------------------------- #
class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Pre-norm transformer block used only on text tokens (no AdaLN / RoPE)."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)

    @property
    def attention_backend(self) -> str | None:
        return self.attn.attention_backend

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
        dense_backend: str | None = None,
    ) -> None:
        self.attn.set_attention_backend(
            backend,
            backend_options=backend_options,
            dense_backend=dense_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    """Shallow text refiner run once per request before packing into the DiT stream."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(arch) for _ in range(arch.token_refiner_num_layers)])
        self.final_norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.attention_backend: str | None = None

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        for block in self.blocks:
            block.set_attention_backend(backend, backend_options=backend_options)
        self.attention_backend = self.blocks[0].attention_backend if self.blocks else backend

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, cu_seqlens_host=cu_seqlens_host)
        return self.final_norm(x)


# --------------------------------------------------------------------------- #
# DiT block (AdaLN-modulated attention + MLP with gated residuals).
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTBlock(nn.Module):
    """AdaLN-modulated attention + MLP with per-row gated residuals.

    ``combined_indices`` select among ``num_unique_timesteps × 3`` modality
    vectors (video / audio / text) produced by :class:`MiniMaxH3AdalnProj`.
    """

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm1 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = make_rms_norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch)
        self.mlp = MiniMaxH3MLP(arch)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    @property
    def attention_backend(self) -> str | None:
        return self.attn.attention_backend

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
        dense_backend: str | None = None,
    ) -> None:
        self.attn.set_attention_backend(
            backend,
            backend_options=backend_options,
            dense_backend=dense_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        attention_groups: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        ffn_conditioner: Any = None,
        cp_seq_len: int | None = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = rms_norm_indexed_scale_shift(x, self.norm1, shift_msa, scale_msa, combined_indices, dtype=x.dtype)
        if attention_groups is None:
            h = self.attn(
                h,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                cp_seq_len=cp_seq_len,
            )
        else:
            h = self.attn(
                h,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                attention_groups=attention_groups,
                cp_seq_len=cp_seq_len,
            )
        x = indexed_gate(residual, gate_msa, h, combined_indices, dtype=x.dtype)

        residual = x
        if ffn_conditioner is not None:
            x = ffn_conditioner(x)
        h = rms_norm_indexed_scale_shift(x, self.norm2, shift_mlp, scale_mlp, combined_indices, dtype=x.dtype)
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices, dtype=x.dtype)


# --------------------------------------------------------------------------- #
# Final layer (single-modality AdaLN -> fp32 -> dual video/audio heads).
# --------------------------------------------------------------------------- #
class MiniMaxH3FinalLayer(nn.Module):
    """Final AdaLN (timestep-only) then fp32 video-patch and audio heads."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.norm = make_rms_norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = nn.Linear(arch.hidden_size, video_patch_dim, bias=True, dtype=_FP32)
        self.audio_out = nn.Linear(arch.hidden_size, arch.audio_latents_dim, bias=True, dtype=_FP32)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = rms_norm_indexed_scale_shift(x, self.norm, shift, scale, inverse_indices, dtype=x.dtype)
        h = h.to(_FP32)  # full precision through both output projections
        return self.video_out(h), self.audio_out(h)


# --------------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------------- #
class MiniMaxH3DiTModel(nn.Module):
    """Single-GPU MiniMax H3 packed-token audio-video DiT."""

    def __init__(self, config: MiniMaxH3DiTArchConfig | None = None, **overrides: Any) -> None:
        super().__init__()
        arch = config if config is not None else MiniMaxH3DiTArchConfig(**overrides)
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        self.attention_backend: str | None = None
        self.dense_attention_backend: str = "torch"
        self._block_graph_runner: Any = None
        self._piecewise_graph: Any = None
        self._piecewise_segment_names: tuple[str, ...] = ()
        self._piecewise_segment_bounds: dict[str, tuple[int, int]] = {}
        self._dense_initial_steps: int = 0
        self._context_parallel: MiniMaxH3ContextParallelConfig | None = None
        self._cp_window_stats: dict[str, int] = {
            "all_to_all_calls": 0,
            "gather_calls": 0,
            "stats_gather_calls": 0,
        }
        self._cp_lifetime_stats: dict[str, int] = {
            "all_to_all_calls": 0,
            "gather_calls": 0,
            "stats_gather_calls": 0,
        }
        self._graph_captured_receipts: dict[
            tuple[int, int, bool, int | None, tuple[int, int, int] | None], dict[str, Any]
        ] = {}
        self._graph_replayed_window: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
            "cp_all_to_all_calls": 0,
        }
        self._graph_replayed_lifetime: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
            "cp_all_to_all_calls": 0,
        }
        self._acceleration_metadata: dict[str, Any] = {}

        video_patch_dim = arch.latents_dim * arch.patch_size[0] * arch.patch_size[1] * arch.patch_size[2]
        self.video_patch_proj = nn.Linear(video_patch_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.audio_patch_proj = nn.Linear(arch.audio_latents_dim, arch.hidden_size, bias=True, dtype=_FP32)
        self.condition_proj = nn.Linear(arch.text_dim, arch.hidden_size, bias=True, dtype=_BF16)
        self.time_embedder = MiniMaxH3TimeEmbedder(arch)
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(arch)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(arch) for _ in range(arch.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(arch)

    def invalidate_cuda_graphs(self) -> None:
        """Drop any captured block-stack CUDA graphs and release stale graph pools."""

        if self._block_graph_runner is not None:
            self._block_graph_runner.invalidate()
            if hasattr(self._block_graph_runner, "_pool"):
                self._block_graph_runner._pool = None
        if self._piecewise_graph is not None:
            self._piecewise_graph.invalidate()
            for seg_runner in getattr(self._piecewise_graph, "_segments", {}).values():
                if hasattr(seg_runner, "_pool"):
                    seg_runner._pool = None
        self._graph_captured_receipts.clear()

    def set_attention_backend(
        self,
        backend: str | None,
        *,
        backend_options: Mapping[str, Any] | None = None,
        dense_layers: Sequence[int] = (),
        include_refiner: bool = False,
        dense_backend: str = "torch",
    ) -> None:
        """Configure the attention backend across DiT blocks (``AttentionBackendConfigurable``).

        ``dense_layers`` pins sensitive blocks (e.g. ``(0, 1, -1)`` under sparse
        ``sol_attn``) to ``dense_backend`` (default ``"torch"``, or Hopper
        ``"flash_attention_3"`` when requested) while routing the rest to
        ``backend``. ``token_refiner`` stays on exact ``"torch"`` SDPA unless
        ``include_refiner=True`` (and resets cleanly if previously enabled).
        """

        from worldfoundry.core.attention.backends.probe import normalize_attention_backend

        self.invalidate_cuda_graphs()
        resolved_dense = "torch" if backend is None else normalize_attention_backend(str(dense_backend))
        self.dense_attention_backend = resolved_dense
        num_layers = len(self.blocks)
        pinned = {(idx + num_layers if idx < 0 else idx) for idx in dense_layers}
        for idx, block in enumerate(self.blocks):
            if backend is not None and idx in pinned:
                block.set_attention_backend(resolved_dense, backend_options=None, dense_backend=resolved_dense)
            else:
                block.set_attention_backend(backend, backend_options=backend_options, dense_backend=resolved_dense)
        if include_refiner:
            self.token_refiner.set_attention_backend(backend, backend_options=backend_options)
        elif backend is None or self.token_refiner.attention_backend is not None:
            self.token_refiner.set_attention_backend(None, backend_options=None)
        if backend is None:
            self.attention_backend = None
        else:
            self.attention_backend = normalize_attention_backend(str(backend))

    def set_context_parallel(
        self,
        config: MiniMaxH3ContextParallelConfig | _ResolvedContextParallel | Mapping[str, Any] | int | bool | None,
    ) -> dict[str, Any]:
        """Configure Ulysses Context Parallel (sequence parallelism) on this DiT."""

        parsed = _coerce_context_parallel_config(config)
        if parsed is not None and self.num_attention_heads % parsed.sp_size != 0:
            raise ValueError(
                f"MiniMax H3 context parallel requires num_attention_heads ({self.num_attention_heads}) "
                f"divisible by sp_size ({parsed.sp_size})"
            )
        if parsed != self._context_parallel:
            self.invalidate_cuda_graphs()
            self._context_parallel = parsed
        sp_size = int(parsed.sp_size) if parsed is not None else 1
        return {
            "enabled": sp_size > 1,
            "sp_size": sp_size,
            "local_heads": int(self.num_attention_heads // sp_size),
            "allow_cuda_graph": bool(parsed.allow_cuda_graph) if parsed is not None else False,
        }

    def _resolve_call_context_parallel(self, override: Any) -> _ResolvedContextParallel | None:
        if override is False or override == 1:
            return None
        target = self._context_parallel if override is None else override
        if target is None:
            return None
        return resolve_minimax_h3_context_parallel(target, num_heads=self.num_attention_heads)

    def _record_cp_stats(
        self,
        cp_ctx: _ResolvedContextParallel,
        *,
        prev_a2a: int,
        prev_gather: int,
        prev_stats: int,
    ) -> None:
        da2a = max(0, int(cp_ctx.all_to_all_calls) - prev_a2a)
        dg = max(0, int(cp_ctx.gather_calls) - prev_gather)
        ds = max(0, int(cp_ctx.stats_gather_calls) - prev_stats)
        for bucket in (self._cp_window_stats, self._cp_lifetime_stats):
            bucket["all_to_all_calls"] += da2a
            bucket["gather_calls"] += dg
            bucket["stats_gather_calls"] += ds

    def _snapshot_slice_counters(self, start: int, stop: int) -> tuple[int, int, dict[str, int], int]:
        """Snapshot lifetime FP8/FFN/attention/CP counters for one block slice."""

        from worldfoundry.core.attention.backends.dispatch import attention_provider_runtime_report

        from .triton_fp8_swiglu import FusedFP8SwiGLUMLP

        fp8_total = 0
        ffn_total = 0
        for idx in range(start, stop):
            for mod in self.blocks[idx].modules():
                if getattr(mod, "_worldfoundry_quantization_layer", False):
                    fp8_total += int(getattr(mod, "low_precision_kernel_calls", 0))
                elif isinstance(mod, FusedFP8SwiGLUMLP):
                    ffn_total += int(mod.fused_calls)
        prov_report = attention_provider_runtime_report()
        prov_counts = {
            str(name): int(stats.get("successes", 0))
            for name, stats in prov_report.items()
            if isinstance(stats, Mapping)
        }
        cp_ctx = _ACTIVE_CONTEXT_PARALLEL.get()
        cp_a2a = int(cp_ctx.all_to_all_calls) if cp_ctx is not None else 0
        return fp8_total, ffn_total, prov_counts, cp_a2a

    def _record_graph_replay_receipt(
        self,
        key: tuple[int, int, bool, int | None, tuple[int, int, int] | None],
    ) -> None:
        """Credit baked-in kernel counts for one verified CUDA Graph replay."""

        baked = self._graph_captured_receipts.get(key)
        if not baked:
            return
        for bucket in (self._graph_replayed_window, self._graph_replayed_lifetime):
            bucket["fp8_kernel_calls"] += int(baked.get("fp8_kernel_calls", 0))
            bucket["fused_ffn_calls"] += int(baked.get("fused_ffn_calls", 0))
            bucket["cp_all_to_all_calls"] = int(bucket.get("cp_all_to_all_calls", 0)) + int(
                baked.get("cp_all_to_all_calls", 0)
            )
            prov_dst = bucket["attention_provider_calls"]
            for name, count in baked.get("attention_provider_calls", {}).items():
                prov_dst[name] = prov_dst.get(name, 0) + int(count)
        cp_ctx = _ACTIVE_CONTEXT_PARALLEL.get()
        if cp_ctx is not None:
            cp_ctx.all_to_all_calls += int(baked.get("cp_all_to_all_calls", 0))

    def reset_acceleration_window(self) -> None:
        """Reset per-request telemetry counters on graphs, FP8 linears, and fused FFNs."""

        from worldfoundry.core.acceleration.quantization.linear import reset_quantization_runtime_window
        from worldfoundry.core.attention.backends.dispatch import reset_attention_provider_runtime

        reset_quantization_runtime_window(self)
        reset_attention_provider_runtime()
        for module in self.modules():
            reset_fn = getattr(module, "reset_request_window", None)
            if callable(reset_fn) and not getattr(module, "_worldfoundry_quantization_layer", False):
                reset_fn()
        if self._block_graph_runner is not None:
            self._block_graph_runner.begin_request_window()
        if self._piecewise_graph is not None:
            self._piecewise_graph.begin_request_window()
        self._graph_replayed_window = {
            "fp8_kernel_calls": 0,
            "fused_ffn_calls": 0,
            "attention_provider_calls": {},
            "cp_all_to_all_calls": 0,
        }
        self._cp_window_stats = {
            "all_to_all_calls": 0,
            "gather_calls": 0,
            "stats_gather_calls": 0,
        }

    def acceleration_report(self) -> dict[str, Any]:
        """Return auditable runtime receipts for quantization, attention, fusion, and graphs."""

        from worldfoundry.core.acceleration.quantization.linear import quantization_runtime_report
        from worldfoundry.core.attention.backends.dispatch import attention_provider_runtime_report

        from .triton_fp8_swiglu import FusedFP8SwiGLUMLP

        fused_ffn_layers = 0
        fused_ffn_calls = 0
        fused_ffn_fallbacks = 0
        for module in self.modules():
            if isinstance(module, FusedFP8SwiGLUMLP):
                fused_ffn_layers += 1
                if module._request_window_active:
                    fused_ffn_calls += int(module.request_fused_calls)
                    fused_ffn_fallbacks += int(module.request_fallback_calls)
                else:
                    fused_ffn_calls += int(module.fused_calls)
                    fused_ffn_fallbacks += int(module.fallback_calls)

        graph_report: Mapping[str, Any] | None = None
        if self._block_graph_runner is not None:
            graph_report = self._block_graph_runner.report()
        elif self._piecewise_graph is not None:
            graph_report = self._piecewise_graph.report()

        block_backends: dict[str, int] = {}
        for block in self.blocks:
            name = str(block.attention_backend or "torch")
            block_backends[name] = block_backends.get(name, 0) + 1

        quant_report = quantization_runtime_report(self)
        replayed_fp8 = int(self._graph_replayed_window["fp8_kernel_calls"])
        replayed_ffn = int(self._graph_replayed_window["fused_ffn_calls"])
        replayed_prov = dict(self._graph_replayed_window["attention_provider_calls"])
        if isinstance(quant_report, dict):
            quant_report = dict(quant_report)
            quant_report["graph_replayed_low_precision_kernel_calls"] = replayed_fp8
            quant_report["effective_low_precision_kernel_calls"] = (
                int(quant_report.get("low_precision_kernel_calls", 0)) + replayed_fp8
            )

        prov_runtime = attention_provider_runtime_report()
        prov_effective: dict[str, int] = {}
        for name, stats in prov_runtime.items():
            if isinstance(stats, Mapping):
                prov_effective[str(name)] = int(stats.get("successes", 0))
        for name, count in replayed_prov.items():
            prov_effective[name] = prov_effective.get(name, 0) + int(count)

        cp_cfg = self._context_parallel
        cp_sp = int(cp_cfg.sp_size) if cp_cfg is not None else 1

        return {
            "attention_backend": self.attention_backend or "torch",
            "dense_attention_backend": self.dense_attention_backend,
            "block_attention_backends": block_backends,
            "dense_initial_steps": int(self._dense_initial_steps),
            "attention_provider_runtime": prov_runtime,
            "attention_provider_runtime_effective": prov_effective,
            "quantization": quant_report,
            "fused_fp8_swiglu": {
                "layers": fused_ffn_layers,
                "fused_calls": fused_ffn_calls,
                "graph_replayed_fused_calls": replayed_ffn,
                "effective_fused_calls": fused_ffn_calls + replayed_ffn,
                "fallback_calls": fused_ffn_fallbacks,
            },
            "context_parallel": {
                "enabled": cp_sp > 1,
                "sp_size": cp_sp,
                "local_heads": int(self.num_attention_heads // cp_sp),
                "allow_cuda_graph": bool(cp_cfg.allow_cuda_graph) if cp_cfg is not None else False,
                "all_to_all_calls": int(self._cp_window_stats["all_to_all_calls"]),
                "gather_calls": int(self._cp_window_stats["gather_calls"]),
                "stats_gather_calls": int(self._cp_window_stats["stats_gather_calls"]),
                "lifetime": dict(self._cp_lifetime_stats),
            },
            "cuda_graph": graph_report,
            "graph_replayed_receipts": {
                "window": {
                    "fp8_kernel_calls": replayed_fp8,
                    "fused_ffn_calls": replayed_ffn,
                    "attention_provider_calls": replayed_prov,
                    "cp_all_to_all_calls": int(self._graph_replayed_window.get("cp_all_to_all_calls", 0)),
                },
                "lifetime": {
                    "fp8_kernel_calls": int(self._graph_replayed_lifetime["fp8_kernel_calls"]),
                    "fused_ffn_calls": int(self._graph_replayed_lifetime["fused_ffn_calls"]),
                    "attention_provider_calls": dict(self._graph_replayed_lifetime["attention_provider_calls"]),
                    "cp_all_to_all_calls": int(self._graph_replayed_lifetime.get("cp_all_to_all_calls", 0)),
                },
            },
            "metadata": dict(self._acceleration_metadata),
        }

    # -- weight-dtype guard ------------------------------------------------- #
    def post_load_weights(self) -> None:
        """Assert the mixed fp32/bf16 parameter split survived weight loading."""

        for name in _MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER:
            param = self.get_parameter(name)
            if param.dtype != _FP32:
                raise ValueError(f"{name} must stay fp32 after load, got {param.dtype}.")
        if self.rope.inv_freq.dtype != _FP32:
            raise ValueError(f"rope.inv_freq must stay fp32 after load, got {self.rope.inv_freq.dtype}.")

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        ids = pos_info.get("position_ids") if isinstance(pos_info, dict) else getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        value = psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    @staticmethod
    def _psp_optional_field(psp: Any, field: str) -> Any:
        return psp.get(field) if isinstance(psp, dict) else getattr(psp, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
        refiner_cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Project and refine request-static text conditioning once."""

        if refiner_cu_seqlens.ndim != 1 or int(refiner_cu_seqlens.numel()) < 2:
            raise ValueError(
                f"refiner cu_seqlens must be 1-D with at least 2 entries, got {list(refiner_cu_seqlens.shape)}"
            )
        bounds = (
            list(refiner_cu_seqlens_host)
            if refiner_cu_seqlens_host is not None
            else [int(value) for value in refiner_cu_seqlens.tolist()]
        )
        if bounds[0] != 0 or any(right < left for left, right in itertools.pairwise(bounds)):
            raise ValueError(f"refiner cu_seqlens must start at 0 and be non-decreasing, got {bounds}")
        text_len = bounds[-1]
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError(
                f"refiner cu_seqlens live text length must be in [1, {int(prompt_embeds.shape[0])}], got {text_len}"
            )
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=self.condition_proj.weight.dtype)
        # Preserve caller-provided document boundaries (single document [0, text_len]
        # or caller-supplied per-segment bounds).
        refiner_cu = refiner_cu_seqlens.to(device=device, dtype=torch.int32)
        text_embed = self.condition_proj(text_rows)
        return self.token_refiner(text_embed, cu_seqlens=refiner_cu, cu_seqlens_host=tuple(bounds))

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the request-static RoPE cos/sin cache and position index."""

        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        seq_len = int(img_position_ids.shape[1])
        rope_freqs = self.rope(img_position_ids).to(device)
        return (
            rope_cos_sin_cache(rope_freqs, dtype=self.condition_proj.weight.dtype),
            torch.arange(seq_len, device=device, dtype=torch.long),
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None = None,
        refiner_cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the packed embedding sequence [S, H] bf16 and t_emb [M, t_dim] fp32."""

        if refined_prompt_embeds_length is None:
            text_len = (
                int(refiner_cu_seqlens_host[-1])
                if refiner_cu_seqlens_host is not None
                else int(refiner_cu_seqlens[-1].item())
            )
        elif torch.is_tensor(refined_prompt_embeds_length):
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(text_embeddings_selected.shape[0])}], got {text_len}"
            )
        text_pos = text_pos[:text_len]
        dtype = self.condition_proj.weight.dtype
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(device=device, dtype=dtype)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "refined prompt embeddings must have hidden width "
                    f"{self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                text_embeddings_selected,
                refiner_cu_seqlens,
                device=device,
                refiner_cu_seqlens_host=refiner_cu_seqlens_host,
            )

        seq_len = int(x.shape[1])
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)

        if text_pos.numel():
            embeddings.index_add_(0, text_pos, text_embed[:text_len].to(dtype))

        # Latent embedders stay fp32; only scatter into their packed positions.
        if img_pos.numel():
            x_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(_FP32)
            video_embed = self.video_patch_proj(x_rows)
            embeddings.index_add_(0, img_pos, video_embed.to(dtype))

        if audio_pos.numel():
            audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(_FP32)
            audio_embed = self.audio_patch_proj(audio_rows)
            embeddings.index_add_(0, audio_pos, audio_embed.to(dtype))

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward returning ``(video_logits, audio_logits)``."""

        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: {sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")
        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(_required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info")
        text_pos = self._pos_ids(_required_kwarg(kwargs, "text_pos_info"), "text_pos_info")
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"), "img_pos_for_infer_output_info"
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(torch.int32)
        cu_seqlens_host = _normalize_host_bounds(self._psp_optional_field(psp, "cu_seqlens_q_host"))
        raw_max_seqlen = self._psp_optional_field(psp, "max_seqlen_q")
        max_seqlen = None if raw_max_seqlen is None else int(raw_max_seqlen)
        ignore_trailing_pad = bool(self._psp_optional_field(psp, "ignore_trailing_pad"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q").to(torch.int32)
        refiner_cu_host = _normalize_host_bounds(self._psp_optional_field(refiner_psp, "cu_seqlens_q_host"))

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(f"token_tags must cover the full packed sequence ({seq_len}), got {token_tags.shape[0]}.")
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}")
        device = x.device

        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids).to(device)
            rope_cache = (
                rope_cos_sin_cache(rope_freqs, dtype=self.condition_proj.weight.dtype),
                torch.arange(seq_len, device=device, dtype=torch.long),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        adaln_input = F.silu(t_emb).to(self.condition_proj.weight.dtype)
        inverse_indices = inverse_indices.to(device)
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags.clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != seq_len:
                raise ValueError(
                    f"block_token_tags must cover the packed sequence ({seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(block_token_tags, inverse_indices, alpha=MINIMAX_H3_ADALN_MODALITY_NUM)

        prefix_sink_tokens = _compute_prefix_sink_tokens(
            text_pos=text_pos,
            img_pos=img_pos,
            infer_out_pos=infer_out_pos,
            audio_pos=audio_pos,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            refiner_cu_seqlens_host=refiner_cu_host,
        )
        force_dense_attention = bool(kwargs.get("force_dense_attention", False))
        step_cache = kwargs.get("step_cache")
        step_index = kwargs.get("step_index")
        forward_blocks_kwargs: dict[str, Any] = {
            "adaln_input": adaln_input,
            "combined_indices": block_combined,
            "rope_cache": rope_cache,
            "cu_seqlens": cu_seqlens.to(device),
            "inverse_indices": inverse_indices,
            "cu_seqlens_host": cu_seqlens_host,
            "max_seqlen": max_seqlen,
            "ignore_trailing_pad": ignore_trailing_pad,
            "prefix_sink_tokens": prefix_sink_tokens,
            "force_dense_attention": force_dense_attention,
        }
        if step_cache is not None:
            forward_blocks_kwargs["step_cache"] = step_cache
            forward_blocks_kwargs["step_index"] = step_index
        if "context_parallel" in kwargs:
            forward_blocks_kwargs["context_parallel"] = kwargs.get("context_parallel")
        video_logits, audio_logits = self.forward_blocks(decoder_input, **forward_blocks_kwargs)

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(f"update_mask length mismatch: {update_mask.shape[0]} != {video_logits.shape[0]}")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits

    def _run_block_slice(
        self,
        start: int,
        stop: int,
        hidden: torch.Tensor,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int | None,
        ignore_trailing_pad: bool,
        prefix_sink_tokens: int | None,
        force_dense_attention: bool = False,
        cp_seq_len: int | None = None,
        cp_graph_key: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Pure-tensor block slice callable for whole/piecewise CUDA Graph capture."""

        capturing = hidden.is_cuda and torch.cuda.is_current_stream_capturing()
        before_snapshot = self._snapshot_slice_counters(start, stop) if capturing else None
        rope_cache = (cos_sin_cache, positions)
        for index in range(start, stop):
            hidden = self.blocks[index](
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                cp_seq_len=cp_seq_len,
            )
        if before_snapshot is not None:
            fp8_after, ffn_after, prov_after, cp_a2a_after = self._snapshot_slice_counters(start, stop)
            fp8_before, ffn_before, prov_before, cp_a2a_before = before_snapshot
            prov_delta = {
                name: count - prov_before.get(name, 0)
                for name, count in prov_after.items()
                if count > prov_before.get(name, 0)
            }
            self._graph_captured_receipts[(start, stop, bool(force_dense_attention), cp_seq_len, cp_graph_key)] = {
                "fp8_kernel_calls": max(0, fp8_after - fp8_before),
                "fused_ffn_calls": max(0, ffn_after - ffn_before),
                "attention_provider_calls": prov_delta,
                "cp_all_to_all_calls": max(0, cp_a2a_after - cp_a2a_before),
            }
        return hidden

    def _record_piecewise_skips(self, start: int, stop: int) -> None:
        """Credit ``PiecewiseBlockGraphRunner`` skips for every segment inside ``[start, stop)``."""

        if self._piecewise_graph is None:
            return
        for seg_name in self._piecewise_segment_names:
            bounds = self._piecewise_segment_bounds.get(seg_name)
            if bounds is not None and bounds[0] >= start and bounds[1] <= stop:
                self._piecewise_graph.record_skip(seg_name)

    def _compute_step_cache_probe(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        active_tokens: int,
        stride: int,
        probe_start: int = 0,
        total_tokens: int | None = None,
        row_start: int = 0,
        row_stop: int | None = None,
    ) -> torch.Tensor:
        """Compute a lightweight Block-0 AdaLN-modulated probe for step caching."""

        total = int(hidden.shape[0]) if total_tokens is None else int(total_tokens)
        step_stride = max(1, int(stride))
        _, _, local_start, local_stop = _resolve_probe_bounds(
            active_tokens=active_tokens,
            total_tokens=total,
            probe_start=probe_start,
            probe_stride=step_stride,
            row_start=row_start,
            row_stop=row_stop,
        )
        if local_start >= local_stop:
            return hidden[:0].contiguous()
        h_sub = hidden[local_start:local_stop:step_stride].contiguous()
        idx_sub = combined_indices[local_start:local_stop:step_stride].contiguous()
        if len(self.blocks) > 0:
            raw_block0 = getattr(self.blocks[0], "_orig_mod", self.blocks[0])
            adaln_proj = getattr(raw_block0, "adaln_proj", None)
            norm1 = getattr(raw_block0, "norm1", None)
            if callable(adaln_proj) and isinstance(norm1, nn.RMSNorm):
                params = adaln_proj(adaln_input)
                if isinstance(params, tuple) and len(params) >= 2:
                    shift_msa, scale_msa = params[0], params[1]
                    return rms_norm_indexed_scale_shift(
                        h_sub,
                        norm1,
                        shift_msa,
                        scale_msa,
                        idx_sub,
                        dtype=h_sub.dtype,
                    )
        return h_sub

    def _execute_block_range(
        self,
        start: int,
        stop: int,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        cp_seq_len: int | None = None,
        cp_graph_key: tuple[int, int, int] | None = None,
        allow_cuda_graph: bool = True,
    ) -> torch.Tensor:
        """Execute ``self.blocks[start:stop]`` via whole/piecewise CUDA Graph or eager."""

        if stop <= start:
            return hidden

        # Resolve host bounds once outside graph capture/replay so no block
        # ever triggers a GPU->CPU synchronization inside a captured graph.
        if (
            allow_cuda_graph
            and (self._block_graph_runner is not None or self._piecewise_graph is not None)
            and cu_seqlens_host is None
            and max_seqlen is None
        ):
            cu_seqlens_host = tuple(int(v) for v in cu_seqlens.tolist())

        cos_sin_cache, positions = rope_cache
        if allow_cuda_graph and self._block_graph_runner is not None and start == 0 and stop == len(self.blocks):
            prev_replays = int(self._block_graph_runner.stats["replay"])
            out = self._block_graph_runner(
                hidden,
                adaln_input,
                combined_indices,
                cos_sin_cache,
                positions,
                cu_seqlens,
                cu_seqlens_host,
                max_seqlen,
                ignore_trailing_pad,
                prefix_sink_tokens,
                bool(force_dense_attention),
                cp_seq_len,
                cp_graph_key,
            )
            if int(self._block_graph_runner.stats["replay"]) > prev_replays:
                self._record_graph_replay_receipt(
                    (0, len(self.blocks), bool(force_dense_attention), cp_seq_len, cp_graph_key)
                )
            return out

        if allow_cuda_graph and self._piecewise_graph is not None:
            by_start: dict[int, tuple[str, int]] = {}
            for seg_name in self._piecewise_segment_names:
                bounds = self._piecewise_segment_bounds.get(seg_name)
                if bounds is not None:
                    by_start[bounds[0]] = (seg_name, bounds[1])
            sorted_starts = sorted(by_start)
            cursor = start
            while cursor < stop:
                seg = by_start.get(cursor)
                if seg is not None and seg[1] <= stop:
                    seg_name, seg_stop = seg
                    seg_runner = self._piecewise_graph._segments[seg_name]
                    prev_replays = int(seg_runner.stats["replay"])
                    hidden = self._piecewise_graph.run_segment(
                        seg_name,
                        hidden,
                        adaln_input,
                        combined_indices,
                        cos_sin_cache,
                        positions,
                        cu_seqlens,
                        cu_seqlens_host,
                        max_seqlen,
                        ignore_trailing_pad,
                        prefix_sink_tokens,
                        bool(force_dense_attention),
                        cp_seq_len,
                        cp_graph_key,
                    )
                    if int(seg_runner.stats["replay"]) > prev_replays:
                        self._record_graph_replay_receipt(
                            (cursor, seg_stop, bool(force_dense_attention), cp_seq_len, cp_graph_key)
                        )
                    cursor = seg_stop
                else:
                    next_stop = stop
                    for candidate in sorted_starts:
                        if cursor < candidate < stop:
                            next_stop = candidate
                            break
                    hidden = self._run_block_slice(
                        cursor,
                        next_stop,
                        hidden,
                        adaln_input,
                        combined_indices,
                        cos_sin_cache,
                        positions,
                        cu_seqlens,
                        cu_seqlens_host,
                        max_seqlen,
                        ignore_trailing_pad,
                        prefix_sink_tokens,
                        bool(force_dense_attention),
                        cp_seq_len,
                        cp_graph_key,
                    )
                    cursor = next_stop
            return hidden

        return self._run_block_slice(
            start,
            stop,
            hidden,
            adaln_input,
            combined_indices,
            cos_sin_cache,
            positions,
            cu_seqlens,
            cu_seqlens_host,
            max_seqlen,
            ignore_trailing_pad,
            prefix_sink_tokens,
            bool(force_dense_attention),
            cp_seq_len,
            cp_graph_key,
        )

    def _execute_transformer_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        cp_seq_len: int | None = None,
        cp_graph_key: tuple[int, int, int] | None = None,
        allow_cuda_graph: bool = True,
    ) -> torch.Tensor:
        """Execute ``self.blocks`` eagerly or via whole/piecewise CUDA Graph replay."""

        return self._execute_block_range(
            0,
            len(self.blocks),
            hidden,
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
            cp_seq_len=cp_seq_len,
            cp_graph_key=cp_graph_key,
            allow_cuda_graph=allow_cuda_graph,
        )

    @staticmethod
    def _shard_block_inputs(
        hidden: torch.Tensor,
        *,
        combined_indices: torch.Tensor,
        inverse_indices: torch.Tensor | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cp_ctx: _ResolvedContextParallel,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor],
        int,
        int,
        int,
        int,
    ]:
        orig_seq_len = int(hidden.shape[0])
        pad_len = (-orig_seq_len) % cp_ctx.sp_size
        padded_seq_len = orig_seq_len + pad_len
        local_seq_len = padded_seq_len // cp_ctx.sp_size
        row_start = cp_ctx.rank * local_seq_len
        row_stop = row_start + local_seq_len
        hidden_local = _cp_shard_sequence(hidden, cp_ctx, orig_seq_len=orig_seq_len, pad_len=pad_len)
        combined_local = _cp_shard_dim0_impl(combined_indices, cp_ctx, pad_len)
        inverse_local = _cp_shard_dim0_impl(inverse_indices, cp_ctx, pad_len) if inverse_indices is not None else None
        cos_sin_cache, positions = rope_cache
        rope_local = (
            _cp_shard_dim0_impl(cos_sin_cache, cp_ctx, pad_len),
            _cp_shard_dim0_impl(positions, cp_ctx, pad_len),
        )
        return hidden_local, combined_local, inverse_local, rope_local, orig_seq_len, pad_len, row_start, row_stop

    def _run_transformer_blocks_inner(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int | None,
        ignore_trailing_pad: bool,
        prefix_sink_tokens: int | None,
        force_dense_attention: bool,
        step_cache: Any,
        step_index: int | None,
        cp_ctx: _ResolvedContextParallel | None,
        orig_seq_len: int,
        row_start: int,
        row_stop: int,
    ) -> torch.Tensor:
        cp_seq_len = orig_seq_len if cp_ctx is not None else None
        cp_graph_key = (int(cp_ctx.sp_size), int(cp_ctx.rank), id(cp_ctx.process_group)) if cp_ctx is not None else None
        allow_cg = True if cp_ctx is None else bool(cp_ctx.allow_cuda_graph)

        if step_cache is not None and step_index is not None:
            if (
                ignore_trailing_pad
                and cu_seqlens_host is None
                and cu_seqlens.ndim == 1
                and int(cu_seqlens.numel()) == 3
                and not (hidden.is_cuda and torch.cuda.is_current_stream_capturing())
            ):
                cu_seqlens_host = tuple(int(v) for v in cu_seqlens.tolist())
            active_tokens = (
                int(cu_seqlens_host[1])
                if (
                    ignore_trailing_pad
                    and cu_seqlens_host is not None
                    and len(cu_seqlens_host) == 3
                    and cu_seqlens_host[0] == 0
                    and cu_seqlens_host[2] == orig_seq_len
                )
                else orig_seq_len
            )
            cache_mode = str(getattr(getattr(step_cache, "config", None), "mode", "easycache"))
            stride = max(1, int(getattr(step_cache, "probe_stride", 16)))
            probe_start = int(getattr(step_cache, "probe_start", 0))

            if cache_mode == "firstblock":
                if len(self.blocks) == 0:
                    return hidden
                h1 = self._execute_block_range(
                    0,
                    1,
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=combined_indices,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    ignore_trailing_pad=ignore_trailing_pad,
                    prefix_sink_tokens=prefix_sink_tokens,
                    force_dense_attention=force_dense_attention,
                    cp_seq_len=cp_seq_len,
                    cp_graph_key=cp_graph_key,
                    allow_cuda_graph=allow_cg,
                )
                _, _, loc_start, loc_stop = _resolve_probe_bounds(
                    active_tokens=active_tokens,
                    total_tokens=orig_seq_len,
                    probe_start=probe_start,
                    probe_stride=stride,
                    row_start=row_start,
                    row_stop=row_stop,
                )
                if loc_start < loc_stop:
                    probe = h1[loc_start:loc_stop:stride].float() - hidden[loc_start:loc_stop:stride].float()
                else:
                    probe = h1[:0].float()
                run_kwargs: dict[str, Any] = {
                    "step": int(step_index),
                    "hidden_in": h1,
                    "probe": probe,
                    "active_tokens": active_tokens,
                    "force_compute": bool(force_dense_attention),
                    "compute_fn": lambda: self._execute_block_range(
                        1,
                        len(self.blocks),
                        h1,
                        adaln_input=adaln_input,
                        combined_indices=combined_indices,
                        rope_cache=rope_cache,
                        cu_seqlens=cu_seqlens,
                        cu_seqlens_host=cu_seqlens_host,
                        max_seqlen=max_seqlen,
                        ignore_trailing_pad=ignore_trailing_pad,
                        prefix_sink_tokens=prefix_sink_tokens,
                        force_dense_attention=force_dense_attention,
                        cp_seq_len=cp_seq_len,
                        cp_graph_key=cp_graph_key,
                        allow_cuda_graph=allow_cg,
                    ),
                    "on_hit": lambda: self._record_piecewise_skips(1, len(self.blocks)),
                }
                if cp_ctx is not None:
                    run_kwargs["sp_ctx"] = cp_ctx
                    run_kwargs["total_tokens"] = orig_seq_len
                    run_kwargs["row_start"] = row_start
                    run_kwargs["row_stop"] = row_stop
                return step_cache.run(**run_kwargs)

            probe = self._compute_step_cache_probe(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                active_tokens=active_tokens,
                stride=stride,
                probe_start=probe_start,
                total_tokens=orig_seq_len,
                row_start=row_start,
                row_stop=row_stop,
            )
            run_kwargs = {
                "step": int(step_index),
                "hidden_in": hidden,
                "probe": probe,
                "active_tokens": active_tokens,
                "force_compute": bool(force_dense_attention),
                "compute_fn": lambda: self._execute_transformer_blocks(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=combined_indices,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    ignore_trailing_pad=ignore_trailing_pad,
                    prefix_sink_tokens=prefix_sink_tokens,
                    force_dense_attention=force_dense_attention,
                    cp_seq_len=cp_seq_len,
                    cp_graph_key=cp_graph_key,
                    allow_cuda_graph=allow_cg,
                ),
                "on_hit": lambda: self._record_piecewise_skips(0, len(self.blocks)),
            }
            if cp_ctx is not None:
                run_kwargs["sp_ctx"] = cp_ctx
                run_kwargs["total_tokens"] = orig_seq_len
                run_kwargs["row_start"] = row_start
                run_kwargs["row_stop"] = row_stop
            return step_cache.run(**run_kwargs)

        return self._execute_transformer_blocks(
            hidden,
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ignore_trailing_pad=ignore_trailing_pad,
            prefix_sink_tokens=prefix_sink_tokens,
            force_dense_attention=force_dense_attention,
            cp_seq_len=cp_seq_len,
            cp_graph_key=cp_graph_key,
            allow_cuda_graph=allow_cg,
        )

    def _run_transformer_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        step_cache: Any = None,
        step_index: int | None = None,
        context_parallel: Any = None,
        _cp_ctx: _ResolvedContextParallel | None = None,
        _cp_orig_seq_len: int | None = None,
        _cp_row_start: int = 0,
        _cp_row_stop: int | None = None,
    ) -> torch.Tensor:
        """Execute ``self.blocks`` with optional step caching and Ulysses context parallel."""

        if _cp_ctx is not None:
            orig_len = int(hidden.shape[0]) if _cp_orig_seq_len is None else int(_cp_orig_seq_len)
            r_stop = orig_len if _cp_row_stop is None else int(_cp_row_stop)
            return self._run_transformer_blocks_inner(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                step_cache=step_cache,
                step_index=step_index,
                cp_ctx=_cp_ctx,
                orig_seq_len=orig_len,
                row_start=int(_cp_row_start),
                row_stop=r_stop,
            )

        own_cp = self._resolve_call_context_parallel(context_parallel)
        if own_cp is None:
            seq_len = int(hidden.shape[0])
            return self._run_transformer_blocks_inner(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                step_cache=step_cache,
                step_index=step_index,
                cp_ctx=None,
                orig_seq_len=seq_len,
                row_start=0,
                row_stop=seq_len,
            )

        prev_a2a = int(own_cp.all_to_all_calls)
        prev_gather = int(own_cp.gather_calls)
        prev_stats = int(own_cp.stats_gather_calls)
        hidden_local, combined_local, _, rope_local, orig_seq_len, pad_len, row_start, row_stop = (
            self._shard_block_inputs(
                hidden,
                combined_indices=combined_indices,
                inverse_indices=None,
                rope_cache=rope_cache,
                cp_ctx=own_cp,
            )
        )
        token = _ACTIVE_CONTEXT_PARALLEL.set(own_cp)
        try:
            hidden_local = self._run_transformer_blocks_inner(
                hidden_local,
                adaln_input=adaln_input,
                combined_indices=combined_local,
                rope_cache=rope_local,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                step_cache=step_cache,
                step_index=step_index,
                cp_ctx=own_cp,
                orig_seq_len=orig_seq_len,
                row_start=row_start,
                row_stop=row_stop,
            )
            return _cp_gather_sequence(hidden_local, own_cp, orig_seq_len=orig_seq_len, pad_len=pad_len)
        finally:
            _ACTIVE_CONTEXT_PARALLEL.reset(token)
            self._record_cp_stats(own_cp, prev_a2a=prev_a2a, prev_gather=prev_gather, prev_stats=prev_stats)

    def forward_blocks(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        inverse_indices: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int | None = None,
        ignore_trailing_pad: bool = False,
        prefix_sink_tokens: int | None = None,
        force_dense_attention: bool = False,
        step_cache: Any = None,
        step_index: int | None = None,
        context_parallel: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the packed residual stream through the shared blocks and output heads."""

        cp_ctx = self._resolve_call_context_parallel(context_parallel)
        if cp_ctx is None:
            hidden = self._run_transformer_blocks(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                step_cache=step_cache,
                step_index=step_index,
                context_parallel=False,
            )
            return self.final_layer(hidden, adaln_input=adaln_input, inverse_indices=inverse_indices)

        prev_a2a = int(cp_ctx.all_to_all_calls)
        prev_gather = int(cp_ctx.gather_calls)
        prev_stats = int(cp_ctx.stats_gather_calls)
        hidden_local, combined_local, inverse_local, rope_local, orig_seq_len, pad_len, row_start, row_stop = (
            self._shard_block_inputs(
                hidden,
                combined_indices=combined_indices,
                inverse_indices=inverse_indices,
                rope_cache=rope_cache,
                cp_ctx=cp_ctx,
            )
        )
        assert inverse_local is not None
        token = _ACTIVE_CONTEXT_PARALLEL.set(cp_ctx)
        try:
            hidden_local = self._run_transformer_blocks(
                hidden_local,
                adaln_input=adaln_input,
                combined_indices=combined_local,
                rope_cache=rope_local,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ignore_trailing_pad=ignore_trailing_pad,
                prefix_sink_tokens=prefix_sink_tokens,
                force_dense_attention=force_dense_attention,
                step_cache=step_cache,
                step_index=step_index,
                _cp_ctx=cp_ctx,
                _cp_orig_seq_len=orig_seq_len,
                _cp_row_start=row_start,
                _cp_row_stop=row_stop,
            )
            local_video, local_audio = self.final_layer(
                hidden_local,
                adaln_input=adaln_input,
                inverse_indices=inverse_local,
            )
            if local_video.dtype == local_audio.dtype:
                video_dim = int(local_video.shape[-1])
                audio_dim = int(local_audio.shape[-1])
                packed_logits = torch.cat((local_video, local_audio), dim=-1)
                gathered = _cp_gather_sequence(
                    packed_logits,
                    cp_ctx,
                    orig_seq_len=orig_seq_len,
                    pad_len=pad_len,
                )
                video_logits, audio_logits = gathered.split((video_dim, audio_dim), dim=-1)
                return video_logits, audio_logits
            return (
                _cp_gather_sequence(local_video, cp_ctx, orig_seq_len=orig_seq_len, pad_len=pad_len),
                _cp_gather_sequence(local_audio, cp_ctx, orig_seq_len=orig_seq_len, pad_len=pad_len),
            )
        finally:
            _ACTIVE_CONTEXT_PARALLEL.reset(token)
            self._record_cp_stats(cp_ctx, prev_a2a=prev_a2a, prev_gather=prev_gather, prev_stats=prev_stats)


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3ContextParallelConfig",
    "MiniMaxH3DiTModel",
    "reorder_grouped_qkv_to_qkv",
    "resolve_minimax_h3_context_parallel",
]
