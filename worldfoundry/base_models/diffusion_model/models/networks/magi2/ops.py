"""MAGI-2 fused-operator surface with portable PyTorch fallbacks.

Used by both the preview MoE DiT (``flash_mh_moe_*``, ``attention_with_sink``,
SwiGLU7) and, for SwiGLU7 only, the dense refiner.  FA3 + grouped-GEMM
fast paths are optional; the PyTorch bodies are the numerical reference.

Ported from SandAI's Apache-2.0 MAGI-2-preview. The upstream fast paths are:
a Triton flash-MH-MoE GEMM (``flash_mh_moe/triton/mh_moe_fwd.py``), an FA3
attention kernel with per-head sinks, and Triton MHC/RMSNorm kernels. Only the
MoE GEMM and the FA3-with-sinks attention lack an upstream pure-PyTorch path;
this module provides exact-math fallbacks for both, plus the routing/sort and
SwiGLU7 helpers (which are already pure PyTorch upstream).

The attention fast path uses FlashAttention 3's varlen kernel and its returned
softmax log-normalizer to apply the learned sink exactly, without materializing
the quadratic score matrix.  The pure-PyTorch implementation remains as a
bounded-memory numerical reference for CPU and environments without FA3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_BF16 = torch.bfloat16
_FP32 = torch.float32

_SWIGLU7_ALPHA = 1.702
_SWIGLU7_LIMIT = 7.0
_SWIGLU7_MAX_CHUNK_ELEMENTS = 32 * 1024 * 1024
_REFERENCE_ATTN_MAX_SCORE_ELEMENTS = 16 * 1024 * 1024
_GROUPED_MOE_EXPERTS_PER_BATCH = 8
_GROUPED_MOE_MAX_HIDDEN_ELEMENTS = 32 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Fused-kernel probes (Stage: fused kernels wired later; fallbacks are exact).
# --------------------------------------------------------------------------- #
def can_use_fused_mh_moe(*tensors: torch.Tensor) -> bool:
    """Whether the grouped CUDA BF16 MoE path is available/eligible."""

    if len(tensors) < 4:
        return False
    x, W_gate, W_up, W_down = tensors[:4]
    return (
        x.is_cuda
        and W_gate.is_cuda
        and W_up.is_cuda
        and W_down.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and W_gate.dtype == x.dtype
        and W_up.dtype == x.dtype
        and W_down.dtype == x.dtype
    )


def _flash_attn_varlen_func():
    """Return the optional FA3 varlen entry point without making it mandatory."""

    try:
        from flash_attn_interface import flash_attn_varlen_func
    except (ImportError, OSError):
        return None
    return flash_attn_varlen_func


def can_use_fused_attn_sink(*tensors: torch.Tensor) -> bool:
    """Whether exact FA3 varlen attention plus sink renormalization is usable."""

    if len(tensors) < 3:
        return False
    q, k, v = tensors[:3]
    return (
        _flash_attn_varlen_func() is not None
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
    )


# --------------------------------------------------------------------------- #
# SwiGLU7 activation (GPT-OSS style: clamp + swish gate * (linear + 1)).
# --------------------------------------------------------------------------- #
def swiglu7_interleaved(x: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dense-MLP SwiGLU7 with bounded fp32 activation memory.

    MAGI-2's dense projections can contain billions of BF16 elements at the
    published resolution.  Converting the complete projection to fp32 (as the
    straightforward formula does) creates several multi-GiB temporaries.  Work
    on whole-token chunks instead; this preserves the exact elementwise math
    while keeping the fp32 working set bounded.
    """

    if x.shape[-1] % 2:
        raise ValueError("SwiGLU7 expects an even interleaved feature dimension")
    out_dtype = x.dtype if out_dtype is None else out_dtype
    feature_dim = x.shape[-1]
    flat_x = x.reshape(-1, feature_dim)
    flat_out = torch.empty(
        (flat_x.shape[0], feature_dim // 2), device=x.device, dtype=out_dtype
    )
    rows_per_chunk = max(1, _SWIGLU7_MAX_CHUNK_ELEMENTS // feature_dim)
    for start in range(0, flat_x.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, flat_x.shape[0])
        chunk = flat_x[start:stop].to(_FP32)
        gate = chunk[..., ::2].clamp(max=_SWIGLU7_LIMIT)
        linear = chunk[..., 1::2].clamp(
            min=-_SWIGLU7_LIMIT, max=_SWIGLU7_LIMIT
        )
        gate.mul_(torch.sigmoid(_SWIGLU7_ALPHA * gate))
        gate.mul_(linear.add_(1.0))
        flat_out[start:stop].copy_(gate)
    return flat_out.reshape(*x.shape[:-1], feature_dim // 2)


def swiglu7_split(gate: torch.Tensor, up: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """MoE SwiGLU7: separate gate/up projections (kernel layout).

    ``swish(clamp(gate)) * (clamp(up) + 1)`` with the GPT-OSS +1 bias.
    """

    if gate.shape != up.shape:
        raise ValueError(
            f"SwiGLU7 gate/up shapes must match, got {gate.shape} and {up.shape}"
        )
    out_dtype = gate.dtype if out_dtype is None else out_dtype
    feature_dim = gate.shape[-1]
    flat_gate = gate.reshape(-1, feature_dim)
    flat_up = up.reshape(-1, feature_dim)
    flat_out = torch.empty_like(flat_gate, dtype=out_dtype)
    rows_per_chunk = max(1, _SWIGLU7_MAX_CHUNK_ELEMENTS // feature_dim)
    for start in range(0, flat_gate.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, flat_gate.shape[0])
        gate_c = flat_gate[start:stop].to(_FP32).clamp_(max=_SWIGLU7_LIMIT)
        up_c = flat_up[start:stop].to(_FP32).clamp_(
            min=-_SWIGLU7_LIMIT, max=_SWIGLU7_LIMIT
        )
        gate_c.mul_(torch.sigmoid(_SWIGLU7_ALPHA * gate_c))
        gate_c.mul_(up_c.add_(1.0))
        flat_out[start:stop].copy_(gate_c)
    return flat_out.reshape_as(gate)


# --------------------------------------------------------------------------- #
# MoE routing + CSR sort (pure PyTorch upstream, reused verbatim).
# --------------------------------------------------------------------------- #
def compute_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    score_func: str = "sigmoid",
    expert_bias: torch.Tensor | None = None,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k routing. ``router_logits [H,S,E]`` -> ``(probs [H,S,K], idx [H,S,K])``.

    Scores via sigmoid/softmax; ``expert_bias`` biases *selection only* (not the
    returned probs); probs L1-normalized when ``route_norm``.
    """

    if score_func == "sigmoid":
        router_scores = torch.sigmoid(router_logits.to(_FP32))
    elif score_func == "softmax":
        router_scores = torch.softmax(router_logits.to(_FP32), dim=-1)
    else:
        raise ValueError(f"unknown score_func {score_func!r}")
    topk_scores = router_scores if expert_bias is None else router_scores + expert_bias.unsqueeze(1)
    _, topk_indices = torch.topk(topk_scores, top_k, dim=-1)
    topk_probs = router_scores.gather(-1, topk_indices)
    if route_norm:
        topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=norm_eps)
    return topk_probs, topk_indices


def flash_mh_moe_global_sort(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map per-head expert ids to global ``[0, H*E)`` and argsort into CSR.

    Returns ``(gather_ids [T] int32, probs_sorted [T] float, expert_offsets [H*E+1] long)``.
    """

    H, S, K = topk_indices.shape
    E = num_experts
    device = topk_indices.device
    head_offset = torch.arange(H, device=device).view(H, 1, 1) * E
    global_indices = (topk_indices + head_offset).reshape(-1)
    flat_probs = topk_probs.reshape(-1)
    flat_token_ids = torch.arange(S, device=device).view(1, S, 1).expand(H, S, K).reshape(-1)
    order = global_indices.argsort(stable=True)
    gather_ids = flat_token_ids[order].to(torch.int32)
    probs_sorted = flat_probs[order].float()
    counts = torch.zeros(H * E, device=device, dtype=torch.long)
    counts.scatter_add_(0, global_indices[order], torch.ones_like(order, dtype=torch.long))
    expert_offsets = torch.zeros(H * E + 1, device=device, dtype=torch.long)
    expert_offsets[1:] = counts.cumsum(0)
    return gather_ids, probs_sorted, expert_offsets


# --------------------------------------------------------------------------- #
# MoE grouped GEMM (exact fallback for the Triton flash-MH-MoE kernel).
# --------------------------------------------------------------------------- #
def flash_mh_moe_fwd(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Gather -> per-expert (gate/up GEMM -> SwiGLU7 -> down GEMM) -> scale -> scatter-add.

    Shapes:
      ``x``            ``[S, H, d_head]`` per-head token features
      ``gather_ids``   ``[T]`` token index (into S) for each routed slot
      ``probs``        ``[T]`` routing weight per slot
      ``expert_offsets`` ``[H*E+1]`` CSR boundaries over the ``H*E`` expert slots
      ``W_gate/W_up``  ``[H*E, d_head, d_expert]``
      ``W_down``       ``[H*E, d_expert, d_head]``
    Returns ``[S, H, d_head]`` (scatter-add of every slot's contribution).

    Exact math of ``flash_mh_moe/triton/mh_moe_fwd.py`` in plain PyTorch: for
    each expert slot ``e`` (head ``h = e // E``), gather its token rows, project,
    SwiGLU7, down-project, scale by prob, and scatter-add back to head ``h``.
    """

    if can_use_fused_mh_moe(x, W_gate, W_up, W_down):
        # The released Hopper kernel avoids thousands of padded GEMM launches
        # per layer and preserves the upstream intermediate rounding.
        if (
            x.dtype == torch.bfloat16
            and x.shape[-1] == 256
            and W_down.shape[1] == 1280
            and torch.cuda.get_device_capability(x.device)[0] >= 9
        ):
            try:
                from .mh_moe_triton import mh_moe_fwd_func
            except ImportError:
                pass
            else:
                with torch.cuda.device(x.device):
                    return mh_moe_fwd_func(
                        x=x, gather_ids=gather_ids, probs=probs,
                        expert_offsets=expert_offsets, W_gate=W_gate,
                        W_up=W_up, W_down=W_down, deterministic=True,
                    )
        return _grouped_mh_moe_fwd(
            x=x,
            gather_ids=gather_ids,
            probs=probs,
            expert_offsets=expert_offsets,
            W_gate=W_gate,
            W_up=W_up,
            W_down=W_down,
        )
    return _reference_mh_moe_fwd(
        x=x,
        gather_ids=gather_ids,
        probs=probs,
        expert_offsets=expert_offsets,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
    )


def _reference_mh_moe_fwd(
    *,
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Straightforward per-expert reference used on CPU and unsupported GPUs."""

    S, H, _ = x.shape
    HE = W_gate.shape[0]
    out = torch.zeros_like(x, dtype=_FP32)
    offsets = expert_offsets.tolist()
    gather_ids_l = gather_ids.to(torch.long)
    E = HE // H
    x_flat = x  # [S, H, d_head]
    for e in range(HE):
        start, stop = offsets[e], offsets[e + 1]
        if stop <= start:
            continue
        head = e // E
        rows = gather_ids_l[start:stop]  # token ids into S
        xe = x_flat[rows, head, :].to(W_gate.dtype)  # [n, d_head]
        gate = xe @ W_gate[e]  # [n, d_expert]
        up = xe @ W_up[e]  # [n, d_expert]
        h = swiglu7_split(gate, up, out_dtype=W_down.dtype)  # [n, d_expert]
        ye = (h @ W_down[e]).to(_FP32)  # [n, d_head]
        ye = ye * probs[start:stop].to(_FP32).unsqueeze(-1)
        # Advanced indexing returns a temporary tensor; update the head view.
        out[:, head, :].index_add_(0, rows, ye)
    return out.to(x.dtype)


def _grouped_mh_moe_fwd(
    *,
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """Run routed experts as padded grouped GEMMs instead of per-expert GEMMs.

    The global sort makes rows for each expert contiguous. Experts are batched
    in small groups, padded only to the largest active row count in that group,
    and evaluated with tensor-core bmm. This keeps peak intermediates bounded
    while reducing thousands of individual GEMM launches per layer.
    """

    _, H, d_head = x.shape
    HE = W_gate.shape[0]
    E = HE // H
    offsets = expert_offsets.tolist()
    gather_ids_l = gather_ids.to(torch.long)
    active = [
        (expert, offsets[expert], offsets[expert + 1])
        for expert in range(HE)
        if offsets[expert + 1] > offsets[expert]
    ]
    # Similar-size experts minimize padding.  A few highly selected experts can
    # otherwise make an eight-expert group allocate several GiB even though the
    # remaining seven experts only contain a small number of routed rows.
    active.sort(key=lambda item: item[2] - item[1], reverse=True)
    out = torch.zeros_like(x, dtype=_FP32)

    groups: list[list[tuple[int, int, int]]] = []
    group: list[tuple[int, int, int]] = []
    group_max_rows = 0
    for item in active:
        rows = item[2] - item[1]
        next_max_rows = max(group_max_rows, rows)
        next_size = len(group) + 1
        would_exceed_memory = (
            next_size * next_max_rows * W_gate.shape[-1]
            > _GROUPED_MOE_MAX_HIDDEN_ELEMENTS
        )
        if group and (
            next_size > _GROUPED_MOE_EXPERTS_PER_BATCH or would_exceed_memory
        ):
            groups.append(group)
            group = []
            group_max_rows = 0
        group.append(item)
        group_max_rows = max(group_max_rows, rows)
    if group:
        groups.append(group)

    for group in groups:
        max_rows = max(stop - start for _, start, stop in group)
        expert_ids = torch.tensor(
            [expert for expert, _, _ in group], device=x.device, dtype=torch.long
        )
        x_batch = torch.zeros(
            len(group), max_rows, d_head, device=x.device, dtype=x.dtype
        )
        row_sets: list[torch.Tensor] = []
        for local_idx, (expert, start, stop) in enumerate(group):
            rows = gather_ids_l[start:stop]
            row_sets.append(rows)
            head = expert // E
            x_batch[local_idx, : stop - start].copy_(x[rows, head, :])

        gate = torch.bmm(x_batch, W_gate.index_select(0, expert_ids))
        up = torch.bmm(x_batch, W_up.index_select(0, expert_ids))
        hidden = swiglu7_split(gate, up, out_dtype=W_down.dtype)
        projected = torch.bmm(hidden, W_down.index_select(0, expert_ids))

        for local_idx, ((expert, start, stop), rows) in enumerate(zip(group, row_sets)):
            count = stop - start
            contribution = projected[local_idx, :count].to(_FP32)
            contribution.mul_(probs[start:stop].to(_FP32).unsqueeze(-1))
            out[:, expert // E, :].index_add_(0, rows, contribution)

    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# Attention with per-head sinks (exact fallback for FA3-with-sink).
# --------------------------------------------------------------------------- #
def attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Non-causal varlen attention with a learned per-head sink logit.

    ``q/k/v``: ``[T, num_heads, head_dim]`` packed rows. ``sinks``: ``[num_heads]``
    (or ``[sink_token_num, num_heads]``, reduced over the sink axis) — a learned
    logit appended to the softmax denominator as an always-attended null key
    with zero value contribution. ``cu_seqlens``: ``[num_docs+1]``.

    Matches ``FA3VarlenFuncWithSink`` semantics: ``out = (sum_j p_j v_j)`` where
    the softmax normalizer includes ``exp(sink - max)`` but the sink carries no
    value, so it only *attenuates* the output.
    """

    T, num_heads, head_dim = q.shape
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    sink = sinks.to(_FP32)
    if sink.ndim == 2:  # [sink_token_num, num_heads] -> reduce sink axis in logsumexp
        sink_logits = torch.logsumexp(sink, dim=0)  # [num_heads]
    else:
        sink_logits = sink  # [num_heads]
    if can_use_fused_attn_sink(q, k, v):
        flash_attn_varlen_func = _flash_attn_varlen_func()
        assert flash_attn_varlen_func is not None
        cu_seqlens_i32 = cu_seqlens.to(device=q.device, dtype=torch.int32)
        lengths = cu_seqlens_i32[1:] - cu_seqlens_i32[:-1]
        if lengths.numel() == 0 or bool((lengths <= 0).any()):
            raise ValueError("cu_seqlens must describe at least one non-empty sequence")
        max_seqlen = int(lengths.max().item())
        attended, softmax_lse = flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_i32,
            cu_seqlens_i32,
            None,
            None,
            max_seqlen,
            max_seqlen,
            softmax_scale=scale,
            causal=False,
        )
        # FA3 returns log(sum_j exp(qk_j * scale)) as [num_heads, T].
        # A learned null-value sink contributes exp(sink) to that denominator,
        # so the ordinary attention output only needs this exact attenuation.
        sink_weight = torch.sigmoid(
            softmax_lse.transpose(0, 1).to(_FP32) - sink_logits.view(1, num_heads)
        )
        return (attended.to(_FP32) * sink_weight.unsqueeze(-1)).to(q.dtype)

    if (
        q.is_cuda
        and torch.version.hip is None
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
        and q.shape[-1] % 8 == 0
        and torch.cuda.get_device_capability(q.device)[0] >= 8
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")
    ):
        # PyTorch ships FlashAttention even when the optional FA3 package is
        # absent. Its ATen interface exposes the same log-normalizer needed
        # for exact sink correction; public SDPA returns only the output.
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop <= start:
                continue
            attended, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                q[start:stop].transpose(0, 1).unsqueeze(0),
                k[start:stop].transpose(0, 1).unsqueeze(0),
                v[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=scale,
            )
            weight = torch.sigmoid(lse[0].float() - sink_logits[:, None])
            out[start:stop] = (attended[0].float() * weight[..., None]).transpose(0, 1)
        return out

    out = torch.empty_like(q, dtype=_FP32)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        length = int(stop) - int(start)
        if length <= 0:
            continue
        ks = k[start:stop].to(_FP32).transpose(0, 1)
        vs = v[start:stop].to(_FP32).transpose(0, 1)
        query_chunk = max(
            1,
            min(length, _REFERENCE_ATTN_MAX_SCORE_ELEMENTS // max(1, num_heads * length)),
        )
        for query_start in range(0, length, query_chunk):
            query_stop = min(length, query_start + query_chunk)
            qs = q[start + query_start : start + query_stop].to(_FP32).transpose(0, 1)
            scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
            sink_col = sink_logits.view(num_heads, 1, 1).expand(
                num_heads, query_stop - query_start, 1
            )
            full = torch.cat([scores, sink_col], dim=-1)
            probs = torch.softmax(full, dim=-1)[..., :length]
            attended = torch.matmul(probs, vs)
            out[start + query_start : start + query_stop] = attended.transpose(0, 1)
    return out.to(q.dtype)


def make_rms_norm(size: int, *, eps: float = 1e-6, dtype: torch.dtype = _BF16) -> torch.nn.RMSNorm:
    """RMSNorm with fp32-accumulation semantics (reference contract)."""

    return torch.nn.RMSNorm(size, eps=eps, dtype=dtype)


__all__ = [
    "attention_with_sink",
    "can_use_fused_attn_sink",
    "can_use_fused_mh_moe",
    "compute_topk_probs_and_indices",
    "flash_mh_moe_fwd",
    "flash_mh_moe_global_sort",
    "make_rms_norm",
    "swiglu7_interleaved",
    "swiglu7_split",
]
