"""Ulysses sequence-parallel (SP) self-attention for the base Wan2.2 DiT.

Video self-attention over the full 3D latent (T·H·W tokens) is the long-context
bottleneck. Ulysses SP shards the token sequence across ``sp_degree`` GPUs: each
rank holds ``S/P`` tokens, an all-to-all scatters heads and gathers the full
sequence so every rank runs exact attention over the whole sequence with a
subset of heads, then an inverse all-to-all restores the sequence shard. This is
**numerically equivalent** to single-GPU attention (not lossy) — it trades NCCL
communication for per-GPU compute/memory.

This module installs a replacement :class:`SelfAttentionProcessor` on every Wan
``SelfAttention`` (the same seam the approximate-attention lane uses). It reuses
the model's own complex RoPE — applied on the rank-local sequence slice — and
xfuser's ``xFuserLongContextAttention`` for the all-to-all + attention, so it
composes with FA3/FP8 (which run inside the wrapped attention after heads are
scattered).

**OFF by default, opt-in only.** A single-GPU / non-distributed run never
installs it. Constraint: ``num_heads % sp_degree == 0`` (Wan2.2 has 24 heads, so
sp ∈ {2,3,4,6,8,12} all divide). The pipeline is responsible for sharding the
input sequence and all-gathering the output; this module only rewrites the
per-block attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class _SPState:
    """Shared runtime state for one SP installation (telemetry + config)."""

    sp_degree: int
    backend: str = "native-ulysses"
    wrapped_blocks: int = 0
    num_heads: int | None = None
    head_parallel: bool = True
    fused_rope_calls: int = 0
    complex_rope_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def reset_request_window(self) -> None:
        """Clear execution receipts that must not leak across generation requests."""

        self.fused_rope_calls = 0
        self.complex_rope_calls = 0


def _rank_slice_freqs(freqs: torch.Tensor, sp_rank: int, s_local: int) -> torch.Tensor:
    """Slice the full RoPE table to this rank's contiguous sequence shard.

    ``freqs`` is the full ``[S, 1, D/2]`` complex table; the pipeline shards the
    sequence into ``P`` contiguous chunks, so rank ``r`` uses tokens
    ``[r*s_local : (r+1)*s_local]``. Padded if the shard runs past the table.
    """
    start = sp_rank * s_local
    end = start + s_local
    if end <= freqs.shape[0]:
        return freqs[start:end]
    pad = end - freqs.shape[0]
    tail = torch.ones(pad, *freqs.shape[1:], dtype=freqs.dtype, device=freqs.device)
    return torch.cat([freqs[start:], tail], dim=0)


class SequenceParallelSelfAttentionProcessor:
    """SP replacement for the Wan self-attention processor (Ulysses all-to-all).

    ``x`` arriving here is already the rank-local sequence shard ``[B, S/P, dim]``
    (the pipeline shards before the blocks). RoPE is applied on that shard with
    the rank-sliced freqs; xfuser's USP attention performs the all-to-all,
    runs exact attention over the full sequence, and scatters back.
    """

    def __init__(self, state: _SPState) -> None:
        self._state = state

    def __call__(self, attention: nn.Module, x: torch.Tensor, freqs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        import torch.distributed as dist
        from einops import rearrange

        from worldfoundry.core.attention import apply_complex_rotary_embedding as rope_apply
        from worldfoundry.core.attention import packed_sequence_attention
        from worldfoundry.core.distributed.sequence_parallel.runtime import (
            all_to_all_4D,
            all_to_all_4d_many,
            get_sequence_parallel_group,
        )

        num_heads = attention.num_heads
        b, s_local, _ = x.shape

        if hasattr(attention, "qkv"):
            from .qkv_fusion import project_fused_qkv

            q, k, v = project_fused_qkv(attention.qkv, x)
        else:
            q = attention.q(x)
            k = attention.k(x)
            v = attention.v(x)
        # Rank within the SP group, derived natively (the xfuser-provided
        # get_sequence_parallel_rank is None when xfuser is absent).
        group = get_sequence_parallel_group()
        sp_rank = dist.get_rank(group) if dist.is_initialized() else 0
        fused_table = kwargs.pop("_worldfoundry_rope_table", None)
        fused_grid = kwargs.pop("_worldfoundry_rope_grid", None)
        rope_precision = str(
            kwargs.pop("_worldfoundry_rope_precision", "fp64")
        ).strip().casefold()
        if rope_precision not in {"fp32", "fp64"}:
            raise ValueError("Wan RoPE precision must be 'fp32' or 'fp64'")
        if (fused_table is None) != (fused_grid is None):
            raise ValueError("Wan fused RoPE table and grid must be provided together")
        if fused_table is not None:
            # The shared fused primitive was deliberately designed for both
            # sequence and head sharding.  ``sequence_offset`` restores each
            # local shard's global 3D token positions; ``valid_tokens`` makes
            # right-padding an identity rotation. This composes SP with fused
            # Q/K RMSNorm+RoPE without gathering the hidden sequence first.
            from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_3d

            grid = tuple(int(value) for value in fused_grid)

            def fused_call() -> tuple[torch.Tensor, torch.Tensor]:
                return hidden_qk_rmsnorm_rope_3d(
                    q,
                    k,
                    attention.norm_q.weight,
                    attention.norm_k.weight,
                    fused_table,
                    num_heads=num_heads,
                    grid_size=grid,
                    eps=attention.norm_q.eps,
                    sequence_offset=sp_rank * s_local,
                    valid_tokens=grid[0] * grid[1] * grid[2],
                )

            runtime_state = getattr(
                attention,
                "_worldfoundry_fused_rope_runtime",
                None,
            )
            if runtime_state is None:
                q, k = fused_call()
            elif torch.compiler.is_compiling():
                from .fused_rope import record_compiled_fused_rope_graph_trace

                record_compiled_fused_rope_graph_trace(runtime_state)
                q, k = fused_call()
            else:
                from worldfoundry.core.kernels.registry import (
                    kernel_dispatch_receipt_scope,
                )

                receipt: dict[str, Any] = {}
                with kernel_dispatch_receipt_scope(receipt):
                    q, k = fused_call()
                runtime_state.record_eager_dispatch(receipt)
            self._state.fused_rope_calls += 1
        else:
            q = attention.norm_q(q)
            k = attention.norm_k(k)
            freqs_rank = _rank_slice_freqs(freqs, sp_rank, s_local)
            rope_dtype = (
                torch.float32 if rope_precision == "fp32" else torch.float64
            )
            q = rope_apply(q, freqs_rank, num_heads, compute_dtype=rope_dtype)
            k = rope_apply(k, freqs_rank, num_heads, compute_dtype=rope_dtype)
            self._state.complex_rope_calls += 1

        # [B, S/P, dim] -> [B, S/P, H, D].
        q = rearrange(q, "b s (h d) -> b s h d", h=num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=num_heads)

        # Ulysses via the in-tree native NCCL all-to-all (no xfuser dependency).
        # scatter heads / gather sequence: [B, S/P, H, D] -> [B, S, H/P, D].
        q, k, v = all_to_all_4d_many(
            (q, k, v),
            group,
            scatter_dim=2,
            gather_dim=1,
            assume_even=True,
        )

        # Exact attention over the full sequence on this rank's head subset.
        # Keep the model-selected backend (FA2/FA3/SDPA) instead of bypassing
        # the Wan dispatcher with a raw PyTorch SDPA call. Apart from restoring
        # the requested fast kernel, using the same provider on SP1 and SP>1
        # avoids diffusion-step drift from mixing different softmax kernels.
        local_heads = q.shape[2]
        head_dim = q.shape[3]
        attention_module = attention.attn
        out = packed_sequence_attention(
            q=q.flatten(2),
            k=k.flatten(2),
            v=v.flatten(2),
            num_heads=local_heads,
            compatibility_mode=bool(
                getattr(attention_module, "compatibility_mode", False)
            ),
            backend=getattr(attention_module, "attention_backend", None),
        ).reshape(b, -1, local_heads, head_dim)

        # scatter sequence / gather heads: [B, S, H/P, D] -> [B, S/P, H, D].
        out = all_to_all_4D(
            out,
            group,
            scatter_dim=1,
            gather_dim=2,
            assume_even=True,
        )
        out = out.flatten(2)  # [B, S/P, dim]
        return attention.o(out)


def enable_sequence_parallel(model: nn.Module, sp_degree: int) -> _SPState:
    """Install the SP self-attention processor on every Wan ``SelfAttention``.

    Requires an already-initialized sequence-parallel process group of size
    ``sp_degree`` (see :func:`worldfoundry.core.distributed.sequence_parallel.runtime.set_multi_gpus_devices`).
    Idempotent: re-enabling replaces a prior SP processor. Returns a state handle
    for the audit record.
    """
    if sp_degree < 2:
        raise ValueError(f"sequence parallel requires sp_degree >= 2, got {sp_degree}")

    state = _SPState(sp_degree=sp_degree)
    wrapped = 0
    heads_seen: set[int] = set()
    for module in model.modules():
        if (
            type(module).__name__ == "SelfAttention"
            and hasattr(module, "set_processor")
            and hasattr(module, "get_processor")
        ):
            heads_seen.add(module.num_heads)
            if module.num_heads % sp_degree != 0:
                raise ValueError(
                    f"Wan num_heads={module.num_heads} is not divisible by "
                    f"sequence_parallel={sp_degree}"
                )
            # QKV fusion installs an instance-level fast forward that bypasses
            # processors. Restore the class forward so this Ulysses processor
            # owns dispatch while retaining the merged ``qkv`` projection.
            if getattr(module, "_qkv_fused", False) and "forward" in module.__dict__:
                delattr(module, "forward")
            module.set_processor(SequenceParallelSelfAttentionProcessor(state))
            wrapped += 1
    state.wrapped_blocks = wrapped
    state.num_heads = next(iter(heads_seen)) if len(heads_seen) == 1 else None
    return state


def require_sequence_parallel_runtime(sp_degree: int) -> None:
    """Require a live process group matching ``sp_degree`` before model load."""

    import torch.distributed as dist

    from worldfoundry.core.distributed.sequence_parallel.runtime import (
        get_sequence_parallel_group,
    )

    if not dist.is_initialized():
        raise RuntimeError(
            "sequence_parallel requires torchrun/NCCL initialization before model loading"
        )
    group = get_sequence_parallel_group()
    if group is None:
        raise RuntimeError("sequence_parallel process group is not initialized")
    actual = int(dist.get_world_size(group))
    if actual != int(sp_degree):
        raise ValueError(
            f"sequence_parallel={sp_degree} but the active group has size {actual}"
        )


def sequence_parallel_report(state: _SPState) -> dict[str, Any]:
    return {
        "sp_degree": state.sp_degree,
        "backend": state.backend,
        "wrapped_blocks": state.wrapped_blocks,
        "head_parallel": state.head_parallel,
        "fused_rope_calls": state.fused_rope_calls,
        "complex_rope_calls": state.complex_rope_calls,
    }


__all__ = [
    "SequenceParallelSelfAttentionProcessor",
    "enable_sequence_parallel",
    "require_sequence_parallel_runtime",
    "sequence_parallel_report",
]
