# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine acceleration stack for LTX-2.5 22B (dev, distilled, and refinement).

Composes orthogonal, auditable, zero-fallback inference optimizations cleanly
around the canonical :class:`LTXModel` and LTX-2.5 video/audio codecs without
mutating shared LTX-2.x base-model modules:

1. **FirstBlockCache (FBCache)** — runs block 0 as a lightweight probe (1/48
   of transformer compute) and reuses the cached block-1..N residual whenever
   block-0's relative L1 drift remains below a calibrated threshold, guarded by
   dense first/last boundary steps, cumulative drift tracking, and a hard
   consecutive-skip cap.
2. **Kernel Fusion + Hopper FA3 + 3D-Tiled Sol-Attn** — fuses QKV / cross-KV
   projections, enables LTX Triton fused ops (QK-RMSNorm+RoPE, post-SA
   residual+RMSNorm+AdaLN, gated-attention+FP8, dual AV AdaLN, scale-shift),
   binds lossless Hopper FlashAttention-3, and optionally layers 3D-tiled
   ``sol_attn`` on video self-attention with dense edge blocks & warmup steps.
3. **Hopper Rowwise FP8 W8A8 + Fused GELU-FP8 FFN** — quantizes heavy DiT
   block projections with calibrated exclusions (stems, AdaLN, gate logits,
   output heads, and optional edge blocks) and ``use_fast_accum=False`` by
   default for multi-step dev parity.
4. **Split Piecewise CUDA Graphs** — captures Block 0 and Tail Blocks 1..N as
   independent CUDA Graph segments via :class:`PiecewiseBlockGraphRunner` so
   FBCache gating stays outside captured replay with zero graph conflicts.
5. **VideoVAE Acceleration** — supports both Conv VAE (``channels_last_3d`` +
   ``torch.compile`` + untiled fast path + CUDA-graphed audio/vocoder) and
   DiffVAE (``combined_compile`` / ``chunked_compile``).
6. **Multi-GPU Parallelism** — CFG=2 branch parallelism for multi-step dev
   mode and Ulysses Sequence Parallelism (SP=2/4/8) across LTX-2.5 video
   self-attention with exact video-to-audio cross-attention gathering.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import Attention
from worldfoundry.base_models.diffusion_model.models.networks.ltx.fused_ops import (
    LTX_AUDIO_FP8_EXCLUDES,
    LTX_DEFAULT_FP8_EXCLUDES,
    enable_ltx_fp8_quantization,
    enable_ltx_fused_ops,
    fuse_ltx_projections,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import (
    LTXModel,
    _build_tile_3d_permutation,
    _permute_video_args,
    resolve_cached_video_grid_shape,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs
from worldfoundry.core.acceleration.cache import BlockCacheEvent
from worldfoundry.core.attention.backends.dispatch import (
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.core.attention.backends.model_backends import PytorchAttention
from worldfoundry.core.attention.backends.probe import (
    attention_backend_capability,
    normalize_attention_backend,
    resolve_attention_backend,
)
from worldfoundry.core.distributed.sequence_parallel.ops import all_to_all, all_to_all_many
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner

# ──────────────────────────────────────────────────────────────────────────
# FirstBlockCache (FBCache) — block-0 probe + tail residual reuse
# ──────────────────────────────────────────────────────────────────────────


def _validate_int_flag(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _validate_float_flag(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive:
        if parsed <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {parsed}")
    elif parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class LTX25FirstBlockCacheConfig:
    """Calibrated FirstBlockCache policy for 48-layer LTX-2.5 transformers.

    Runs block 0 on every step and reuses the residual of blocks ``1..47``
    whenever the block-0 video residual relative L1 drift stays below
    ``residual_diff_threshold``. Dense first/last boundaries, cumulative drift
    against the last dense probe, and ``max_consecutive_skips`` prevent quality
    drift across multi-step denoising.
    """

    enabled: bool = True
    residual_diff_threshold: float = 0.08
    audio_diff_threshold: float | None = None
    dense_first: int = 2
    dense_last: int = 2
    max_consecutive_skips: int = 2
    downsample_factor: int = 4
    cumulative: bool = True
    min_steps_to_enable: int = 4
    max_tokens: int | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"fbcache enabled must be bool, got {self.enabled!r}")
        _validate_float_flag(self.residual_diff_threshold, name="residual_diff_threshold", minimum=0.0)
        if self.audio_diff_threshold is not None:
            _validate_float_flag(self.audio_diff_threshold, name="audio_diff_threshold", minimum=0.0)
        _validate_int_flag(self.dense_first, name="dense_first", minimum=1)
        _validate_int_flag(self.dense_last, name="dense_last", minimum=0)
        _validate_int_flag(self.max_consecutive_skips, name="max_consecutive_skips", minimum=0)
        _validate_int_flag(self.downsample_factor, name="downsample_factor", minimum=1)
        if not isinstance(self.cumulative, bool):
            raise TypeError(f"fbcache cumulative must be bool, got {self.cumulative!r}")
        _validate_int_flag(self.min_steps_to_enable, name="min_steps_to_enable", minimum=2)
        if self.max_tokens is not None:
            _validate_int_flag(self.max_tokens, name="max_tokens", minimum=1)
        _validate_float_flag(self.eps, name="eps", minimum=0.0, strictly_positive=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_sig(tensor: torch.Tensor | None) -> tuple[tuple[int, ...], str, int | None, torch.dtype] | None:
    if tensor is None:
        return None
    return (tuple(int(d) for d in tensor.shape), tensor.device.type, tensor.device.index, tensor.dtype)


def _relative_l1_probe(
    current: torch.Tensor,
    reference: torch.Tensor,
    eps: float,
    *,
    sp_size: int = 1,
    sp_rank: int = 0,
    sp_orig_tokens: int | None = None,
    sp_group: Any = None,
    all_reduce_fn: Callable[[torch.Tensor, Any], None] | None = None,
) -> float:
    """Compute exact global relative-L1 probe drift (synchronized across SP ranks when ``sp_size > 1``)."""
    cur = current.detach().float()
    ref = reference.to(device=cur.device).float()
    if sp_size <= 1:
        num = (cur - ref).abs().mean()
        den = ref.abs().mean().clamp_min(eps)
        return float((num / den).item())

    # Exclude trailing SP padding tokens so global relative L1 matches 1-GPU bit-for-bit
    # and every SP rank makes the exact same hit/miss decision (zero collective deadlock).
    local_len = int(cur.shape[1]) if cur.ndim >= 2 else 0
    if sp_orig_tokens is not None and cur.ndim >= 2:
        start = int(sp_rank) * local_len
        valid_local = max(0, min(local_len, int(sp_orig_tokens) - start))
        cur = cur[:, :valid_local]
        ref = ref[:, :valid_local]

    if cur.numel() > 0:
        num_sum = (cur - ref).abs().sum()
        den_sum = ref.abs().sum()
        elem_count = float(cur.numel())
    else:
        num_sum = torch.zeros((), dtype=torch.float32, device=cur.device)
        den_sum = torch.zeros((), dtype=torch.float32, device=cur.device)
        elem_count = 0.0

    stats = torch.stack((num_sum, den_sum, torch.tensor(elem_count, dtype=torch.float32, device=cur.device)))
    if all_reduce_fn is not None:
        all_reduce_fn(stats, sp_group)
    else:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=sp_group)

    total_num, total_den, total_elems = (float(v) for v in stats.tolist())
    denom_elems = max(total_elems, 1.0)
    mean_num = total_num / denom_elems
    mean_den = max(total_den / denom_elems, float(eps))
    return mean_num / mean_den


class LTX25FirstBlockCache:
    """Request/stage/branch-scoped FirstBlockCache controller for LTX-2.5 AV DiT."""

    algorithm = "firstblock"

    def __init__(
        self,
        config: LTX25FirstBlockCacheConfig,
        *,
        branch: str = "positive",
        stage_index: int = 0,
    ) -> None:
        self.config = config
        self.branch = str(branch)
        self.stage_index = int(stage_index)
        self.events: list[BlockCacheEvent] = []
        self.dense_block_calls: int = 0
        self.skipped_block_calls: int = 0
        self.hits: int = 0
        self._clear_state()

    def _clear_state(self) -> None:
        self._ref_video_probe: torch.Tensor | None = None
        self._ref_audio_probe: torch.Tensor | None = None
        self._cached_video_residual: torch.Tensor | None = None
        self._cached_audio_residual: torch.Tensor | None = None
        self._video_sig: tuple[tuple[int, ...], str, int | None, torch.dtype] | None = None
        self._audio_sig: tuple[tuple[int, ...], str, int | None, torch.dtype] | None = None
        self._last_sol_force_dense: bool | None = None
        self._consecutive_skips: int = 0

    def reset(self) -> None:
        """Drop all cached residuals and reset telemetry for a new generation."""
        self._clear_state()
        self.events.clear()
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    def _sample_probe(self, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        stride = self.config.downsample_factor
        return detached[..., ::stride] if stride > 1 else detached

    def _is_dense_boundary(self, step: int, total_steps: int | None) -> bool:
        if step < self.config.dense_first:
            return True
        if self.config.dense_last > 0 and total_steps is not None:
            return step >= max(int(total_steps) - self.config.dense_last, 0)
        return False

    def evaluate_probe(
        self,
        *,
        step: int,
        total_steps: int | None,
        num_blocks: int,
        block0_video_in: torch.Tensor | None,
        block0_video_out: torch.Tensor | None,
        block0_audio_in: torch.Tensor | None,
        block0_audio_out: torch.Tensor | None,
        sol_force_dense: bool = False,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_orig_tokens: int | None = None,
        sp_group: Any = None,
        all_reduce_fn: Callable[[torch.Tensor, Any], None] | None = None,
    ) -> tuple[bool, str, float | None, torch.Tensor | None, torch.Tensor | None]:
        """Decide whether blocks ``1..num_blocks-1`` can reuse their cached residual."""
        if not self.config.enabled or num_blocks < 2:
            return False, "disabled", None, None, None
        if torch.is_grad_enabled():
            return False, "autograd", None, None, None
        if step == 0:
            self._clear_state()

        v_probe = (
            self._sample_probe(block0_video_out - block0_video_in)
            if block0_video_in is not None and block0_video_out is not None
            else None
        )
        a_probe = (
            self._sample_probe(block0_audio_out - block0_audio_in)
            if block0_audio_in is not None and block0_audio_out is not None
            else None
        )

        if total_steps is not None and int(total_steps) < self.config.min_steps_to_enable:
            return False, "short-schedule", None, v_probe, a_probe
        effective_video_tokens = (
            int(sp_orig_tokens)
            if sp_orig_tokens is not None
            else (int(block0_video_out.shape[1]) if block0_video_out is not None else 0)
        )
        if (
            self.config.max_tokens is not None
            and block0_video_out is not None
            and effective_video_tokens > self.config.max_tokens
        ):
            return False, "max-tokens", None, v_probe, a_probe
        if step == 0:
            return False, "seed", None, v_probe, a_probe
        if self._is_dense_boundary(step, total_steps):
            return False, "dense-boundary", None, v_probe, a_probe
        if self._last_sol_force_dense is not None and self._last_sol_force_dense != bool(sol_force_dense):
            return False, "attention-mode-transition", None, v_probe, a_probe
        if self._consecutive_skips >= self.config.max_consecutive_skips:
            return False, "max-consecutive-skips", None, v_probe, a_probe

        cur_v_sig = _tensor_sig(block0_video_out)
        cur_a_sig = _tensor_sig(block0_audio_out)
        if v_probe is not None and (
            self._ref_video_probe is None or self._cached_video_residual is None or self._video_sig != cur_v_sig
        ):
            return False, "cache-missing", None, v_probe, a_probe
        if a_probe is not None and (
            self._ref_audio_probe is None or self._cached_audio_residual is None or self._audio_sig != cur_a_sig
        ):
            return False, "cache-missing", None, v_probe, a_probe

        metric: float | None = None
        if v_probe is not None:
            assert self._ref_video_probe is not None
            metric = _relative_l1_probe(
                v_probe,
                self._ref_video_probe,
                self.config.eps,
                sp_size=sp_size,
                sp_rank=sp_rank,
                sp_orig_tokens=sp_orig_tokens,
                sp_group=sp_group,
                all_reduce_fn=all_reduce_fn,
            )
            if not math.isfinite(metric):
                return False, "non-finite-metric", metric, v_probe, a_probe
            if metric >= self.config.residual_diff_threshold:
                return False, "threshold", metric, v_probe, a_probe

        if a_probe is not None and self._ref_audio_probe is not None and self.config.audio_diff_threshold is not None:
            audio_metric = _relative_l1_probe(a_probe, self._ref_audio_probe, self.config.eps)
            if not math.isfinite(audio_metric):
                return False, "non-finite-audio-metric", audio_metric, v_probe, a_probe
            if audio_metric >= self.config.audio_diff_threshold:
                return False, "audio-threshold", audio_metric, v_probe, a_probe

        if v_probe is None and a_probe is not None:
            assert self._ref_audio_probe is not None
            metric = _relative_l1_probe(a_probe, self._ref_audio_probe, self.config.eps)
            if not math.isfinite(metric):
                return False, "non-finite-metric", metric, v_probe, a_probe
            if metric >= self.config.residual_diff_threshold:
                return False, "threshold", metric, v_probe, a_probe

        return True, "below-threshold", metric, v_probe, a_probe

    def apply_hit(
        self,
        *,
        step: int,
        num_blocks: int,
        block0_video_out: torch.Tensor | None,
        block0_audio_out: torch.Tensor | None,
        v_probe: torch.Tensor | None,
        a_probe: torch.Tensor | None,
        metric: float | None,
        reason: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply cached tail residuals on a hit and record telemetry."""
        v_out = None
        if block0_video_out is not None:
            assert self._cached_video_residual is not None
            v_out = block0_video_out + self._cached_video_residual
        a_out = None
        if block0_audio_out is not None:
            assert self._cached_audio_residual is not None
            a_out = block0_audio_out + self._cached_audio_residual
        if not self.config.cumulative:
            if v_probe is not None:
                self._ref_video_probe = v_probe.detach().clone()
            if a_probe is not None:
                self._ref_audio_probe = a_probe.detach().clone()
        self._consecutive_skips += 1
        self._record(
            step=step,
            reason=reason,
            dense_blocks=(0,),
            skipped_blocks=tuple(range(1, num_blocks)),
            metric=metric,
        )
        return v_out, a_out

    def record_dense(
        self,
        *,
        step: int,
        num_blocks: int,
        block0_video_out: torch.Tensor | None,
        final_video_out: torch.Tensor | None,
        block0_audio_out: torch.Tensor | None,
        final_audio_out: torch.Tensor | None,
        v_probe: torch.Tensor | None,
        a_probe: torch.Tensor | None,
        metric: float | None,
        reason: str,
        sol_force_dense: bool = False,
    ) -> None:
        """Store detached block-1..N residuals after a dense pass and record telemetry."""
        if not torch.is_grad_enabled():
            if block0_video_out is not None and final_video_out is not None:
                self._cached_video_residual = (final_video_out - block0_video_out).detach()
                self._video_sig = _tensor_sig(block0_video_out)
            if block0_audio_out is not None and final_audio_out is not None:
                self._cached_audio_residual = (final_audio_out - block0_audio_out).detach()
                self._audio_sig = _tensor_sig(block0_audio_out)
            if v_probe is not None:
                self._ref_video_probe = v_probe.detach().clone()
            if a_probe is not None:
                self._ref_audio_probe = a_probe.detach().clone()
            self._last_sol_force_dense = bool(sol_force_dense)
            self._consecutive_skips = 0
        self._record(
            step=step,
            reason=reason,
            dense_blocks=tuple(range(num_blocks)),
            skipped_blocks=(),
            metric=metric,
        )

    def _record(
        self,
        *,
        step: int,
        reason: str,
        dense_blocks: tuple[int, ...],
        skipped_blocks: tuple[int, ...],
        metric: float | None,
    ) -> None:
        event = BlockCacheEvent(
            algorithm=self.algorithm,
            step=int(step),
            hit=bool(skipped_blocks),
            reason=reason,
            dense_blocks=dense_blocks,
            skipped_blocks=skipped_blocks,
            accumulated_change=metric,
        )
        self.events.append(event)
        self.dense_block_calls += event.dense_block_calls
        self.skipped_block_calls += event.skipped_block_calls
        self.hits += int(event.hit)

    def receipt(self) -> dict[str, Any]:
        """Return JSON-serializable telemetry matching WorldFoundry cache contracts."""
        total_steps = len(self.events)
        total_blocks = self.dense_block_calls + self.skipped_block_calls
        return {
            "algorithm": self.algorithm,
            "branch": self.branch,
            "stage_index": self.stage_index,
            "config": self.config.to_dict(),
            "steps": total_steps,
            "hits": self.hits,
            "dense_steps": total_steps - self.hits,
            "skip_rate": round(self.hits / total_steps, 4) if total_steps > 0 else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "block_skip_rate": round(self.skipped_block_calls / total_blocks, 4) if total_blocks > 0 else 0.0,
            "event_receipts": [event.receipt() for event in self.events],
        }


# ──────────────────────────────────────────────────────────────────────────
# Multi-GPU Parallelism — CFG=2 Parallel + Ulysses Sequence Parallel (SP=2/4)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LTX25ParallelConfig:
    """Multi-GPU parallel execution configuration for LTX-2.5 22B.

    Supports orthogonal 2D composition on 2x/4x/8x H200:
    - ``cfg_parallel=2`` evaluates positive & negative CFG branches concurrently
      across 2 ranks during multi-step dev generation (``guidance_scale > 1``).
    - ``sequence_parallel=2`` or ``4`` shards video tokens across SP ranks and
      uses Ulysses all-to-all head/sequence exchange on every LTX-2.5 video
      self-attention block while gathering video K/V for audio<-video cross-attn.
    """

    cfg_parallel: int = 1
    sequence_parallel: int = 1
    pad_sequence_to_sp: bool = True
    gather_v2a_video_kv: bool = True

    def __post_init__(self) -> None:
        cfg = _validate_int_flag(self.cfg_parallel, name="cfg_parallel", minimum=1)
        if cfg not in (1, 2):
            raise ValueError(f"LTX-2.5 cfg_parallel must be 1 or 2, got {cfg}")
        sp = _validate_int_flag(self.sequence_parallel, name="sequence_parallel", minimum=1)
        if sp > 1 and 32 % sp != 0:
            raise ValueError(f"LTX-2.5 sequence_parallel must divide 32 attention heads, got {sp}")
        if not isinstance(self.pad_sequence_to_sp, bool):
            raise TypeError("pad_sequence_to_sp must be bool")
        if not isinstance(self.gather_v2a_video_kv, bool):
            raise TypeError("gather_v2a_video_kv must be bool")

    @property
    def required_world_size(self) -> int:
        return int(self.cfg_parallel) * int(self.sequence_parallel)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pad_dim(tensor: torch.Tensor, dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    zeros = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, zeros), dim=dim)


def _shard_rope_tuple(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    sp_rank: int,
    sp_size: int,
    pad_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    dim = 2 if pe[0].ndim == 4 else 1
    cos = _pad_dim(pe[0], dim, pad_len)
    sin = _pad_dim(pe[1], dim, pad_len)
    local_len = cos.shape[dim] // sp_size
    start = sp_rank * local_len
    end = start + local_len
    return (
        cos.narrow(dim, start, end - start).contiguous(),
        sin.narrow(dim, start, end - start).contiguous(),
    )


def shard_ltx25_video_args(
    video_args: TransformerArgs,
    *,
    sp_rank: int,
    sp_size: int,
    allow_padding: bool = True,
) -> tuple[TransformerArgs, int, int]:
    """Shard ``video_args`` across ``sp_size`` ranks along token axis ``dim=1``.

    Returns ``(sharded_args, original_tokens, padded_tokens)``.
    """
    if sp_size <= 1:
        total = int(video_args.x.shape[1])
        return video_args, total, total
    if sp_rank < 0 or sp_rank >= sp_size:
        raise ValueError(f"sp_rank {sp_rank} out of range for sp_size={sp_size}")

    orig_tokens = int(video_args.x.shape[1])
    rem = orig_tokens % sp_size
    if rem != 0 and not allow_padding:
        raise ValueError(f"video token count {orig_tokens} is not divisible by sequence_parallel={sp_size}")
    pad_len = (sp_size - rem) % sp_size
    padded_tokens = orig_tokens + pad_len
    local_tokens = padded_tokens // sp_size
    start = sp_rank * local_tokens

    def _shard_token_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or tensor.ndim < 2 or int(tensor.shape[1]) != orig_tokens:
            return tensor
        padded = _pad_dim(tensor, 1, pad_len)
        return padded.narrow(1, start, local_tokens).contiguous()

    context_mask = video_args.context_mask
    if context_mask is not None and context_mask.ndim == 4 and int(context_mask.shape[2]) == orig_tokens:
        padded_cm = _pad_dim(context_mask, 2, pad_len)
        context_mask = padded_cm.narrow(2, start, local_tokens).contiguous()

    sharded = replace(
        video_args,
        x=_pad_dim(video_args.x, 1, pad_len).narrow(1, start, local_tokens).contiguous(),
        context_mask=context_mask,
        timesteps=cast(torch.Tensor, _shard_token_tensor(video_args.timesteps)),
        embedded_timestep=cast(torch.Tensor, _shard_token_tensor(video_args.embedded_timestep)),
        positional_embeddings=cast(
            tuple[torch.Tensor, torch.Tensor],
            _shard_rope_tuple(video_args.positional_embeddings, sp_rank=sp_rank, sp_size=sp_size, pad_len=pad_len),
        ),
        cross_positional_embeddings=_shard_rope_tuple(
            video_args.cross_positional_embeddings,
            sp_rank=sp_rank,
            sp_size=sp_size,
            pad_len=pad_len,
        ),
        cross_scale_shift_timestep=_shard_token_tensor(video_args.cross_scale_shift_timestep),
        cross_gate_timestep=_shard_token_tensor(video_args.cross_gate_timestep),
    )
    return sharded, orig_tokens, padded_tokens


def _all_gather_tokens(
    tensor: torch.Tensor,
    *,
    dim: int,
    orig_len: int,
    sp_group: Any,
    sp_size: int,
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
) -> torch.Tensor:
    if sp_size <= 1:
        return tensor
    chunks = [torch.empty_like(tensor) for _ in range(sp_size)]
    if all_gather_fn is not None:
        all_gather_fn(chunks, tensor.contiguous(), sp_group)
    else:
        dist.all_gather(chunks, tensor.contiguous(), group=sp_group)
    full = torch.cat(chunks, dim=dim)
    if full.shape[dim] > orig_len:
        full = full.narrow(dim, 0, orig_len)
    return full.contiguous()


def gather_ltx25_video_hidden(
    local_x: torch.Tensor,
    *,
    orig_tokens: int,
    sp_group: Any,
    sp_size: int,
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
) -> torch.Tensor:
    """Reassemble sharded ``(B, S_local, D)`` video tokens and strip SP padding."""
    return _all_gather_tokens(
        local_x,
        dim=1,
        orig_len=orig_tokens,
        sp_group=sp_group,
        sp_size=sp_size,
        all_gather_fn=all_gather_fn,
    )


class LTX25UlyssesAttentionAdapter:
    """Reversible Ulysses Sequence Parallel hook for LTX-2.5 video self-attn & V2A CA.

    Wraps each block's ``attn1._run_attention`` with Ulysses all-to-all
    ``(B, S_local, H, D) <-> (B, S_full, H/sp, D)`` while delegating local
    attention to the already-bound backend (FA3, Sol-Attn, or PyTorch SDPA).
    Optionally wraps ``video_to_audio_attn.forward`` to gather video K/V and
    rotary embeddings across the SP group so bidirectional A<->V cross-attention
    preserves exact mathematical parity with single-GPU execution.
    """

    def __init__(
        self,
        model: LTXModel,
        *,
        sp_size: int,
        sp_rank: int,
        sp_group: Any = None,
        gather_v2a_video_kv: bool = True,
        all_to_all_many_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
        all_to_all_fn: Callable[..., torch.Tensor] | None = None,
        all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
    ) -> None:
        if sp_size < 1:
            raise ValueError(f"sp_size must be >= 1, got {sp_size}")
        if sp_size > 1 and int(model.num_attention_heads) % sp_size != 0:
            raise ValueError(
                f"LTX-2.5 video heads ({model.num_attention_heads}) must be divisible by sp_size ({sp_size})"
            )
        self.model = model
        self.sp_size = int(sp_size)
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.gather_v2a_video_kv = bool(gather_v2a_video_kv)
        self._a2a_many = all_to_all_many_fn or all_to_all_many
        self._a2a = all_to_all_fn or all_to_all
        self._all_gather = all_gather_fn
        self._active_orig_tokens: int | None = None
        self._orig_attn1_runs: list[tuple[Attention, Any]] = []
        self._orig_v2a_forwards: list[tuple[Attention, Any]] = []
        self._installed = False

    def set_orig_tokens(self, orig_tokens: int | None) -> None:
        self._active_orig_tokens = orig_tokens

    def install(self) -> dict[str, int]:
        """Attach Ulysses hooks onto all video self-attention (and optional V2A) modules."""
        if self._installed or self.sp_size <= 1:
            return {"self_attn_wrapped": len(self._orig_attn1_runs), "v2a_wrapped": len(self._orig_v2a_forwards)}

        for block in self.model.transformer_blocks:
            attn1 = getattr(block, "attn1", None)
            if isinstance(attn1, Attention):
                orig_run = attn1._run_attention
                self._orig_attn1_runs.append((attn1, orig_run))
                attn1._run_attention = partial(self._ulysses_run_attention, attn1, orig_run)  # type: ignore[method-assign]

            if self.gather_v2a_video_kv:
                v2a = getattr(block, "video_to_audio_attn", None)
                if isinstance(v2a, Attention):
                    orig_fwd = v2a.forward
                    self._orig_v2a_forwards.append((v2a, orig_fwd))
                    v2a.forward = partial(self._ulysses_v2a_forward, v2a, orig_fwd)  # type: ignore[method-assign]

        self._installed = True
        return {
            "self_attn_wrapped": len(self._orig_attn1_runs),
            "v2a_wrapped": len(self._orig_v2a_forwards),
        }

    def restore(self) -> None:
        """Remove Ulysses hooks and restore original bound methods."""
        for attn1, orig_run in self._orig_attn1_runs:
            attn1._run_attention = orig_run  # type: ignore[method-assign]
        for v2a, orig_fwd in self._orig_v2a_forwards:
            v2a.forward = orig_fwd  # type: ignore[method-assign]
        self._orig_attn1_runs.clear()
        self._orig_v2a_forwards.clear()
        self._installed = False

    def _ulysses_run_attention(
        self,
        module: Attention,
        orig_run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, local_tokens, width = q.shape
        full_heads = int(module.heads)
        dim_head = int(module.dim_head)
        local_heads = full_heads // self.sp_size

        q4 = q.view(batch, local_tokens, full_heads, dim_head)
        k4 = k.view(batch, local_tokens, full_heads, dim_head)
        v4 = v.view(batch, local_tokens, full_heads, dim_head)
        q4, k4, v4 = self._a2a_many((q4, k4, v4), scatter_dim=2, gather_dim=1, group=self.sp_group)

        full_padded_tokens = int(q4.shape[1])
        orig_tokens = self._active_orig_tokens or full_padded_tokens
        trimmed = orig_tokens < full_padded_tokens
        if trimmed:
            q4 = q4.narrow(1, 0, orig_tokens)
            k4 = k4.narrow(1, 0, orig_tokens)
            v4 = v4.narrow(1, 0, orig_tokens)

        local_width = local_heads * dim_head
        q_flat = q4.reshape(batch, -1, local_width)
        k_flat = k4.reshape(batch, -1, local_width)
        v_flat = v4.reshape(batch, -1, local_width)

        prev_heads = module.heads
        prev_num_heads = module.num_heads
        module.heads = local_heads
        module.num_heads = local_heads
        try:
            out_flat = orig_run(q_flat, k_flat, v_flat, mask)
        finally:
            module.heads = prev_heads
            module.num_heads = prev_num_heads

        out4 = out_flat.view(batch, -1, local_heads, dim_head)
        if trimmed:
            out4 = _pad_dim(out4, 1, full_padded_tokens - orig_tokens)
        out4 = self._a2a(out4, scatter_dim=1, gather_dim=2, group=self.sp_group)
        return out4.reshape(batch, local_tokens, width)

    def _ulysses_v2a_forward(
        self,
        _module: Attention,
        orig_fwd: Callable[..., torch.Tensor],
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: Any = None,
        k_pe: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if context is not None and self.sp_size > 1:
            orig_tokens = self._active_orig_tokens or int(context.shape[1]) * self.sp_size
            context = _all_gather_tokens(
                context,
                dim=1,
                orig_len=orig_tokens,
                sp_group=self.sp_group,
                sp_size=self.sp_size,
                all_gather_fn=self._all_gather,
            )
            if isinstance(k_pe, tuple) and len(k_pe) == 2:
                rope_dim = 2 if k_pe[0].ndim == 4 else 1
                k_pe = (
                    _all_gather_tokens(
                        k_pe[0],
                        dim=rope_dim,
                        orig_len=orig_tokens,
                        sp_group=self.sp_group,
                        sp_size=self.sp_size,
                        all_gather_fn=self._all_gather,
                    ),
                    _all_gather_tokens(
                        k_pe[1],
                        dim=rope_dim,
                        orig_len=orig_tokens,
                        sp_group=self.sp_group,
                        sp_size=self.sp_size,
                        all_gather_fn=self._all_gather,
                    ),
                )
        return orig_fwd(x, context=context, mask=mask, pe=pe, k_pe=k_pe, **kwargs)


class LTX25CFGParallelRunner:
    """Evaluate positive & negative CFG branches concurrently across 2 ranks."""

    def __init__(
        self,
        *,
        cfg_group: Any = None,
        cfg_rank: int | None = None,
        cfg_world_size: int | None = None,
        all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
    ) -> None:
        self.cfg_group = cfg_group
        self._all_gather_fn = all_gather_fn
        if cfg_world_size is not None:
            self.world_size = int(cfg_world_size)
            if self.world_size <= 1:
                self.world_size = 1
                self.rank = 0
            elif cfg_rank is not None:
                self.rank = int(cfg_rank)
            elif dist.is_available() and dist.is_initialized():
                self.rank = int(dist.get_rank(cfg_group))
            else:
                raise ValueError("cfg_rank is required when cfg_world_size > 1 outside torch.distributed")
        elif dist.is_available() and dist.is_initialized():
            self.world_size = int(dist.get_world_size(cfg_group))
            self.rank = int(dist.get_rank(cfg_group))
        else:
            self.world_size = 1
            self.rank = 0
        self.cfg_steps: int = 0

    @property
    def active(self) -> bool:
        return self.world_size == 2

    def _gather_pair(self, local_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contig = local_tensor.contiguous()
        gathered = [torch.empty_like(contig), torch.empty_like(contig)]
        if self._all_gather_fn is not None:
            self._all_gather_fn(gathered, contig, self.cfg_group)
        else:
            dist.all_gather(gathered, contig, group=self.cfg_group)
        return gathered[0], gathered[1]

    def run_cfg_step(
        self,
        *,
        run_positive: Callable[[], tuple[torch.Tensor | None, torch.Tensor | None]],
        run_negative: Callable[[], tuple[torch.Tensor | None, torch.Tensor | None]],
        guidance_scale: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute ``neg + guidance_scale * (pos - neg)`` across 1 or 2 CFG ranks."""
        scale = float(guidance_scale)
        if scale == 1.0:
            return run_positive()

        if not self.active:
            pos_v, pos_a = run_positive()
            neg_v, neg_a = run_negative()
            out_v = None if pos_v is None or neg_v is None else neg_v + scale * (pos_v - neg_v)
            out_a = None if pos_a is None or neg_a is None else neg_a + scale * (pos_a - neg_a)
            return out_v, out_a

        self.cfg_steps += 1
        local_v, local_a = run_positive() if self.rank == 0 else run_negative()
        out_v = None
        if local_v is not None:
            pos_v, neg_v = self._gather_pair(local_v)
            out_v = neg_v + scale * (pos_v - neg_v)
        out_a = None
        if local_a is not None:
            pos_a, neg_a = self._gather_pair(local_a)
            out_a = neg_a + scale * (pos_a - neg_a)
        return out_v, out_a

    def report(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "world_size": self.world_size,
            "rank": self.rank,
            "cfg_steps": self.cfg_steps,
        }


# ──────────────────────────────────────────────────────────────────────────
# LTX-2.5 Acceleration Config & SOL-Engine Presets
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LTX25AccelerationConfig:
    """Declarative SOL-Engine acceleration stack for LTX-2.5 22B."""

    # Kernel fusion
    fuse_qkv: bool = True
    fuse_cross_kv: bool = True
    fused_ops: bool = True

    # Attention backend + 3D-tiled Sol-Attn
    attention_backend: str = "flash_attention_3"
    sol_attn: bool = False
    sol_tau: float = 0.5
    sol_thresh_type: Literal["exact", "diag"] = "exact"
    sol_kv_splits: Literal[1, 2, 4] = 2
    sol_tile_3d: tuple[int, int, int] | None = (1, 8, 8)
    sol_dense_blocks: tuple[int, ...] = (0, 1, -1)
    sol_dense_steps: int = 1
    sol_min_tokens: int = 4096

    # Hopper rowwise FP8 W8A8
    quantization: Literal["none", "fp8"] = "none"
    fp8_scaling: Literal["rowwise", "tensorwise"] = "rowwise"
    fp8_use_fast_accum: bool = False
    fp8_exclude_edge_blocks: bool = True
    fp8_exclude_audio: bool = False
    fp8_fuse_ffn: bool = True
    fp8_keep_dense_fallback: bool = False
    fp8_min_features: int = 1024

    # FirstBlockCache (FBCache)
    fbcache: LTX25FirstBlockCacheConfig | None = None

    # Execution graph
    cuda_graph: Literal["off", "whole", "piecewise"] = "piecewise"
    cuda_graph_warmup: int = 1
    cuda_graph_max_graphs: int = 16
    cuda_graph_clone_outputs: bool = True

    # Video & Audio VAE acceleration
    vae_channels_last_3d: bool = True
    vae_compile: bool = True
    vae_compile_mode: str = "default"
    vae_tiled: bool = False
    diffvae_optimization: str = "combined_compile"
    audio_cuda_graph: bool = True
    parallel_audio: bool = True
    compile_vocoder: bool = True

    # Multi-GPU parallelism
    parallel: LTX25ParallelConfig = field(default_factory=LTX25ParallelConfig)

    # Human-readable audit notes
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.quantization not in ("none", "fp8"):
            raise ValueError(f"unsupported LTX-2.5 quantization: {self.quantization!r}")
        if self.fp8_scaling not in ("rowwise", "tensorwise"):
            raise ValueError(f"unsupported LTX-2.5 fp8_scaling: {self.fp8_scaling!r}")
        if self.cuda_graph not in ("off", "whole", "piecewise"):
            raise ValueError(f"unsupported LTX-2.5 cuda_graph: {self.cuda_graph!r}")
        if self.sol_thresh_type not in ("exact", "diag"):
            raise ValueError(f"unsupported LTX-2.5 sol_thresh_type: {self.sol_thresh_type!r}")
        if isinstance(self.sol_kv_splits, bool) or self.sol_kv_splits not in (1, 2, 4):
            raise ValueError(f"sol_kv_splits must be 1, 2, or 4, got {self.sol_kv_splits!r}")
        _validate_float_flag(self.sol_tau, name="sol_tau", minimum=0.0)
        _validate_int_flag(self.sol_dense_steps, name="sol_dense_steps", minimum=0)
        _validate_int_flag(self.sol_min_tokens, name="sol_min_tokens", minimum=0)
        _validate_int_flag(self.cuda_graph_warmup, name="cuda_graph_warmup", minimum=0)
        _validate_int_flag(self.cuda_graph_max_graphs, name="cuda_graph_max_graphs", minimum=1)
        _validate_int_flag(self.fp8_min_features, name="fp8_min_features", minimum=1)
        if self.sol_tile_3d is not None and (
            len(self.sol_tile_3d) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.sol_tile_3d)
        ):
            raise ValueError(f"sol_tile_3d must be 3 positive ints, got {self.sol_tile_3d!r}")
        if self.fbcache is not None and self.fbcache.enabled and self.cuda_graph == "whole":
            raise ValueError(
                "LTX-2.5 FirstBlockCache cannot use cuda_graph='whole'; use cuda_graph='piecewise' "
                "(captures Block-0 and Tail-1..47 independently) or cuda_graph='off'"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fbcache"] = self.fbcache.to_dict() if self.fbcache is not None else None
        payload["parallel"] = self.parallel.to_dict()
        return payload


def _resolve_hopper_dense_attention(sm_major: int) -> str:
    if int(sm_major) == 9 and attention_backend_capability("flash_attention_3").usable:
        return "flash_attention_3"
    return "torch"


def ltx25_sol_profile(
    tier: Literal[
        "baseline", "l0", "fused_exact", "l1", "sol_high_fidelity", "l2", "sol_balanced", "l3", "sol_fast", "l4"
    ]
    | str,
    *,
    variant: Literal["distilled", "dev"] = "distilled",
    sm_major: int = 9,
    parallel: LTX25ParallelConfig | None = None,
    fbcache_threshold: float | None = None,
    fp8_use_fast_accum: bool | None = None,
    sol_attn: bool | None = None,
    sol_tau: float | None = None,
    sol_kv_splits: Literal[1, 2, 4] = 2,
    cuda_graph: Literal["off", "whole", "piecewise"] | None = None,
    vae_compile: bool | None = None,
) -> LTX25AccelerationConfig:
    """Build a validated SOL-Engine acceleration preset for LTX-2.5 22B."""
    normalized = str(tier).strip().lower()
    is_dev = str(variant).strip().lower() == "dev"
    fp8_ok = int(sm_major) >= 9
    dense_attn = _resolve_hopper_dense_attention(sm_major)
    eff_parallel = parallel or LTX25ParallelConfig()

    if normalized in {"baseline", "l0"}:
        return LTX25AccelerationConfig(
            fuse_qkv=False,
            fuse_cross_kv=False,
            fused_ops=False,
            attention_backend="torch",
            sol_attn=False,
            quantization="none",
            fbcache=None,
            cuda_graph="off",
            vae_channels_last_3d=False,
            vae_compile=False,
            vae_tiled=True,
            diffvae_optimization="chunked_eager",
            audio_cuda_graph=False,
            parallel_audio=False,
            compile_vocoder=False,
            parallel=eff_parallel,
            notes=("L0 BF16 eager PyTorch SDPA baseline (no fusion, no FP8, no graph, eager VAE)",),
        )

    if normalized in {"fused_exact", "l1"}:
        return LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=True,
            attention_backend=dense_attn,
            sol_attn=False,
            quantization="none",
            fbcache=None,
            cuda_graph=cuda_graph or "whole",
            vae_channels_last_3d=True,
            vae_compile=True if vae_compile is None else bool(vae_compile),
            vae_tiled=False,
            diffvae_optimization="combined_compile",
            audio_cuda_graph=True,
            parallel_audio=True,
            compile_vocoder=True,
            parallel=eff_parallel,
            notes=(f"L1 lossless BF16 kernel fusion + {dense_attn} + CUDA Graph + accelerated VAE",),
        )

    if normalized in {"sol_high_fidelity", "l2"}:
        eff_fast_accum = False if fp8_use_fast_accum is None else bool(fp8_use_fast_accum)
        eff_sol = True if sol_attn is None else bool(sol_attn)
        fb_cfg = (
            LTX25FirstBlockCacheConfig(
                enabled=True,
                residual_diff_threshold=0.05 if fbcache_threshold is None else float(fbcache_threshold),
                dense_first=3 if is_dev else 2,
                dense_last=3 if is_dev else 2,
                max_consecutive_skips=1,
                downsample_factor=4,
                cumulative=True,
                min_steps_to_enable=6,
            )
            if is_dev or fbcache_threshold is not None
            else None
        )
        return LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=True,
            attention_backend=dense_attn,
            sol_attn=eff_sol and int(sm_major) == 9,
            sol_tau=0.25 if sol_tau is None else float(sol_tau),
            sol_thresh_type="exact",
            sol_kv_splits=sol_kv_splits,
            sol_tile_3d=(1, 8, 8),
            sol_dense_blocks=(0, 1, -1),
            sol_dense_steps=1,
            sol_min_tokens=4096,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=eff_fast_accum,
            fp8_exclude_edge_blocks=True,
            fp8_fuse_ffn=True,
            fbcache=fb_cfg,
            cuda_graph=cuda_graph or "piecewise",
            vae_channels_last_3d=True,
            vae_compile=True if vae_compile is None else bool(vae_compile),
            vae_tiled=False,
            diffvae_optimization="combined_compile",
            audio_cuda_graph=True,
            parallel_audio=True,
            compile_vocoder=True,
            parallel=eff_parallel,
            notes=(
                f"L2 high-fidelity: fused ops + {dense_attn} + FP8(exact_accum={not eff_fast_accum}) "
                "+ 3D-tiled Sol-Attn(tau=0.25) + conservative FBCache + piecewise CUDA Graph + compiled VAE",
            ),
        )

    if normalized in {"sol_balanced", "l3"}:
        # Calibrated default: exact FP8 accumulation on dev multi-step; tested parity on distilled.
        eff_fast_accum = (not is_dev) if fp8_use_fast_accum is None else bool(fp8_use_fast_accum)
        eff_sol = True if sol_attn is None else bool(sol_attn)
        default_fb_thresh = 0.08 if is_dev else 0.06
        fb_cfg = LTX25FirstBlockCacheConfig(
            enabled=True,
            residual_diff_threshold=default_fb_thresh if fbcache_threshold is None else float(fbcache_threshold),
            dense_first=2,
            dense_last=2,
            max_consecutive_skips=2 if is_dev else 1,
            downsample_factor=4,
            cumulative=True,
            min_steps_to_enable=4,
        )
        return LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=True,
            attention_backend=dense_attn,
            sol_attn=eff_sol and int(sm_major) == 9,
            sol_tau=0.5 if sol_tau is None else float(sol_tau),
            sol_thresh_type="exact",
            sol_kv_splits=sol_kv_splits,
            sol_tile_3d=(1, 8, 8),
            sol_dense_blocks=(0, 1, -1),
            sol_dense_steps=1,
            sol_min_tokens=4096,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=eff_fast_accum,
            fp8_exclude_edge_blocks=True,
            fp8_fuse_ffn=True,
            fbcache=fb_cfg,
            cuda_graph=cuda_graph or "piecewise",
            vae_channels_last_3d=True,
            vae_compile=True if vae_compile is None else bool(vae_compile),
            vae_tiled=False,
            diffvae_optimization="combined_compile",
            audio_cuda_graph=True,
            parallel_audio=True,
            compile_vocoder=True,
            parallel=eff_parallel,
            notes=(
                f"L3 SOL-Balanced: FBCache + fused ops + {dense_attn} + FP8 W8A8 "
                "+ 3D-tiled Sol-Attn(tau=0.5) + split piecewise CUDA Graph + accelerated VAE",
            ),
        )

    if normalized in {"sol_fast", "l4"}:
        eff_fast_accum = (not is_dev) if fp8_use_fast_accum is None else bool(fp8_use_fast_accum)
        eff_sol = True if sol_attn is None else bool(sol_attn)
        default_fb_thresh = 0.14 if is_dev else 0.10
        fb_cfg = LTX25FirstBlockCacheConfig(
            enabled=True,
            residual_diff_threshold=default_fb_thresh if fbcache_threshold is None else float(fbcache_threshold),
            dense_first=2,
            dense_last=1,
            max_consecutive_skips=3 if is_dev else 2,
            downsample_factor=4,
            cumulative=True,
            min_steps_to_enable=4,
        )
        return LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=True,
            attention_backend=dense_attn,
            sol_attn=eff_sol and int(sm_major) == 9,
            sol_tau=0.75 if sol_tau is None else float(sol_tau),
            sol_thresh_type="exact",
            sol_kv_splits=sol_kv_splits,
            sol_tile_3d=(1, 8, 8),
            sol_dense_blocks=(0, 1, -1),
            sol_dense_steps=0,
            sol_min_tokens=4096,
            quantization="fp8" if fp8_ok else "none",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=eff_fast_accum,
            fp8_exclude_edge_blocks=False,
            fp8_fuse_ffn=True,
            fbcache=fb_cfg,
            cuda_graph=cuda_graph or "piecewise",
            vae_channels_last_3d=True,
            vae_compile=True if vae_compile is None else bool(vae_compile),
            vae_tiled=False,
            diffvae_optimization="combined_compile",
            audio_cuda_graph=True,
            parallel_audio=True,
            compile_vocoder=True,
            parallel=eff_parallel,
            notes=(
                f"L4 SOL-Fast: calibrated aggressive FBCache + fused ops + {dense_attn} + FP8 W8A8 "
                "+ 3D-tiled Sol-Attn + split piecewise CUDA Graph + accelerated VAE",
            ),
        )

    raise ValueError(
        f"unsupported LTX-2.5 SOL profile tier {tier!r}; expected one of: "
        "baseline/l0, fused_exact/l1, sol_high_fidelity/l2, sol_balanced/l3, sol_fast/l4"
    )


def coerce_ltx25_acceleration_config(
    raw: LTX25AccelerationConfig | Mapping[str, Any] | str | None,
    *,
    variant: Literal["distilled", "dev"] = "distilled",
    sm_major: int = 9,
) -> LTX25AccelerationConfig | None:
    """Coerce a profile name, mapping, or config dataclass into ``LTX25AccelerationConfig``."""
    if raw is None:
        return None
    if isinstance(raw, LTX25AccelerationConfig):
        return raw
    if isinstance(raw, str):
        return ltx25_sol_profile(raw, variant=variant, sm_major=sm_major)
    if isinstance(raw, Mapping):
        data = dict(raw)
        fb = data.get("fbcache")
        if isinstance(fb, Mapping):
            data["fbcache"] = LTX25FirstBlockCacheConfig(**dict(fb))
        par = data.get("parallel")
        if isinstance(par, Mapping):
            data["parallel"] = LTX25ParallelConfig(**dict(par))
        if "sol_tile_3d" in data and data["sol_tile_3d"] is not None:
            data["sol_tile_3d"] = tuple(int(v) for v in data["sol_tile_3d"])
        if "sol_dense_blocks" in data and data["sol_dense_blocks"] is not None:
            data["sol_dense_blocks"] = tuple(int(v) for v in data["sol_dense_blocks"])
        if "notes" in data and data["notes"] is not None:
            data["notes"] = tuple(str(v) for v in data["notes"])
        return LTX25AccelerationConfig(**data)
    raise TypeError(f"unsupported LTX-2.5 acceleration specification: {type(raw).__name__}")


# ──────────────────────────────────────────────────────────────────────────
# Transformer Acceleration & Split Piecewise-Graph + FBCache Executor
# ──────────────────────────────────────────────────────────────────────────


def accelerate_ltx25_transformer(
    model: LTXModel,
    config: LTX25AccelerationConfig,
    *,
    device: torch.device | str | None = None,
    sp_group: Any = None,
    sp_rank: int = 0,
    all_to_all_many_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
    all_to_all_fn: Callable[..., torch.Tensor] | None = None,
    all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
) -> tuple[dict[str, Any], LTX25UlyssesAttentionAdapter | None]:
    """Apply validated post-load SOL-Engine passes to ``LTXModel`` in place.

    Pass order matters:
    1. Fuse QKV and Cross-KV projections *before* FP8 so fused linears convert
       into single wide ``Float8Linear`` layers.
    2. Install LTX Triton fused ops (QK-RMSNorm+RoPE, post-SA+AdaLN, gated
       attention+FP8, dual AV AdaLN).
    3. Configure exact dense attention backend (Hopper FA3 or PyTorch SDPA) and
       optionally layer 3D-tiled ``sol_attn`` on video self-attention with dense
       edge blocks and dense fallback backend preservation.
    4. Quantize eligible linears to Hopper rowwise FP8 W8A8 + fused GELU-FP8 FFN.
    5. Install optional Ulysses Sequence Parallel hooks.
    """
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    receipt: dict[str, Any] = {"config": config.to_dict()}

    # 1. Projection fusion (must precede FP8 quantization).
    if config.fuse_qkv or config.fuse_cross_kv:
        receipt["projection_fusion"] = fuse_ltx_projections(
            model,
            fuse_self_qkv=config.fuse_qkv,
            fuse_cross_kv=config.fuse_cross_kv,
        )
    else:
        receipt["projection_fusion"] = {"fused_qkv": 0, "fused_kv": 0}

    # 2. Fused Triton ops.
    if config.fused_ops:
        receipt["fused_ops_blocks"] = int(enable_ltx_fused_ops(model))
    else:
        receipt["fused_ops_blocks"] = 0

    # 3. Attention routing.
    req_dense = normalize_attention_backend(config.attention_backend)
    resolved_dense = resolve_attention_backend(req_dense, target_device)
    if resolved_dense == "torch":
        # Explicitly bind PyTorch SDPA on every Attention so L0 baseline does not
        # stay silently on split-checkpoint FA3.
        pt_attn = PytorchAttention()
        for child in model.modules():
            if isinstance(child, Attention):
                child.attention_function = pt_attn
                child.masked_attention_function = pt_attn
                child.set_attention_backend("torch")
        dense_configured = sum(1 for c in model.modules() if isinstance(c, Attention))
    else:
        dense_configured = model.configure_attention_backend(resolved_dense)

    sol_configured = 0
    sol_schedule: dict[str, Any] | None = None
    if config.sol_attn:
        sol_resolved = resolve_attention_backend("sol_attn", target_device)
        sol_opts: dict[str, Any] = {
            "tau": float(config.sol_tau),
            "thresh_type": config.sol_thresh_type,
            "compress_kv": True,
            "min_tokens": int(config.sol_min_tokens),
        }
        sm_major = torch.cuda.get_device_capability(target_device)[0] if target_device.type == "cuda" else 0
        if int(config.sol_kv_splits) > 1 and sm_major == 9:
            sol_opts["kv_splits"] = int(config.sol_kv_splits)
            sol_opts["allow_multi_split"] = True
        sol_configured = model.configure_attention_backend(
            sol_resolved,
            self_only=True,
            backend_options=sol_opts,
            dense_blocks=config.sol_dense_blocks,
            dense_backend=resolved_dense,
        )
        sol_schedule = {
            "dense_steps": int(config.sol_dense_steps),
            "dense_blocks": list(config.sol_dense_blocks),
            "tile_3d": config.sol_tile_3d,
            "min_tokens": int(config.sol_min_tokens),
            "tau": float(config.sol_tau),
            "kv_splits": int(sol_opts.get("kv_splits", 1)),
            "dense_backend": resolved_dense,
        }
        model.__dict__["_worldfoundry_sol_schedule"] = sol_schedule
    else:
        model.__dict__.pop("_worldfoundry_sol_schedule", None)

    receipt["attention"] = {
        "dense_requested": req_dense,
        "dense_resolved": resolved_dense,
        "dense_modules_configured": int(dense_configured),
        "sol_attn_enabled": bool(config.sol_attn),
        "sol_attn_modules_configured": int(sol_configured),
        "sol_schedule": sol_schedule,
    }

    # 4. Rowwise FP8 W8A8 quantization + fused GELU-FP8 FFN.
    if config.quantization == "fp8":
        excludes: list[str] = list(LTX_DEFAULT_FP8_EXCLUDES)
        if config.fp8_exclude_audio:
            excludes.extend(LTX_AUDIO_FP8_EXCLUDES)
        num_layers = len(model.transformer_blocks)
        if config.fp8_exclude_edge_blocks and num_layers >= 2:
            excludes.extend(["transformer_blocks.0.", f"transformer_blocks.{num_layers - 1}."])
        quant_stats = enable_ltx_fp8_quantization(
            model,
            exclude=tuple(dict.fromkeys(excludes)),
            min_features=int(config.fp8_min_features),
            scaling=config.fp8_scaling,
            use_fast_accum=bool(config.fp8_use_fast_accum),
            keep_dense_fallback=bool(config.fp8_keep_dense_fallback),
            fuse_fp8_ffn=bool(config.fp8_fuse_ffn),
        )
        receipt["quantization"] = {
            "mode": "fp8",
            "scaling": config.fp8_scaling,
            "use_fast_accum": bool(config.fp8_use_fast_accum),
            "exclude_edge_blocks": bool(config.fp8_exclude_edge_blocks),
            "exclude_audio": bool(config.fp8_exclude_audio),
            "keep_dense_fallback": bool(config.fp8_keep_dense_fallback),
            **quant_stats,
        }
    else:
        receipt["quantization"] = {"mode": "none", "replaced_linears": 0, "fused_fp8_ffn": 0}

    # 5. Optional Ulysses Sequence Parallel adapter.
    ulysses_adapter: LTX25UlyssesAttentionAdapter | None = None
    if config.parallel.sequence_parallel > 1:
        ulysses_adapter = LTX25UlyssesAttentionAdapter(
            model,
            sp_size=config.parallel.sequence_parallel,
            sp_rank=sp_rank,
            sp_group=sp_group,
            gather_v2a_video_kv=config.parallel.gather_v2a_video_kv,
            all_to_all_many_fn=all_to_all_many_fn,
            all_to_all_fn=all_to_all_fn,
            all_gather_fn=all_gather_fn,
        )
        receipt["sequence_parallel"] = {
            "enabled": True,
            "sp_size": config.parallel.sequence_parallel,
            "sp_rank": sp_rank,
            **ulysses_adapter.install(),
        }
    else:
        receipt["sequence_parallel"] = {"enabled": False, "sp_size": 1, "sp_rank": 0}

    return receipt, ulysses_adapter


class LTX25AcceleratedExecutor:
    """Zero-sync executor pairing LTX-2.5 with FBCache, split CUDA Graphs, and SP/CFG-P."""

    def __init__(
        self,
        model: LTXModel,
        config: LTX25AccelerationConfig,
        *,
        device: torch.device | str | None = None,
        sp_group: Any = None,
        sp_rank: int = 0,
        all_to_all_many_fn: Callable[..., tuple[torch.Tensor, ...]] | None = None,
        all_to_all_fn: Callable[..., torch.Tensor] | None = None,
        all_gather_fn: Callable[[list[torch.Tensor], torch.Tensor, Any], None] | None = None,
        all_reduce_fn: Callable[[torch.Tensor, Any], None] | None = None,
        apply_passes: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.sp_group = sp_group
        self.sp_rank = int(sp_rank)
        self._all_gather_fn = all_gather_fn
        self._all_reduce_fn = all_reduce_fn
        self._video_grid_cache: dict[int, Any] = {}
        self._tile_perm_cache: dict[Any, tuple[torch.Tensor, torch.Tensor, int] | None] = {}
        self._fbcache_reports: list[dict[str, Any]] = []
        self._active_fbcaches: dict[tuple[int, str], LTX25FirstBlockCache] = {}
        self._ulysses: LTX25UlyssesAttentionAdapter | None = None
        self._segment_call_counts: dict[str, list[int]] = {}
        self._graph_captured_receipts: dict[str, dict[str, Any]] = {}
        self._graph_replayed_window: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "attention_provider_calls": {},
        }
        self._graph_replayed_lifetime: dict[str, Any] = {
            "fp8_kernel_calls": 0,
            "attention_provider_calls": {},
        }

        if apply_passes:
            self._install_receipt, self._ulysses = accelerate_ltx25_transformer(
                model,
                config,
                device=device,
                sp_group=sp_group,
                sp_rank=sp_rank,
                all_to_all_many_fn=all_to_all_many_fn,
                all_to_all_fn=all_to_all_fn,
                all_gather_fn=all_gather_fn,
            )
        else:
            self._install_receipt = {"config": config.to_dict()}

        graph_enabled = config.cuda_graph != "off"
        self._graph_runner: PiecewiseBlockGraphRunner | None = (
            PiecewiseBlockGraphRunner(
                warmup=int(config.cuda_graph_warmup),
                max_graphs_per_segment=int(config.cuda_graph_max_graphs),
                clone_outputs=bool(config.cuda_graph_clone_outputs),
                enabled=True,
            )
            if graph_enabled
            else None
        )

    def invalidate_cuda_graphs(self) -> None:
        """Drop all captured CUDA graphs and release stale segment memory pools."""
        if self._graph_runner is not None:
            self._graph_runner.invalidate()
            for seg_runner in getattr(self._graph_runner, "_segments", {}).values():
                if hasattr(seg_runner, "_pool"):
                    seg_runner._pool = None
        self._graph_captured_receipts.clear()

    def begin_request(self) -> None:
        """Reset per-request FBCache state and runtime telemetry windows."""
        self._active_fbcaches.clear()
        self._fbcache_reports.clear()
        self._graph_replayed_window = {
            "fp8_kernel_calls": 0,
            "attention_provider_calls": {},
        }
        if self._graph_runner is not None:
            self._graph_runner.begin_request_window()
        reset_attention_provider_runtime()
        from worldfoundry.core.acceleration.quantization.linear import reset_quantization_runtime_window

        reset_quantization_runtime_window(self.model)

    def _snapshot_fp8_calls(self) -> int:
        from worldfoundry.core.acceleration.quantization.linear import quantization_runtime_report

        rep = quantization_runtime_report(self.model)
        if isinstance(rep, Mapping):
            return int(rep.get("lifetime_low_precision_kernel_calls", 0))
        return 0

    @staticmethod
    def _snapshot_attn_successes() -> dict[str, int]:
        return {
            str(name): int(row.get("successes", 0))
            for name, row in attention_provider_runtime_report().items()
            if isinstance(row, Mapping)
        }

    def _record_graph_replay_receipt(self, seg_name: str) -> None:
        baked = self._graph_captured_receipts.get(seg_name)
        if not baked:
            return
        fp8_calls = int(baked.get("fp8_kernel_calls", 0))
        attn_calls = baked.get("attention_provider_calls") or {}
        for bucket in (self._graph_replayed_window, self._graph_replayed_lifetime):
            bucket["fp8_kernel_calls"] = int(bucket["fp8_kernel_calls"]) + fp8_calls
            dst_attn = bucket["attention_provider_calls"]
            for prov, count in attn_calls.items():
                dst_attn[prov] = int(dst_attn.get(prov, 0)) + int(count)

    def finalize_request_caches(self) -> list[dict[str, Any]]:
        """Snapshot all active per-(stage, branch) FBCache receipts."""
        self._fbcache_reports = [cache.receipt() for cache in self._active_fbcaches.values()]
        return list(self._fbcache_reports)

    def get_fbcache(self, *, stage_index: int = 0, branch: str = "positive") -> LTX25FirstBlockCache | None:
        if self.config.fbcache is None or not self.config.fbcache.enabled:
            return None
        key = (int(stage_index), str(branch))
        cache = self._active_fbcaches.get(key)
        if cache is None:
            cache = LTX25FirstBlockCache(self.config.fbcache, branch=branch, stage_index=stage_index)
            self._active_fbcaches[key] = cache
        return cache

    def _run_block_slice_eager(
        self,
        start: int,
        stop: int,
        video_args: TransformerArgs | None,
        audio_args: TransformerArgs | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        v, a = video_args, audio_args
        for idx in range(start, stop):
            v, a = self.model.transformer_blocks[idx](v, a)
        return (None if v is None else v.x, None if a is None else a.x)

    def _run_block_slice(
        self,
        seg_name: str,
        start: int,
        stop: int,
        video_args: TransformerArgs | None,
        audio_args: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        can_graph = (
            self._graph_runner is not None
            and not self.model.training
            and not torch.is_grad_enabled()
            and (video_args is None or (video_args.x.device.type == "cuda" and video_args.video_control is None))
            and (audio_args is None or audio_args.x.device.type == "cuda")
        )
        if can_graph and self._graph_runner is not None:
            counter = self._segment_call_counts.setdefault(seg_name, [0])

            def _counted_slice(
                v_args: TransformerArgs | None,
                a_args: TransformerArgs | None,
                _start: int = start,
                _stop: int = stop,
                _counter: list[int] = counter,
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
                _counter[0] += 1
                return self._run_block_slice_eager(_start, _stop, v_args, a_args)

            seg_runner = self._graph_runner.get_or_register_segment(
                seg_name,
                _counted_slice,
                extra_key=f"ltx25:{id(self.model)}:{seg_name}",
            )
            prev_cap = int(seg_runner.stats.get("capture", 0))
            prev_rep = int(seg_runner.stats.get("replay", 0))
            need_bake = seg_name not in self._graph_captured_receipts
            prev_calls = counter[0] if need_bake else 0
            prev_fp8 = self._snapshot_fp8_calls() if need_bake else 0
            prev_attn = self._snapshot_attn_successes() if need_bake else {}
            vx, ax = self._graph_runner.run_segment(seg_name, video_args, audio_args)
            new_cap = int(seg_runner.stats.get("capture", 0))
            new_rep = int(seg_runner.stats.get("replay", 0))
            if need_bake and new_cap > prev_cap:
                n_calls = max(1, counter[0] - prev_calls)
                new_fp8 = self._snapshot_fp8_calls()
                new_attn = self._snapshot_attn_successes()
                attn_delta = {
                    prov: (cnt - int(prev_attn.get(prov, 0))) // n_calls
                    for prov, cnt in new_attn.items()
                    if (cnt - int(prev_attn.get(prov, 0))) // n_calls > 0
                }
                self._graph_captured_receipts[seg_name] = {
                    "fp8_kernel_calls": max(0, new_fp8 - prev_fp8) // n_calls,
                    "attention_provider_calls": attn_delta,
                }
            elif new_cap == prev_cap and new_rep > prev_rep:
                self._record_graph_replay_receipt(seg_name)
        else:
            vx, ax = self._run_block_slice_eager(start, stop, video_args, audio_args)
        v_out = None if video_args is None or vx is None else replace(video_args, x=vx)
        a_out = None if audio_args is None or ax is None else replace(audio_args, x=ax)
        return v_out, a_out

    def _resolve_tile_perm(
        self,
        video: Modality,
        video_args: TransformerArgs,
        tile_3d: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        grid_shape = resolve_cached_video_grid_shape(self._video_grid_cache, video.positions)
        if grid_shape is None or grid_shape[0] * grid_shape[1] * grid_shape[2] != int(video_args.x.shape[1]):
            return None
        tf, th, tw = tile_3d
        key = (grid_shape[0], grid_shape[1], grid_shape[2], tf, th, tw, video_args.x.device)
        if key in self._tile_perm_cache:
            return self._tile_perm_cache[key]
        perm = _build_tile_3d_permutation(
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            tf,
            th,
            tw,
            video_args.x.device,
        )
        if len(self._tile_perm_cache) >= 16:
            self._tile_perm_cache.pop(next(iter(self._tile_perm_cache)))
        self._tile_perm_cache[key] = perm
        return perm

    def forward_velocity(
        self,
        video: Modality | None,
        audio: Modality | None,
        *,
        step_index: int = 0,
        total_steps: int | None = None,
        stage_index: int = 0,
        branch: str = "positive",
        fbcache: LTX25FirstBlockCache | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run one accelerated LTX-2.5 velocity prediction step."""
        video_args = self.model.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.model.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        # Resolve Sol-Attn dense warmup and 3D-tile permutation outside graph capture.
        sol_schedule = getattr(self.model, "_worldfoundry_sol_schedule", None)
        sol_force_dense = False
        orig_video_args = video_args
        tile_inv_perm: torch.Tensor | None = None
        tile_sink_tokens: int | None = None
        if isinstance(sol_schedule, Mapping):
            dense_steps = int(sol_schedule.get("dense_steps", 0))
            sol_force_dense = dense_steps > 0 and int(step_index) < dense_steps
            tile_3d = sol_schedule.get("tile_3d")
            if (
                tile_3d is not None
                and not sol_force_dense
                and video is not None
                and video_args is not None
                and video_args.enabled
                and int(video_args.x.shape[1]) >= int(sol_schedule.get("min_tokens", 0))
                and not self.model.training
                and not torch.is_grad_enabled()
                and video_args.self_attention_mask is None
                and video_args.video_control is None
                and video_args.attention_cache is None
            ):
                cached_perm = self._resolve_tile_perm(video, video_args, tuple(int(v) for v in tile_3d))
                if cached_perm is not None:
                    tile_perm, tile_inv_perm, sink = cached_perm
                    video_args = _permute_video_args(video_args, tile_perm)
                    if sink > 0:
                        tile_sink_tokens = sink

        # Shard across Ulysses SP ranks if enabled.
        sp_size = int(self.config.parallel.sequence_parallel)
        sp_orig_tokens: int | None = None
        pre_sp_video_args = video_args
        if sp_size > 1 and video_args is not None:
            video_args, sp_orig_tokens, _ = shard_ltx25_video_args(
                video_args,
                sp_rank=self.sp_rank,
                sp_size=sp_size,
                allow_padding=self.config.parallel.pad_sequence_to_sp,
            )
            if self._ulysses is not None:
                self._ulysses.set_orig_tokens(sp_orig_tokens)

        sol_attn_modules: list[Attention] = []
        sol_prev_opts: list[tuple[Attention, dict[str, Any] | None]] = []
        clamped_kv_splits: int | None = None
        if video_args is not None and not sol_force_dense and isinstance(sol_schedule, Mapping):
            req_splits = int(sol_schedule.get("kv_splits", 1))
            if req_splits > 1:
                t_eff = int(video_args.x.shape[1]) if sp_orig_tokens is None else int(sp_orig_tokens)
                route_groups = ((t_eff + 63) // 64 + 63) // 64
                if route_groups < req_splits:
                    clamped_kv_splits = 4 if route_groups >= 4 else (2 if route_groups >= 2 else 1)

        if sol_force_dense or tile_sink_tokens is not None or clamped_kv_splits is not None:
            for block in self.model.transformer_blocks:
                attn1 = getattr(block, "attn1", None)
                if isinstance(attn1, Attention) and attn1.attention_backend == "sol_attn":
                    sol_attn_modules.append(attn1)
            for mod in sol_attn_modules:
                if sol_force_dense:
                    mod._worldfoundry_force_dense = True
                if tile_sink_tokens is not None:
                    mod._worldfoundry_sol_sink_tokens = tile_sink_tokens
                if clamped_kv_splits is not None:
                    sol_prev_opts.append((mod, mod._attention_backend_options))
                    updated_opts = dict(mod._attention_backend_options or {})
                    if clamped_kv_splits <= 1:
                        updated_opts.pop("kv_splits", None)
                        updated_opts.pop("allow_multi_split", None)
                    else:
                        updated_opts["kv_splits"] = clamped_kv_splits
                        updated_opts["allow_multi_split"] = True
                    mod._attention_backend_options = updated_opts

        seg_suffix = f"s{stage_index}"
        if sol_force_dense:
            seg_suffix = f"{seg_suffix}_dense"
        else:
            if tile_sink_tokens is not None:
                seg_suffix = f"{seg_suffix}_sink{tile_sink_tokens}"
            if clamped_kv_splits is not None:
                seg_suffix = f"{seg_suffix}_kv{clamped_kv_splits}"

        num_blocks = len(self.model.transformer_blocks)
        active_cache = fbcache if fbcache is not None else self.get_fbcache(stage_index=stage_index, branch=branch)

        try:
            if active_cache is not None and num_blocks >= 2:
                # Split execution: Segment A (block 0) -> Python FBCache probe -> Segment B (blocks 1..N-1).
                v1, a1 = self._run_block_slice(f"block0_{seg_suffix}", 0, 1, video_args, audio_args)
                can_hit, reason, metric, v_probe, a_probe = active_cache.evaluate_probe(
                    step=step_index,
                    total_steps=total_steps,
                    num_blocks=num_blocks,
                    block0_video_in=None if video_args is None else video_args.x,
                    block0_video_out=None if v1 is None else v1.x,
                    block0_audio_in=None if audio_args is None else audio_args.x,
                    block0_audio_out=None if a1 is None else a1.x,
                    sol_force_dense=sol_force_dense,
                    sp_size=sp_size,
                    sp_rank=self.sp_rank,
                    sp_orig_tokens=sp_orig_tokens,
                    sp_group=self.sp_group,
                    all_reduce_fn=self._all_reduce_fn,
                )
                tail_seg_name = f"tail_1_{num_blocks}_{seg_suffix}"
                if can_hit:
                    vx_out, ax_out = active_cache.apply_hit(
                        step=step_index,
                        num_blocks=num_blocks,
                        block0_video_out=None if v1 is None else v1.x,
                        block0_audio_out=None if a1 is None else a1.x,
                        v_probe=v_probe,
                        a_probe=a_probe,
                        metric=metric,
                        reason=reason,
                    )
                    if self._graph_runner is not None:
                        self._graph_runner.record_skip(tail_seg_name)
                    video_out = None if v1 is None or vx_out is None else replace(v1, x=vx_out)
                    audio_out = None if a1 is None or ax_out is None else replace(a1, x=ax_out)
                else:
                    video_out, audio_out = self._run_block_slice(tail_seg_name, 1, num_blocks, v1, a1)
                    active_cache.record_dense(
                        step=step_index,
                        num_blocks=num_blocks,
                        block0_video_out=None if v1 is None else v1.x,
                        final_video_out=None if video_out is None else video_out.x,
                        block0_audio_out=None if a1 is None else a1.x,
                        final_audio_out=None if audio_out is None else audio_out.x,
                        v_probe=v_probe,
                        a_probe=a_probe,
                        metric=metric,
                        reason=reason,
                        sol_force_dense=sol_force_dense,
                    )
            else:
                # Whole block stack in one segment.
                video_out, audio_out = self._run_block_slice(
                    f"blocks_0_{num_blocks}_{seg_suffix}",
                    0,
                    num_blocks,
                    video_args,
                    audio_args,
                )
        finally:
            for mod in sol_attn_modules:
                if sol_force_dense:
                    mod._worldfoundry_force_dense = False
                if tile_sink_tokens is not None:
                    mod._worldfoundry_sol_sink_tokens = None
            for mod, prev_opts in sol_prev_opts:
                mod._attention_backend_options = prev_opts
            if self._ulysses is not None:
                self._ulysses.set_orig_tokens(None)

        # Gather across Ulysses SP ranks before unpermuting & output head.
        if sp_size > 1 and video_out is not None and pre_sp_video_args is not None and sp_orig_tokens is not None:
            gathered_x = gather_ltx25_video_hidden(
                video_out.x,
                orig_tokens=sp_orig_tokens,
                sp_group=self.sp_group,
                sp_size=sp_size,
                all_gather_fn=self._all_gather_fn,
            )
            video_out = replace(pre_sp_video_args, x=gathered_x)

        if tile_inv_perm is not None and video_out is not None and orig_video_args is not None:
            video_out = replace(orig_video_args, x=video_out.x.index_select(1, tile_inv_perm))

        vx = (
            self.model._process_output(
                self.model.scale_shift_table,
                self.model.norm_out,
                self.model.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax = (
            self.model._process_output(
                self.model.audio_scale_shift_table,
                self.model.audio_norm_out,
                self.model.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    def acceleration_report(self) -> dict[str, Any]:
        """Emit an auditable, zero-fallback acceleration receipt."""
        from worldfoundry.core.acceleration.quantization.linear import quantization_runtime_report

        attn_stats = attention_provider_runtime_report()
        replayed_fp8 = int(self._graph_replayed_window["fp8_kernel_calls"])
        replayed_attn = dict(self._graph_replayed_window["attention_provider_calls"])
        attn_effective: dict[str, int] = {}
        for name, row in attn_stats.items():
            if isinstance(row, Mapping):
                attn_effective[str(name)] = int(row.get("successes", 0))
        for name, count in replayed_attn.items():
            attn_effective[str(name)] = attn_effective.get(str(name), 0) + int(count)

        raw_quant_runtime = quantization_runtime_report(self.model)
        quant_runtime: dict[str, Any] | None = None
        if isinstance(raw_quant_runtime, Mapping):
            quant_runtime = {k: v for k, v in raw_quant_runtime.items() if k != "layer_reports"}
            direct_lp = int(quant_runtime.get("low_precision_kernel_calls", 0))
            quant_runtime["graph_replayed_low_precision_kernel_calls"] = replayed_fp8
            quant_runtime["effective_low_precision_kernel_calls"] = direct_lp + replayed_fp8
            if direct_lp == 0 and replayed_fp8 > 0 and quant_runtime.get("effective") == "pending":
                quant_runtime["effective"] = "low-precision-graph-replayed"

        graph_report = self._graph_runner.report() if self._graph_runner is not None else {"enabled": False}
        fb_reports = (
            [cache.receipt() for cache in self._active_fbcaches.values()]
            if self._active_fbcaches
            else list(self._fbcache_reports)
        )

        # Audit zero fallbacks across attention providers, FP8 quantization, and CUDA Graphs.
        attn_errors = sum(int(row.get("errors", 0)) for row in attn_stats.values())
        attn_fallbacks = sum(int(row.get("fallbacks", 0)) for row in attn_stats.values())
        quant_fallbacks = 0
        if isinstance(quant_runtime, Mapping):
            quant_fallbacks = int(quant_runtime.get("dense_fallback_calls", 0))
            quant_fallbacks += int(quant_runtime.get("lifetime_dense_fallback_calls", 0))
        graph_errors = 0
        if isinstance(graph_report, Mapping) and graph_report.get("enabled"):
            for seg in (graph_report.get("segments") or {}).values():
                if isinstance(seg, Mapping):
                    graph_errors += int(seg.get("capture_failed", 0))
                    graph_errors += int(seg.get("disabled_signatures", 0))
                    lifetime = seg.get("lifetime")
                    if isinstance(lifetime, Mapping):
                        graph_errors += int(lifetime.get("capture_failed", 0))
        zero_fallbacks = attn_errors == 0 and attn_fallbacks == 0 and quant_fallbacks == 0 and graph_errors == 0

        return {
            "install": self._install_receipt,
            "attention_provider_runtime": attn_stats,
            "attention_provider_runtime_effective": attn_effective,
            "quantization_runtime": quant_runtime,
            "cuda_graph": graph_report,
            "graph_replayed_receipts": {
                "window": {
                    "fp8_kernel_calls": replayed_fp8,
                    "attention_provider_calls": replayed_attn,
                },
                "lifetime": {
                    "fp8_kernel_calls": int(self._graph_replayed_lifetime["fp8_kernel_calls"]),
                    "attention_provider_calls": dict(self._graph_replayed_lifetime["attention_provider_calls"]),
                },
            },
            "fbcache": fb_reports,
            "zero_fallbacks_verified": zero_fallbacks,
        }


# ──────────────────────────────────────────────────────────────────────────
# LTX-2.5 Video VAE Loaders & Acceleration (ConvVAE + DiffVAE)
# ──────────────────────────────────────────────────────────────────────────


def convert_ltx25_conv_video_decoder_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    """Map LTX-2.5 split Conv VAE decoder keys into ``LTXVideoDecoderModule``."""
    prefixes = {
        "vae.decoder.": "decoder.",
        "decoder.": "decoder.",
        "vae.per_channel_statistics.": "decoder.per_channel_statistics.",
        "per_channel_statistics.": "decoder.per_channel_statistics.",
    }
    converted: dict[str, object] = {}
    for key in state_dict:
        for src, dst in prefixes.items():
            if key.startswith(src):
                suffix = key.removeprefix(src)
                if "per_channel_statistics." in dst and suffix not in {"mean-of-means", "std-of-means"}:
                    break
                converted[f"{dst}{suffix}"] = key if lazy else state_dict[key]
                break
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        return TransformedStateDict(state_dict, converted)
    return converted


def convert_ltx25_conv_video_encoder_state_dict(
    state_dict: Mapping[str, object],
    *,
    lazy: bool = False,
) -> Mapping[str, object]:
    """Map LTX-2.5 split Video VAE encoder keys into ``LTXVideoEncoderModule``."""
    prefixes = {
        "vae.encoder.": "encoder.",
        "encoder.": "encoder.",
        "vae.per_channel_statistics.": "encoder.per_channel_statistics.",
        "per_channel_statistics.": "encoder.per_channel_statistics.",
    }
    converted: dict[str, object] = {}
    for key in state_dict:
        for src, dst in prefixes.items():
            if key.startswith(src):
                suffix = key.removeprefix(src)
                if "per_channel_statistics." in dst and suffix not in {"mean-of-means", "std-of-means"}:
                    break
                converted[f"{dst}{suffix}"] = key if lazy else state_dict[key]
                break
    if lazy:
        from worldfoundry.core.model_loading.checkpoints.transforms import TransformedStateDict

        return TransformedStateDict(state_dict, converted)
    return converted


def accelerate_ltx25_video_vae(
    module: nn.Module,
    config: LTX25AccelerationConfig,
) -> dict[str, Any]:
    """Enable ``channels_last_3d`` and optional ``torch.compile`` on an LTX-2.5 Video VAE."""
    target = getattr(module, "decoder", getattr(module, "encoder", module))
    conv3d_converted = 0
    compiled = False
    if config.vae_channels_last_3d:
        enable_cl = getattr(target, "enable_channels_last_3d", None)
        if callable(enable_cl):
            conv3d_converted = int(enable_cl())
        else:
            from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

            cl_report = convert_convolution_weight_layouts(target, conv3d=True)
            conv3d_converted = int(cl_report.conv3d_converted)
    if config.vae_compile:
        enable_comp = getattr(target, "enable_compile", None)
        if callable(enable_comp):
            param = next(target.parameters(), None)
            if param is not None and param.is_cuda:
                compiled = bool(
                    enable_comp(
                        mode=config.vae_compile_mode,
                        dynamic=(True if config.vae_tiled else None),
                    )
                )
    return {
        "module_type": type(target).__name__,
        "channels_last_3d_conv3d": conv3d_converted,
        "compiled": compiled,
        "compile_mode": config.vae_compile_mode if compiled else None,
        "tiled": bool(config.vae_tiled),
    }


__all__ = [
    "LTX25AcceleratedExecutor",
    "LTX25AccelerationConfig",
    "LTX25CFGParallelRunner",
    "LTX25FirstBlockCache",
    "LTX25FirstBlockCacheConfig",
    "LTX25ParallelConfig",
    "LTX25UlyssesAttentionAdapter",
    "accelerate_ltx25_transformer",
    "accelerate_ltx25_video_vae",
    "coerce_ltx25_acceleration_config",
    "convert_ltx25_conv_video_decoder_state_dict",
    "convert_ltx25_conv_video_encoder_state_dict",
    "gather_ltx25_video_hidden",
    "ltx25_sol_profile",
    "shard_ltx25_video_args",
]
