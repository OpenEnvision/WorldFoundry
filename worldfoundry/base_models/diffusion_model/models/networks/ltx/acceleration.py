# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""SOL-Engine inference acceleration controllers for native LTX / LTX-2.3 DiT models.

Provides two opt-in, graph-compatible, auditable runtime accelerations:

1. :class:`LTXTokenPruningConfig` + :func:`select_ltx_salient_tokens` — Stage-2
   interior video token pruning across middle transformer blocks
   ``[start_block, end_block)`` with exact residual preservation for dropped
   tokens, untouched dense edge blocks, and strict protection of I2V
   conditioning / keyframe tokens (``denoise_mask < 1`` or ``keyframes_mask``).
2. :class:`LTXFirstBlockCacheConfig` + :class:`LTXFirstBlockCache` — block-0
   probe cross-step caching (``FBCache``) that runs block 0 on every step and
   reuses the block ``1..N-1`` video/audio residual whenever block 0's relative
   L1 drift stays below threshold, guarded by dense first/last step boundaries,
   cumulative drift against the last dense probe, consecutive-skip caps, and
   piecewise CUDA Graph compatibility.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import torch

from worldfoundry.core.acceleration.cache import BlockCacheEvent

_VALID_PRUNE_METHODS = frozenset({"feat_norm", "delta_norm", "feat_l1", "feat_var", "uniform"})
_VALID_RESTORE_MODES = frozenset({"nearest_delta", "mean_delta", "zero"})


def _validate_int_option(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _validate_float_option(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
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
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed


# ──────────────────────────────────────────────────────────────────────────
# Interior Video Token Pruning (Stage-2 middle blocks + residual scatter)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LTXTokenPruningConfig:
    """Opt-in interior video token pruning policy for LTX transformer stacks.

    Retains ``round(T * keep_ratio)`` salient video tokens across middle blocks
    ``[start_block, end_block)`` while keeping front/tail dense edge blocks and
    all audio tokens untouched. Conditioning tokens (I2V ``denoise_mask < 1``
    or ``keyframes_mask > 0``) are strictly preserved whenever
    ``preserve_conditioning_tokens=True``.
    """

    keep_ratio: float = 0.75
    method: str = "feat_norm"
    restore_mode: str = "nearest_delta"
    stratify_group_size: int | None = None
    start_block: int = 2
    end_block: int = -2
    min_tokens: int = 4096
    dense_first_steps: int = 0
    dense_last_steps: int = 0
    min_steps_to_enable: int = 5
    allow_distilled_pruning: bool = False
    preserve_conditioning_tokens: bool = True

    def __post_init__(self) -> None:
        _validate_float_option(
            self.keep_ratio,
            name="token_pruning keep_ratio",
            minimum=0.0,
            maximum=1.0,
            strictly_positive=True,
        )
        if not isinstance(self.method, str) or self.method.strip().casefold() not in _VALID_PRUNE_METHODS:
            allowed = ", ".join(sorted(_VALID_PRUNE_METHODS))
            raise ValueError(f"token_pruning method must be one of {{{allowed}}}, got {self.method!r}")
        if not isinstance(self.restore_mode, str) or self.restore_mode.strip().casefold() not in _VALID_RESTORE_MODES:
            allowed_r = ", ".join(sorted(_VALID_RESTORE_MODES))
            raise ValueError(f"token_pruning restore_mode must be one of {{{allowed_r}}}, got {self.restore_mode!r}")
        if self.stratify_group_size is not None:
            _validate_int_option(self.stratify_group_size, name="token_pruning stratify_group_size", minimum=2)
        _validate_int_option(self.start_block, name="token_pruning start_block", minimum=0)
        _validate_int_option(self.end_block, name="token_pruning end_block")
        _validate_int_option(self.min_tokens, name="token_pruning min_tokens", minimum=0)
        _validate_int_option(self.dense_first_steps, name="token_pruning dense_first_steps", minimum=0)
        _validate_int_option(self.dense_last_steps, name="token_pruning dense_last_steps", minimum=0)
        _validate_int_option(self.min_steps_to_enable, name="token_pruning min_steps_to_enable", minimum=1)
        if not isinstance(self.allow_distilled_pruning, bool):
            raise TypeError(
                f"token_pruning allow_distilled_pruning must be a bool, got {self.allow_distilled_pruning!r}"
            )
        if not isinstance(self.preserve_conditioning_tokens, bool):
            raise TypeError(
                f"token_pruning preserve_conditioning_tokens must be a bool, got {self.preserve_conditioning_tokens!r}"
            )

    @property
    def normalized_method(self) -> str:
        return self.method.strip().casefold()

    @property
    def normalized_restore_mode(self) -> str:
        return self.restore_mode.strip().casefold()

    def is_step_eligible(self, step_index: int, total_steps: int | None) -> tuple[bool, str]:
        """Return ``(eligible, reason)`` for one denoising step, guarding short distilled schedules.

        Dropping spatial tokens during short-step distilled refinement (e.g., LTX-2.3
        3-step Stage-2 upscaling) leaves un-refined latent patches that the VideoVAE
        decodes as visible color blotches/smudges. By default (``allow_distilled_pruning=False``),
        schedules with ``total_steps < min_steps_to_enable`` stay dense.
        """
        if (
            total_steps is not None
            and int(total_steps) < int(self.min_steps_to_enable)
            and not self.allow_distilled_pruning
        ):
            return False, "distilled_short_step_guard"
        if int(step_index) < int(self.dense_first_steps):
            return False, "dense_first_step"
        if total_steps is not None and int(step_index) >= int(total_steps) - int(self.dense_last_steps):
            return False, "dense_last_step"
        return True, "eligible"

    def resolve_span(self, num_blocks: int) -> tuple[int, int]:
        """Resolve ``[start_block, end_block)`` against ``num_blocks`` and fail closed if invalid."""
        if num_blocks < 2:
            raise ValueError(f"token_pruning requires at least 2 transformer blocks, got {num_blocks}")
        resolved_end = num_blocks + self.end_block if self.end_block <= 0 else self.end_block
        if not (0 <= self.start_block < resolved_end <= num_blocks):
            raise ValueError(
                f"token_pruning block span [{self.start_block}, {resolved_end}) is invalid for {num_blocks} blocks"
            )
        return self.start_block, resolved_end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_distilled_pruning_guard(
    tp_cfg: LTXTokenPruningConfig,
    tp_rt: dict[str, Any] | None,
    *,
    total_steps: int | None,
) -> None:
    """Increment short-step guard telemetry and emit a one-time quality-protection warning."""
    warned = False
    if tp_rt is not None and not torch.compiler.is_compiling():
        tp_rt["distilled_guard_skips"] = int(tp_rt.get("distilled_guard_skips", 0)) + 1
        warned = bool(tp_rt.get("_distilled_guard_warned", False))
        tp_rt["_distilled_guard_warned"] = True
    if not warned:
        warnings.warn(
            "LTX token pruning skipped by distilled short-step safety guard "
            f"(total_steps={total_steps} < min_steps_to_enable={tp_cfg.min_steps_to_enable}): "
            "dropping spatial tokens during short distilled Stage-2 refinement leaves un-refined "
            "latent patches that VideoVAE decodes as visible color blotches. "
            "Pass allow_distilled_pruning=True only for explicitly visually-audited experiments.",
            UserWarning,
            stacklevel=3,
        )


def select_ltx_salient_tokens(
    hidden_states: torch.Tensor,
    keep_tokens: int,
    *,
    method: str = "feat_norm",
    entry_hidden: torch.Tensor | None = None,
    protected_mask: torch.Tensor | None = None,
    stratify_group_size: int | None = None,
) -> torch.Tensor:
    """Return sorted ascending 1-D indices of ``keep_tokens`` salient video tokens.

    Pure GPU tensor operations with static output shape ``(keep_tokens,)`` and
    zero host synchronization so selection captures cleanly inside CUDA Graphs.
    When ``protected_mask`` (shape ``(T,)``, bool) is provided, every protected
    token receives ``+inf`` priority so conditioning/keyframe tokens are never
    dropped as long as ``protected_mask.sum() <= keep_tokens``.
    When ``stratify_group_size`` is set (and no ``protected_mask`` is present),
    performs local top-k inside every contiguous group so no spatial/temporal
    region can ever be dropped entirely.
    """
    total_tokens = int(hidden_states.shape[1])
    if keep_tokens <= 0 or keep_tokens > total_tokens:
        raise ValueError(f"keep_tokens must be in [1, {total_tokens}], got {keep_tokens}")
    device = hidden_states.device
    if keep_tokens == total_tokens:
        return torch.arange(total_tokens, device=device, dtype=torch.long)

    normalized = method.strip().casefold()
    if normalized == "uniform":
        base = ((torch.arange(keep_tokens, device=device, dtype=torch.long) * total_tokens) // keep_tokens).clamp_(
            max=total_tokens - 1
        )
        if protected_mask is None:
            return base
        scores = torch.zeros(total_tokens, device=device, dtype=torch.float32)
        scores.index_fill_(0, base, 1.0)
        scores = torch.where(protected_mask.to(device=device), torch.full_like(scores, 2.0), scores)
        return torch.sort(torch.topk(scores, keep_tokens, largest=True).indices).values

    source = (
        hidden_states - entry_hidden
        if normalized == "delta_norm" and entry_hidden is not None and entry_hidden.shape == hidden_states.shape
        else hidden_states
    )
    source_f = source.float()
    if normalized in {"feat_norm", "delta_norm"}:
        scores = source_f.pow(2).sum(dim=-1)
    elif normalized == "feat_l1":
        scores = source_f.abs().sum(dim=-1)
    elif normalized == "feat_var":
        scores = source_f.var(dim=-1, unbiased=False)
    else:
        raise ValueError(f"unsupported LTX token pruning method: {method!r}")

    if scores.ndim > 1:
        scores = scores.mean(dim=0)
    # Sanitize NaN / +inf so corrupted unprotected activations can never outrank +inf protected tokens.
    scores = torch.nan_to_num(
        scores,
        nan=float("-inf"),
        posinf=torch.finfo(scores.dtype).max,
        neginf=float("-inf"),
    )
    if (
        stratify_group_size is not None
        and protected_mask is None
        and stratify_group_size > 1
        and total_tokens % stratify_group_size == 0
    ):
        num_groups = total_tokens // stratify_group_size
        k_per_group = keep_tokens // num_groups
        if 0 < k_per_group < stratify_group_size and k_per_group * num_groups == keep_tokens:
            grouped = scores.view(num_groups, stratify_group_size)
            local_top = torch.topk(grouped, k_per_group, dim=-1, largest=True).indices
            offsets = (torch.arange(num_groups, device=device, dtype=torch.long) * stratify_group_size).unsqueeze(1)
            return torch.sort((local_top + offsets).reshape(-1)).values
    if protected_mask is not None:
        scores = torch.where(protected_mask.to(device=device), torch.full_like(scores, float("inf")), scores)
    return torch.sort(torch.topk(scores, keep_tokens, largest=True).indices).values


def restore_ltx_pruned_tokens(
    pre_hidden: torch.Tensor,
    mid_in_hidden: torch.Tensor,
    mid_out_hidden: torch.Tensor,
    keep_idx: torch.Tensor,
    *,
    restore_mode: str = "nearest_delta",
) -> torch.Tensor:
    """Scatter processed tokens back and restore dropped tokens with graph-safe residual delta propagation.

    Kept positions always receive ``mid_out_hidden`` verbatim. Dropped positions keep
    their exact front-block representation ``pre_hidden`` plus either:

    - ``"nearest_delta"``: the middle-block residual delta ``mid_out - mid_in`` of the
      nearest kept token in sequence order (spatially local inside 3D tiles),
    - ``"mean_delta"``: the mean middle-block residual delta across kept tokens, or
    - ``"zero"``: zero middle-block delta (identity bypass).
    """
    normalized = restore_mode.strip().casefold()
    mid_out_cast = mid_out_hidden if mid_out_hidden.dtype == pre_hidden.dtype else mid_out_hidden.to(pre_hidden.dtype)
    if normalized == "zero":
        out = pre_hidden.clone()
        out.index_copy_(1, keep_idx, mid_out_cast)
        return out
    mid_delta = mid_out_hidden - mid_in_hidden
    if mid_delta.dtype != pre_hidden.dtype:
        mid_delta = mid_delta.to(pre_hidden.dtype)
    if normalized == "mean_delta":
        out = pre_hidden + mid_delta.mean(dim=1, keepdim=True)
        out.index_copy_(1, keep_idx, mid_out_cast)
        return out
    if normalized == "nearest_delta":
        total_tokens = int(pre_hidden.shape[1])
        kept_count = int(keep_idx.shape[0])
        all_idx = torch.arange(total_tokens, device=keep_idx.device, dtype=keep_idx.dtype)
        right = torch.searchsorted(keep_idx, all_idx).clamp(max=kept_count - 1)
        left = (right - 1).clamp(min=0)
        dist_left = (all_idx - keep_idx.index_select(0, left)).abs()
        dist_right = (keep_idx.index_select(0, right) - all_idx).abs()
        nearest_pos = torch.where(dist_right < dist_left, right, left)
        out = pre_hidden + mid_delta.index_select(1, nearest_pos)
        out.index_copy_(1, keep_idx, mid_out_cast)
        return out
    raise ValueError(f"unsupported LTX token pruning restore_mode: {restore_mode!r}")


def derive_ltx_protected_token_mask(
    timesteps: torch.Tensor | None,
    keyframes_mask: torch.Tensor | None,
    token_count: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Derive a ``(T,)`` bool mask of I2V conditioning / keyframe tokens without host syncs."""
    mask: torch.Tensor | None = None
    if timesteps is not None and timesteps.ndim >= 2 and int(timesteps.shape[1]) == token_count:
        ts = timesteps.detach().float()
        max_ts = ts.amax(dim=1, keepdim=True)
        # Conditioning tokens carry strictly smaller effective sigma (e.g. denoise_mask == 0).
        cond_mask = (ts < (max_ts - 1e-6)).movedim(1, 0).reshape(token_count, -1).any(dim=1)
        mask = cond_mask.to(device=device)
    if keyframes_mask is not None and keyframes_mask.ndim >= 2 and int(keyframes_mask.shape[1]) == token_count:
        kf_mask = (keyframes_mask.detach() > 0).movedim(1, 0).reshape(token_count, -1).any(dim=1).to(device=device)
        mask = kf_mask if mask is None else (mask | kf_mask)
    return mask


# ──────────────────────────────────────────────────────────────────────────
# FirstBlockCache (FBCache) — Block-0 probe + Block 1..N-1 residual reuse
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LTXFirstBlockCacheConfig:
    """Calibrated FirstBlockCache policy for LTX / LTX-2.3 DiT transformers.

    Runs transformer block 0 on every step and reuses the residual of blocks
    ``1..N-1`` whenever block 0's video residual relative L1 drift stays below
    ``residual_diff_threshold``. Dense boundary steps, cumulative drift tracking
    against the last dense probe, and ``max_consecutive_skips`` prevent scene
    collapse across denoising steps.
    """

    residual_diff_threshold: float = 0.08
    audio_diff_threshold: float | None = None
    dense_first: int = 1
    dense_last: int = 1
    max_consecutive_skips: int = 2
    downsample_factor: int = 4
    cumulative: bool = True
    min_steps_to_enable: int = 4
    max_tokens: int | None = 4096
    eps: float = 1e-6

    def __post_init__(self) -> None:
        _validate_float_option(self.residual_diff_threshold, name="fbcache residual_diff_threshold", minimum=0.0)
        if self.audio_diff_threshold is not None:
            _validate_float_option(self.audio_diff_threshold, name="fbcache audio_diff_threshold", minimum=0.0)
        _validate_int_option(self.dense_first, name="fbcache dense_first", minimum=1)
        _validate_int_option(self.dense_last, name="fbcache dense_last", minimum=0)
        _validate_int_option(self.max_consecutive_skips, name="fbcache max_consecutive_skips", minimum=0)
        _validate_int_option(self.downsample_factor, name="fbcache downsample_factor", minimum=1)
        if not isinstance(self.cumulative, bool):
            raise TypeError(f"fbcache cumulative must be a bool, got {self.cumulative!r}")
        _validate_int_option(self.min_steps_to_enable, name="fbcache min_steps_to_enable", minimum=2)
        if self.max_tokens is not None:
            _validate_int_option(self.max_tokens, name="fbcache max_tokens", minimum=1)
        _validate_float_option(self.eps, name="fbcache eps", minimum=0.0, strictly_positive=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TensorSig = tuple[tuple[int, ...], str, int | None, torch.dtype]


def _tensor_sig(tensor: torch.Tensor | None) -> _TensorSig | None:
    if tensor is None:
        return None
    return (tuple(int(dim) for dim in tensor.shape), tensor.device.type, tensor.device.index, tensor.dtype)


def _relative_l1_probe(current: torch.Tensor, reference: torch.Tensor, eps: float) -> float:
    cur = current.detach().float()
    ref = reference.to(device=cur.device).float()
    num = (cur - ref).abs().mean()
    den = ref.abs().mean().clamp_min(eps)
    return float((num / den).item())


class LTXFirstBlockCache:
    """Request- and CFG-branch-scoped FirstBlockCache controller for LTX DiTs."""

    algorithm = "firstblock"

    def __init__(
        self,
        config: LTXFirstBlockCacheConfig,
        *,
        branch: str = "positive",
        request_id: str | None = None,
        request_epoch: int = 0,
    ) -> None:
        self.config = config
        self.branch = str(branch)
        self.request_id = request_id
        self.request_epoch = int(request_epoch)
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
        self._video_sig: _TensorSig | None = None
        self._audio_sig: _TensorSig | None = None
        self._last_layout_mode: tuple[bool, bool] | None = None
        self._consecutive_skips: int = 0

    def clear_state(self) -> None:
        """Clear stored probe/residual tensors for a new denoising stage without erasing request telemetry."""
        self._clear_state()

    def reset(self) -> None:
        """Clear all stored residuals/probes and reset telemetry counters."""
        self._clear_state()
        self.events.clear()
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    @property
    def cached_video_residual(self) -> torch.Tensor | None:
        return self._cached_video_residual

    @property
    def cached_audio_residual(self) -> torch.Tensor | None:
        return self._cached_audio_residual

    def _sample_probe(self, residual: torch.Tensor) -> torch.Tensor:
        detached = residual.detach()
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
        force_dense: bool = False,
        tiled: bool = False,
    ) -> tuple[bool, str, float | None, torch.Tensor | None, torch.Tensor | None]:
        """Decide whether blocks ``1..num_blocks-1`` can reuse their cached residual."""
        if num_blocks < 2:
            return False, "too-few-blocks", None, None, None
        if torch.is_grad_enabled():
            return False, "autograd", None, None, None
        if step == 0:
            self._clear_state()

        v_probe = (
            self._sample_probe(block0_video_out - block0_video_in)
            if block0_video_in is not None
            and block0_video_out is not None
            and block0_video_in.numel() > 0
            and block0_video_out.numel() > 0
            else None
        )
        a_probe = (
            self._sample_probe(block0_audio_out - block0_audio_in)
            if block0_audio_in is not None
            and block0_audio_out is not None
            and block0_audio_in.numel() > 0
            and block0_audio_out.numel() > 0
            else None
        )

        if v_probe is None and a_probe is None:
            return False, "empty-probe", None, None, None
        if total_steps is not None and int(total_steps) < self.config.min_steps_to_enable:
            return False, "short-schedule", None, v_probe, a_probe
        if (
            self.config.max_tokens is not None
            and v_probe is not None
            and block0_video_out is not None
            and int(block0_video_out.shape[1]) > self.config.max_tokens
        ):
            return False, "max-tokens", None, v_probe, a_probe
        if step == 0:
            return False, "seed", None, v_probe, a_probe
        if self._is_dense_boundary(step, total_steps):
            return False, "dense-boundary", None, v_probe, a_probe

        layout_mode = (bool(force_dense), bool(tiled))
        if self._last_layout_mode is not None and self._last_layout_mode != layout_mode:
            return False, "layout-transition", None, v_probe, a_probe
        if self._consecutive_skips >= self.config.max_consecutive_skips:
            return False, "max-consecutive-skips", None, v_probe, a_probe

        cur_v_sig = _tensor_sig(block0_video_out) if v_probe is not None else None
        cur_a_sig = _tensor_sig(block0_audio_out) if a_probe is not None else None
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
            metric = _relative_l1_probe(v_probe, self._ref_video_probe, self.config.eps)
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

    def record_hit(
        self,
        *,
        step: int,
        num_blocks: int,
        reason: str,
        metric: float | None,
        v_probe: torch.Tensor | None,
        a_probe: torch.Tensor | None,
    ) -> None:
        """Record a cache hit that skipped blocks ``1..num_blocks-1``."""
        self._consecutive_skips += 1
        if not self.config.cumulative:
            if v_probe is not None:
                self._ref_video_probe = v_probe.detach().clone()
            if a_probe is not None:
                self._ref_audio_probe = a_probe.detach().clone()
        skipped = tuple(range(1, num_blocks))
        event = BlockCacheEvent(
            algorithm=self.algorithm,
            step=int(step),
            hit=True,
            reason=reason,
            dense_blocks=(0,),
            skipped_blocks=skipped,
            accumulated_change=metric,
        )
        self.events.append(event)
        self.dense_block_calls += event.dense_block_calls
        self.skipped_block_calls += event.skipped_block_calls
        self.hits += 1

    def record_miss(
        self,
        *,
        step: int,
        num_blocks: int,
        reason: str,
        metric: float | None,
        v_probe: torch.Tensor | None,
        a_probe: torch.Tensor | None,
        video_tail_base: torch.Tensor | None,
        video_tail_out: torch.Tensor | None,
        audio_tail_base: torch.Tensor | None,
        audio_tail_out: torch.Tensor | None,
        force_dense: bool = False,
        tiled: bool = False,
    ) -> None:
        """Record a dense miss and update reference probes and tail residuals."""
        self._consecutive_skips = 0
        if reason != "autograd":
            if v_probe is not None and video_tail_base is not None and video_tail_out is not None:
                self._ref_video_probe = v_probe.detach().clone()
                self._cached_video_residual = (video_tail_out - video_tail_base).detach()
                self._video_sig = _tensor_sig(video_tail_base)
            else:
                self._ref_video_probe = None
                self._cached_video_residual = None
                self._video_sig = None
            if a_probe is not None and audio_tail_base is not None and audio_tail_out is not None:
                self._ref_audio_probe = a_probe.detach().clone()
                self._cached_audio_residual = (audio_tail_out - audio_tail_base).detach()
                self._audio_sig = _tensor_sig(audio_tail_base)
            else:
                self._ref_audio_probe = None
                self._cached_audio_residual = None
                self._audio_sig = None
            self._last_layout_mode = (bool(force_dense), bool(tiled))

        dense = tuple(range(num_blocks))
        event = BlockCacheEvent(
            algorithm=self.algorithm,
            step=int(step),
            hit=False,
            reason=reason,
            dense_blocks=dense,
            skipped_blocks=(),
            accumulated_change=metric,
        )
        self.events.append(event)
        self.dense_block_calls += event.dense_block_calls

    def receipt(self) -> dict[str, Any]:
        """Return JSON-serializable aggregate and per-step FBCache telemetry."""
        payload: dict[str, Any] = {
            "algorithm": self.algorithm,
            "config": self.config.to_dict(),
            "branch": self.branch,
            "events": len(self.events),
            "hits": self.hits,
            "misses": len(self.events) - self.hits,
            "hit_rate": self.hits / len(self.events) if self.events else 0.0,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "event_receipts": [event.receipt() for event in self.events],
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
            payload["request_epoch"] = self.request_epoch
        return payload


__all__ = [
    "LTXFirstBlockCache",
    "LTXFirstBlockCacheConfig",
    "LTXTokenPruningConfig",
    "derive_ltx_protected_token_mask",
    "record_distilled_pruning_guard",
    "restore_ltx_pruned_tokens",
    "select_ltx_salient_tokens",
]
