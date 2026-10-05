"""Model-agnostic cross-step caches for diffusion inference.

Responsibility: decide, per denoising step (or per DiT block), whether
to recompute a residual or reuse / extrapolate the last dense result.
Policies range from a fixed skip set to TeaCache / MagCache / AdaCache /
TaylorSeer and LightX2V-style block feature caches.

This module is not a model, a scheduler, or a CUDA Graph. It never
launches the transformer — callers pass ``compute`` / ``run_block``.
Autograd always forces a dense path so cached residuals cannot leak
into a training graph. One cache instance belongs to one request and
one CFG branch.

Public surface:
- :class:`FixedStepCache` / :class:`AdaptiveResidualCache`
- :class:`TeaCacheResidualCache` / :class:`MagCacheResidualCache` /
  :class:`AdaCacheResidualCache` / :class:`TaylorSeerResidualCache` /
  :class:`CustomTaylorResidualCache`
- :class:`FirstBlockFeatureCache` / :class:`DualBlockFeatureCache` /
  :class:`DynamicBlockFeatureCache` / :class:`BlockTaylorSeerCache`
- :class:`CacheEvent` / :class:`BlockCacheEvent` — telemetry receipts
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────
# Tree helpers — caches store detached tensors so reuse cannot grow the graph
# ──────────────────────────────────────────────────────────────────────────


def _detach(value: T) -> T:
    """Strip ``requires_grad`` from nested tensor containers before storing a hit."""
    if isinstance(value, torch.Tensor):
        return value.detach()  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)  # type: ignore[return-value]
    if isinstance(value, list):
        return [_detach(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}  # type: ignore[return-value]
    return value


def _tree_sub(left: T, right: T) -> T:
    """Subtract matching tensor trees; refuse mixed layouts so a hit cannot silently skip."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left - right  # type: ignore[return-value]
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_tree_sub(a, b) for a, b in zip(left, right, strict=True))  # type: ignore[return-value]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [_tree_sub(a, b) for a, b in zip(left, right, strict=True)]  # type: ignore[return-value]
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return {key: _tree_sub(left[key], right[key]) for key in left}  # type: ignore[return-value]
    raise TypeError("cached output trees must have matching tensor containers")


def _tree_add_scaled(value: T, delta: T, scale: float) -> T:
    """First-order extrapolation: ``value + scale * delta`` on a matching tree."""
    if isinstance(value, torch.Tensor) and isinstance(delta, torch.Tensor):
        return value + delta * scale  # type: ignore[return-value]
    if isinstance(value, tuple) and isinstance(delta, tuple) and len(value) == len(delta):
        return tuple(_tree_add_scaled(a, b, scale) for a, b in zip(value, delta, strict=True))  # type: ignore[return-value]
    if isinstance(value, list) and isinstance(delta, list) and len(value) == len(delta):
        return [_tree_add_scaled(a, b, scale) for a, b in zip(value, delta, strict=True)]  # type: ignore[return-value]
    if isinstance(value, dict) and isinstance(delta, dict) and value.keys() == delta.keys():
        return {key: _tree_add_scaled(value[key], delta[key], scale) for key in value}  # type: ignore[return-value]
    raise TypeError("cached output trees must have matching tensor containers")


def _contains_grad_tensor(value: object) -> bool:
    """True if any stored tensor still requires grad — a cache-store invariant check."""
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, (tuple, list)):
        return any(_contains_grad_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_grad_tensor(item) for item in value.values())
    return False


# ──────────────────────────────────────────────────────────────────────────
# Telemetry — one receipt per step so quality diffs can name the policy
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CacheEvent:
    """One cache decision, suitable for runtime telemetry."""

    step: int
    hit: bool
    reason: str
    accumulated_change: float | None = None


@dataclass(frozen=True, slots=True)
class BlockCacheEvent:
    """Auditable block-level cache decision for one denoising step."""

    algorithm: str
    step: int
    hit: bool
    reason: str
    dense_blocks: tuple[int, ...]
    skipped_blocks: tuple[int, ...]
    accumulated_change: float | None = None

    @property
    def dense_block_calls(self) -> int:
        """How many transformer blocks actually ran on this step."""
        return len(self.dense_blocks)

    @property
    def skipped_block_calls(self) -> int:
        """How many blocks reused a cached residual on this step."""
        return len(self.skipped_blocks)

    def receipt(self) -> dict[str, object]:
        """Return a JSON-serializable decision receipt."""

        return {
            "algorithm": self.algorithm,
            "step": self.step,
            "hit": self.hit,
            "reason": self.reason,
            "dense_blocks": list(self.dense_blocks),
            "skipped_blocks": list(self.skipped_blocks),
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "relative_l1": self.accumulated_change,
        }


# ──────────────────────────────────────────────────────────────────────────
# Fixed-step — explicit skip set; first/last steps stay dense by policy
# ──────────────────────────────────────────────────────────────────────────


class FixedStepCache(Generic[T]):
    """Reuse a previous denoiser output on an explicit set of steps.

    ``delta_scale=0`` replays the last dense output. A non-zero value performs
    first-order extrapolation from the two latest dense outputs.
    """

    def __init__(
        self,
        skip_steps: Iterable[int] = (),
        *,
        delta_scale: float = 0.0,
        dense_first: int = 1,
        dense_last: int = 1,
        total_steps: int | None = None,
    ) -> None:
        """Reject negative skip/boundary counts; ``total_steps`` is optional until ``run``."""
        self.skip_steps = frozenset(int(step) for step in skip_steps)
        if any(step < 0 for step in self.skip_steps):
            raise ValueError("skip_steps must be non-negative")
        if dense_first < 0 or dense_last < 0:
            raise ValueError("dense boundaries must be non-negative")
        if total_steps is not None and total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.delta_scale = float(delta_scale)
        self.dense_first = int(dense_first)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self._last: T | None = None
        self._delta: T | None = None
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Drop stored outputs so the next ``run`` is a seed, not a stale hit."""
        self._last = None
        self._delta = None
        self.events.clear()

    def _is_boundary(self, step: int, total_steps: int | None) -> bool:
        """True for the configured first/last dense windows (quality-critical steps)."""
        if step < self.dense_first:
            return True
        count = self.total_steps if total_steps is None else total_steps
        return count is not None and step >= max(count - self.dense_last, 0)

    def run(self, step: int, compute: Callable[[], T], *, total_steps: int | None = None) -> T:
        """Compute or replay one step output."""

        step = int(step)
        skip = step in self.skip_steps
        reason = "scheduled"
        # Seed / boundary / autograd override a listed skip so quality-critical
        # or training steps never replay a residual.
        if self._last is None:
            skip, reason = False, "seed"
        elif self._is_boundary(step, total_steps):
            skip, reason = False, "dense-boundary"
        elif torch.is_grad_enabled():
            skip, reason = False, "autograd"

        if skip:
            self.events.append(CacheEvent(step=step, hit=True, reason=reason))
            if self.delta_scale and self._delta is not None:
                return _tree_add_scaled(self._last, self._delta, self.delta_scale)
            return self._last

        output = compute()
        previous = self._last
        detached = _detach(output)
        if previous is not None:
            self._delta = _detach(_tree_sub(detached, previous))
        self._last = detached
        dense_reason = reason if step in self.skip_steps else "dense"
        self.events.append(CacheEvent(step=step, hit=False, reason=dense_reason))
        return output


# ──────────────────────────────────────────────────────────────────────────
# Adaptive residual — accumulate cheap signal change until a threshold
# ──────────────────────────────────────────────────────────────────────────


class AdaptiveResidualCache:
    """Reuse a model residual while accumulated input change stays small.

    Warm up densely, estimate normalized change from a cheap signal, accumulate
    it across steps, and replay the last dense residual until the threshold or
    consecutive-hit cap is reached.
    """

    def __init__(
        self,
        threshold: float,
        *,
        warmup_steps: int = 1,
        max_consecutive_hits: int = 3,
        dense_last: int = 1,
        total_steps: int | None = None,
        subsample: int = 1,
        eps: float = 1e-6,
    ) -> None:
        """Reject a negative threshold or non-positive subsample (would crash the slicer)."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if warmup_steps < 0 or max_consecutive_hits < 0 or dense_last < 0:
            raise ValueError("cache step limits must be non-negative")
        if subsample < 1:
            raise ValueError("subsample must be positive")
        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.max_consecutive_hits = int(max_consecutive_hits)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self.subsample = int(subsample)
        self.eps = float(eps)
        self._previous_signal: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Clear residual and accumulator so a new clip cannot reuse the last one."""
        self._previous_signal = None
        self._residual = None
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        self.events.clear()

    def _sample(self, signal: torch.Tensor) -> torch.Tensor:
        """Subsample interior spatial/temporal dims; keep batch and channel intact."""
        detached = signal.detach()
        if self.subsample == 1 or detached.ndim < 2:
            return detached
        slices = [slice(None)] * detached.ndim
        for dim in range(1, detached.ndim - 1):
            slices[dim] = slice(None, None, self.subsample)
        return detached[tuple(slices)]

    def _relative_change(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        """Mean |Δ| / mean |prev| in FP32 so mixed-precision noise does not trip the gate."""
        current_fp32 = current.float()
        previous_fp32 = previous.to(device=current.device).float()
        numerator = (current_fp32 - previous_fp32).abs().mean()
        denominator = previous_fp32.abs().mean().clamp_min(self.eps)
        return float((numerator / denominator).item())

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Return a dense or cached residual for one denoising step."""

        if torch.is_grad_enabled():
            residual = compute_residual()
            self.events.append(CacheEvent(int(step), False, "autograd"))
            return residual

        sampled = self._sample(signal)
        count = self.total_steps if total_steps is None else total_steps
        dense_boundary = int(step) < self.warmup_steps or (
            count is not None and int(step) >= max(int(count) - self.dense_last, 0)
        )

        relative_change = None
        if self._previous_signal is not None:
            relative_change = self._relative_change(sampled, self._previous_signal)
            self._accumulated_change += relative_change

        can_hit = (
            not dense_boundary
            and self._residual is not None
            and self._previous_signal is not None
            and self._accumulated_change < self.threshold
            and self._consecutive_hits < self.max_consecutive_hits
        )
        self._previous_signal = sampled.clone()
        if can_hit:
            self._consecutive_hits += 1
            self.events.append(CacheEvent(int(step), True, "below-threshold", self._accumulated_change))
            return self._residual

        residual = compute_residual()
        self._residual = residual.detach()
        self._accumulated_change = 0.0
        self._consecutive_hits = 0
        reason = "dense-boundary" if dense_boundary else "threshold"
        if relative_change is None:
            reason = "seed"
        self.events.append(CacheEvent(int(step), False, reason, relative_change))
        return residual


# ──────────────────────────────────────────────────────────────────────────
# Block feature caches — skip DiT blocks from a cheap residual probe
# ──────────────────────────────────────────────────────────────────────────


def _relative_l1(current: torch.Tensor, previous: torch.Tensor, eps: float) -> float:
    """Return the scalar relative-L1 signal used by feature-cache policies."""

    current_fp32 = current.detach().float()
    previous_fp32 = previous.to(device=current.device).float()
    numerator = (current_fp32 - previous_fp32).abs().mean()
    denominator = previous_fp32.abs().mean().clamp_min(float(eps))
    return float((numerator / denominator).item())


_BlockRunner = Callable[[int, torch.Tensor], torch.Tensor]
_PhaseOutputs = dict[str, torch.Tensor]
_PhaseBlockRunner = Callable[
    [int, torch.Tensor, _PhaseOutputs | None],
    tuple[torch.Tensor, _PhaseOutputs],
]
_TensorSignature = tuple[tuple[int, ...], str, int | None, torch.dtype]


def _tensor_signature(value: torch.Tensor) -> _TensorSignature:
    """Shape/device/dtype key so a CFG or resolution change cannot reuse a residual."""
    return (
        tuple(value.shape),
        value.device.type,
        value.device.index,
        value.dtype,
    )


def _sample_last_dimension(value: torch.Tensor, factor: int) -> torch.Tensor:
    """Cheap channel stride for the probe metric; factor 1 is a no-op."""
    sampled = value.detach()
    if factor > 1:
        sampled = sampled[..., ::factor]
    return sampled


def _validate_positive_integer(value: object, *, owner: str) -> int:
    """Reject bools (subclass of int) and non-positive counts used as block windows."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if value < 1:
        raise ValueError(f"{owner} must be positive")
    return value


def _validate_nonnegative_integer(value: object, *, owner: str) -> int:
    """Reject bools (subclass of int) and negatives; zero is allowed for optional tails."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if value < 0:
        raise ValueError(f"{owner} must be non-negative")
    return value


class _BlockFeatureCache:
    """Shared validation and telemetry for block-selective feature caches."""

    algorithm = "block-cache"

    def __init__(
        self,
        residual_diff_threshold: float,
        *,
        downsample_factor: int = 1,
        dense_first: int = 1,
        dense_last: int = 0,
        eps: float = 1e-6,
    ) -> None:
        """Finite non-negative threshold; ``eps`` must be positive so relative-L1 is defined."""
        if isinstance(residual_diff_threshold, bool):
            raise TypeError("residual_diff_threshold must be a number")
        threshold = float(residual_diff_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("residual_diff_threshold must be finite and non-negative")
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError("block feature-cache eps must be finite and positive")
        self.residual_diff_threshold = threshold
        self.downsample_factor = _validate_positive_integer(
            downsample_factor,
            owner="downsample_factor",
        )
        self.dense_first = _validate_positive_integer(
            dense_first,
            owner="dense_first",
        )
        self.dense_last = _validate_nonnegative_integer(
            dense_last,
            owner="dense_last",
        )
        self.eps = float(eps)
        self.events: list[BlockCacheEvent] = []
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    def _clear_cache_state(self) -> None:
        """Subclass hook: drop probe/residual tensors without touching counters."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear tensors and telemetry so a new request cannot inherit the last clip."""
        self._clear_cache_state()
        self.events.clear()
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    @staticmethod
    def _validate_run(step: int, block_count: int, total_steps: int | None) -> tuple[int, int]:
        """Normalize step/block_count; refuse bools and a non-positive stack depth."""
        if isinstance(step, bool):
            raise TypeError("feature-cache step must be an integer")
        step = int(step)
        if step < 0:
            raise ValueError("feature-cache step must be non-negative")
        block_count = _validate_positive_integer(block_count, owner="block_count")
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError("feature-cache total_steps must be positive")
        return step, block_count

    def _is_dense_boundary(self, step: int, total_steps: int | None) -> bool:
        """True for the first/last dense window; ``dense_last`` needs ``total_steps``."""
        if step < self.dense_first:
            return True
        if self.dense_last == 0:
            return False
        if total_steps is None:
            raise ValueError("dense_last requires feature-cache total_steps")
        return step >= max(int(total_steps) - self.dense_last, 0)

    @staticmethod
    def _run_block_ids(
        hidden: torch.Tensor,
        block_ids: Iterable[int],
        run_block: _BlockRunner,
    ) -> torch.Tensor:
        """Run a contiguous block range; refuse a non-tensor so a hit cannot store junk."""
        for block_id in block_ids:
            hidden = run_block(block_id, hidden)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("block feature-cache callbacks must return a tensor")
        return hidden

    def _record(
        self,
        *,
        step: int,
        reason: str,
        dense_blocks: Iterable[int],
        skipped_blocks: Iterable[int],
        metric: float | None = None,
    ) -> None:
        """Append one receipt and bump lifetime counters; a hit is any skipped block."""
        dense = tuple(dense_blocks)
        skipped = tuple(skipped_blocks)
        hit = bool(skipped)
        event = BlockCacheEvent(
            algorithm=self.algorithm,
            step=step,
            hit=hit,
            reason=reason,
            dense_blocks=dense,
            skipped_blocks=skipped,
            accumulated_change=metric,
        )
        self.events.append(event)
        self.dense_block_calls += event.dense_block_calls
        self.skipped_block_calls += event.skipped_block_calls
        self.hits += int(hit)

    def receipt(self) -> dict[str, object]:
        """Return aggregate counters plus exact per-step block decisions."""

        receipt: dict[str, object] = {
            "algorithm": self.algorithm,
            "residual_diff_threshold": self.residual_diff_threshold,
            "downsample_factor": self.downsample_factor,
            "dense_first": self.dense_first,
            "dense_last": self.dense_last,
            "events": len(self.events),
            "hits": self.hits,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "event_receipts": [event.receipt() for event in self.events],
        }
        branch = getattr(self, "branch", None)
        if branch is not None:
            receipt["branch"] = str(branch)
        request_epoch = getattr(self, "request_epoch", None)
        if request_epoch is not None:
            receipt["request_epoch"] = int(request_epoch)
        request_id = getattr(self, "request_id", None)
        if request_id is not None:
            receipt["request_id"] = str(request_id)
        if hasattr(self, "front_blocks"):
            receipt["front_blocks"] = int(self.front_blocks)
        if hasattr(self, "back_blocks"):
            receipt["back_blocks"] = int(self.back_blocks)
        return receipt


class _ProbeBlockFeatureCache(_BlockFeatureCache):
    """Shared probe and cached-range state for FirstBlock and DualBlock."""

    def __init__(
        self,
        residual_diff_threshold: float,
        *,
        downsample_factor: int = 1,
        dense_first: int = 1,
        dense_last: int = 0,
        eps: float = 1e-6,
    ) -> None:
        """Allocate empty probe/residual slots; a seed step fills them on first ``run``."""
        super().__init__(
            residual_diff_threshold,
            downsample_factor=downsample_factor,
            dense_first=dense_first,
            dense_last=dense_last,
            eps=eps,
        )
        self._previous_probe: torch.Tensor | None = None
        self._cached_residual: torch.Tensor | None = None
        self._probe_signature: _TensorSignature | None = None
        self._residual_base_signature: _TensorSignature | None = None
        self._block_count: int | None = None

    def _clear_cache_state(self) -> None:
        """Drop probe and middle residual so a new clip cannot add the last residual."""
        self._previous_probe = None
        self._cached_residual = None
        self._probe_signature = None
        self._residual_base_signature = None
        self._block_count = None

    def _decision(
        self,
        *,
        step: int,
        probe: torch.Tensor,
        residual_base: torch.Tensor,
        block_count: int,
        dense_boundary: bool,
    ) -> tuple[bool, str, float | None, torch.Tensor]:
        """Hit only when signatures match and relative-L1 stays finite and below threshold."""
        sampled = _sample_last_dimension(probe, self.downsample_factor)
        if step == 0:
            return False, "seed", None, sampled
        if dense_boundary:
            return False, "dense-boundary", None, sampled
        if self._previous_probe is None or self._cached_residual is None:
            return False, "cache-missing", None, sampled
        if (
            self._block_count != block_count
            or self._probe_signature != _tensor_signature(sampled)
            or self._residual_base_signature != _tensor_signature(residual_base)
            or _tensor_signature(self._cached_residual) != _tensor_signature(residual_base)
        ):
            return False, "incompatible-state", None, sampled
        metric = _relative_l1(sampled, self._previous_probe, self.eps)
        if not math.isfinite(metric):
            return False, "non-finite-metric", metric, sampled
        if metric < self.residual_diff_threshold:
            return True, "below-threshold", metric, sampled
        return False, "threshold", metric, sampled

    def _store_probe(self, probe: torch.Tensor, block_count: int) -> None:
        """Clone the probe so a later in-place DiT write cannot mutate the metric."""
        self._previous_probe = probe.detach().clone()
        self._probe_signature = _tensor_signature(probe)
        self._block_count = block_count

    def _store_residual(self, residual: torch.Tensor, residual_base: torch.Tensor) -> None:
        """Keep the residual plus the base signature used to add it back."""
        self._cached_residual = residual.detach()
        self._residual_base_signature = _tensor_signature(residual_base)


# ──────────────────────────────────────────────────────────────────────────
# FirstBlock — block 0 is the probe; later blocks share one residual
# ──────────────────────────────────────────────────────────────────────────


class FirstBlockFeatureCache(_ProbeBlockFeatureCache):
    """Run block 0 as a probe and reuse the residual of all later blocks.

    This mirrors LightX2V FirstBlock semantics while keeping cache ownership
    outside the model: one instance belongs to one request and one CFG branch.
    """

    algorithm = "firstblock"

    def run_blocks(
        self,
        step: int,
        block_input: torch.Tensor,
        run_block: _BlockRunner,
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Always run block 0; replay the rest when the probe residual is still small."""
        step, block_count = self._validate_run(step, block_count, total_steps)
        if block_count < 2:
            raise ValueError("FirstBlock feature cache requires at least 2 blocks")
        all_blocks = tuple(range(block_count))
        if torch.is_grad_enabled():
            output = self._run_block_ids(block_input, all_blocks, run_block)
            self._record(step=step, reason="autograd", dense_blocks=all_blocks, skipped_blocks=())
            return output

        dense_boundary = self._is_dense_boundary(step, total_steps)
        if step == 0:
            self._clear_cache_state()
        after_probe = self._run_block_ids(block_input, (0,), run_block)
        probe = after_probe - block_input
        can_hit, reason, metric, sampled = self._decision(
            step=step,
            probe=probe,
            residual_base=after_probe,
            block_count=block_count,
            dense_boundary=dense_boundary,
        )
        remaining = tuple(range(1, block_count))
        if can_hit:
            assert self._cached_residual is not None
            output = after_probe + self._cached_residual
            dense_blocks = (0,)
            skipped_blocks = remaining
        else:
            output = self._run_block_ids(after_probe, remaining, run_block)
            self._store_residual(output - after_probe, after_probe)
            dense_blocks = all_blocks
            skipped_blocks = ()
        self._store_probe(sampled, block_count)
        self._record(
            step=step,
            reason=reason,
            dense_blocks=dense_blocks,
            skipped_blocks=skipped_blocks,
            metric=metric,
        )
        return output


# ──────────────────────────────────────────────────────────────────────────
# DualBlock — dense front/back; only the middle residual is reusable
# ──────────────────────────────────────────────────────────────────────────


class DualBlockFeatureCache(_ProbeBlockFeatureCache):
    """Run dense front/back partitions and cache only the middle residual."""

    algorithm = "dualblock"

    def __init__(
        self,
        residual_diff_threshold: float,
        *,
        downsample_factor: int = 1,
        dense_first: int = 1,
        dense_last: int = 0,
        front_blocks: int = 5,
        back_blocks: int = 5,
        eps: float = 1e-6,
    ) -> None:
        """Front/back windows must be positive so at least one middle block remains."""
        super().__init__(
            residual_diff_threshold,
            downsample_factor=downsample_factor,
            dense_first=dense_first,
            dense_last=dense_last,
            eps=eps,
        )
        self.front_blocks = _validate_positive_integer(front_blocks, owner="front_blocks")
        self.back_blocks = _validate_positive_integer(back_blocks, owner="back_blocks")

    def run_blocks(
        self,
        step: int,
        block_input: torch.Tensor,
        run_block: _BlockRunner,
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Always run front and back; skip the middle when the front residual is stable."""
        step, block_count = self._validate_run(step, block_count, total_steps)
        minimum = self.front_blocks + self.back_blocks + 1
        if block_count < minimum:
            raise ValueError(
                "DualBlock feature cache requires at least "
                f"{minimum} blocks ({self.front_blocks} front + one middle + "
                f"{self.back_blocks} back)"
            )
        all_blocks = tuple(range(block_count))
        if torch.is_grad_enabled():
            output = self._run_block_ids(block_input, all_blocks, run_block)
            self._record(step=step, reason="autograd", dense_blocks=all_blocks, skipped_blocks=())
            return output

        dense_boundary = self._is_dense_boundary(step, total_steps)
        if step == 0:
            self._clear_cache_state()
        front = tuple(range(self.front_blocks))
        middle = tuple(range(self.front_blocks, block_count - self.back_blocks))
        back = tuple(range(block_count - self.back_blocks, block_count))
        after_front = self._run_block_ids(block_input, front, run_block)
        probe = after_front - block_input
        can_hit, reason, metric, sampled = self._decision(
            step=step,
            probe=probe,
            residual_base=after_front,
            block_count=block_count,
            dense_boundary=dense_boundary,
        )
        if can_hit:
            assert self._cached_residual is not None
            after_middle = after_front + self._cached_residual
            dense_blocks = front + back
            skipped_blocks = middle
        else:
            after_middle = self._run_block_ids(after_front, middle, run_block)
            self._store_residual(after_middle - after_front, after_front)
            dense_blocks = all_blocks
            skipped_blocks = ()
        self._store_probe(sampled, block_count)
        output = self._run_block_ids(after_middle, back, run_block)
        self._record(
            step=step,
            reason=reason,
            dense_blocks=dense_blocks,
            skipped_blocks=skipped_blocks,
            metric=metric,
        )
        return output


# ──────────────────────────────────────────────────────────────────────────
# DynamicBlock — independent hit/miss per block; partial hits are expected
# ──────────────────────────────────────────────────────────────────────────


class DynamicBlockFeatureCache(_BlockFeatureCache):
    """Make an independent dense/reuse decision at every transformer block."""

    algorithm = "dynamicblock"

    def __init__(
        self,
        residual_diff_threshold: float,
        *,
        downsample_factor: int = 1,
        dense_first: int = 1,
        dense_last: int = 0,
        eps: float = 1e-6,
    ) -> None:
        """Per-block maps start empty; a changed ``block_count`` wipes them on ``run``."""
        super().__init__(
            residual_diff_threshold,
            downsample_factor=downsample_factor,
            dense_first=dense_first,
            dense_last=dense_last,
            eps=eps,
        )
        self._block_inputs: dict[int, torch.Tensor] = {}
        self._block_residuals: dict[int, torch.Tensor] = {}
        self._block_signatures: dict[int, _TensorSignature] = {}
        self._block_count: int | None = None

    def _clear_cache_state(self) -> None:
        """Drop every per-block residual; a length change would otherwise add the wrong Δ."""
        self._block_inputs.clear()
        self._block_residuals.clear()
        self._block_signatures.clear()
        self._block_count = None

    def run_blocks(
        self,
        step: int,
        block_input: torch.Tensor,
        run_block: _BlockRunner,
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Decide reuse independently at each block; record ``partial-hit`` when mixed."""
        step, block_count = self._validate_run(step, block_count, total_steps)
        all_blocks = tuple(range(block_count))
        if torch.is_grad_enabled():
            output = self._run_block_ids(block_input, all_blocks, run_block)
            self._record(step=step, reason="autograd", dense_blocks=all_blocks, skipped_blocks=())
            return output

        dense_boundary = self._is_dense_boundary(step, total_steps)
        block_count_changed = self._block_count not in (None, block_count)
        if step == 0 or block_count_changed:
            self._clear_cache_state()
        self._block_count = block_count

        hidden = block_input
        dense_blocks: list[int] = []
        skipped_blocks: list[int] = []
        decision_reasons: set[str] = set()
        metrics: list[float] = []
        for block_id in all_blocks:
            current = hidden
            sampled = _sample_last_dimension(current, self.downsample_factor)
            signature = _tensor_signature(current)
            previous = self._block_inputs.get(block_id)
            cached_residual = self._block_residuals.get(block_id)
            can_hit = False
            if step == 0:
                decision_reasons.add("seed")
            elif dense_boundary:
                decision_reasons.add("dense-boundary")
            elif previous is None or cached_residual is None:
                decision_reasons.add("cache-missing")
            elif (
                self._block_signatures.get(block_id) != signature
                or _tensor_signature(cached_residual) != signature
                or _tensor_signature(previous) != _tensor_signature(sampled)
            ):
                decision_reasons.add("incompatible-state")
            else:
                metric = _relative_l1(sampled, previous, self.eps)
                metrics.append(metric)
                if not math.isfinite(metric):
                    decision_reasons.add("non-finite-metric")
                elif metric < self.residual_diff_threshold:
                    can_hit = True
                else:
                    decision_reasons.add("threshold")

            if can_hit:
                hidden = current + cached_residual
                skipped_blocks.append(block_id)
                residual = cached_residual
            else:
                hidden = run_block(block_id, current)
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("block feature-cache callbacks must return a tensor")
                dense_blocks.append(block_id)
                residual = hidden - current
            self._block_inputs[block_id] = sampled.detach().clone()
            self._block_residuals[block_id] = residual.detach()
            self._block_signatures[block_id] = signature

        if step == 0:
            reason = "seed"
        elif dense_boundary:
            reason = "dense-boundary"
        elif block_count_changed:
            reason = "block-count-changed"
        elif skipped_blocks and dense_blocks:
            reason = "partial-hit"
        elif skipped_blocks:
            reason = "below-threshold"
        elif "non-finite-metric" in decision_reasons:
            reason = "non-finite-metric"
        elif "incompatible-state" in decision_reasons:
            reason = "incompatible-state"
        elif "cache-missing" in decision_reasons:
            reason = "cache-missing"
        else:
            reason = "threshold"
        # Worst-block L1 is the conservative receipt; a mean would hide a miss.
        metric = max(metrics) if metrics else None
        self._record(
            step=step,
            reason=reason,
            dense_blocks=dense_blocks,
            skipped_blocks=skipped_blocks,
            metric=metric,
        )
        return hidden


# ──────────────────────────────────────────────────────────────────────────
# LightX2V Wan TaylorSeer — per-block self/cross/FFN first-order prediction
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _TaylorPhaseState:
    """Detached value/slope pair from the latest dense Taylor step."""

    value: torch.Tensor
    derivative: torch.Tensor | None


class BlockTaylorSeerCache:
    """LightX2V-equivalent Wan per-block first-order Taylor cache.

    The legacy :class:`TaylorSeerResidualCache` predicts one residual for the
    entire transformer stack.  LightX2V's Wan implementation is materially
    different: on dense steps it stores ``self_attn_out``,
    ``cross_attn_out``, and ``ffn_out`` independently for every block.  On a
    cache step those raw outputs are extrapolated, while the model still
    recomputes the current timestep modulation and applies the current
    attention/FFN gates.  ``dense_pattern=(True, False, False, False)`` is the
    pinned reference scheduler.

    One instance belongs to exactly one request and one CFG branch.  The
    denoiser owns that lifecycle; shape/device/dtype checks additionally fail
    closed if an incompatible trajectory reaches this object.
    """

    algorithm = "blocktaylorseer"
    phase_names = ("self_attn_out", "cross_attn_out", "ffn_out")

    def __init__(
        self,
        *,
        dense_pattern: Iterable[bool] = (True, False, False, False),
        dense_last: int = 0,
        total_steps: int | None = None,
    ) -> None:
        pattern = tuple(dense_pattern)
        if not pattern or any(not isinstance(value, bool) for value in pattern):
            raise TypeError("BlockTaylorSeer dense_pattern must be a non-empty bool sequence")
        if not any(pattern):
            raise ValueError("BlockTaylorSeer dense_pattern must contain a dense step")
        self.dense_pattern = pattern
        self.dense_last = _validate_nonnegative_integer(
            dense_last,
            owner="dense_last",
        )
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError("BlockTaylorSeer total_steps must be positive")
        self.total_steps = total_steps
        self._blocks: dict[int, dict[str, _TaylorPhaseState]] = {}
        self._block_count: int | None = None
        self._input_signature: _TensorSignature | None = None
        self._last_dense_step: int | None = None
        self.events: list[BlockCacheEvent] = []
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    def _clear_cache_state(self) -> None:
        """Drop all phase values and slopes without erasing this request's receipts."""

        self._blocks.clear()
        self._block_count = None
        self._input_signature = None
        self._last_dense_step = None

    def reset(self) -> None:
        """Drop phase history and telemetry before a new request."""

        self._clear_cache_state()
        self.events.clear()
        self.dense_block_calls = 0
        self.skipped_block_calls = 0
        self.hits = 0

    @staticmethod
    def _validate_phase_outputs(outputs: object) -> _PhaseOutputs:
        """Require exactly the three raw Wan phase tensors used by LightX2V."""

        if not isinstance(outputs, dict):
            raise TypeError("BlockTaylorSeer phase callback must return a dict")
        expected = set(BlockTaylorSeerCache.phase_names)
        if set(outputs) != expected:
            raise ValueError(
                "BlockTaylorSeer phase callback must return exactly "
                + ", ".join(BlockTaylorSeerCache.phase_names)
            )
        parsed: _PhaseOutputs = {}
        for name in BlockTaylorSeerCache.phase_names:
            value = outputs[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"BlockTaylorSeer {name} must be a tensor")
            parsed[name] = value
        return parsed

    def _state_is_compatible(
        self,
        block_input: torch.Tensor,
        block_count: int,
    ) -> bool:
        """All blocks/phases must exist on the same tensor layout before any skip."""

        if (
            self._last_dense_step is None
            or self._block_count != block_count
            or self._input_signature != _tensor_signature(block_input)
            or set(self._blocks) != set(range(block_count))
        ):
            return False
        expected = set(self.phase_names)
        root_signature = _tensor_signature(block_input)
        for phases in self._blocks.values():
            if set(phases) != expected:
                return False
            for state in phases.values():
                if _tensor_signature(state.value) != root_signature:
                    return False
                if (
                    state.derivative is not None
                    and _tensor_signature(state.derivative) != root_signature
                ):
                    return False
        return True

    def _record(
        self,
        *,
        step: int,
        reason: str,
        dense_blocks: tuple[int, ...],
        skipped_blocks: tuple[int, ...],
        distance: float | None = None,
    ) -> None:
        event = BlockCacheEvent(
            algorithm=self.algorithm,
            step=step,
            hit=bool(skipped_blocks),
            reason=reason,
            dense_blocks=dense_blocks,
            skipped_blocks=skipped_blocks,
            accumulated_change=distance,
        )
        self.events.append(event)
        self.dense_block_calls += event.dense_block_calls
        self.skipped_block_calls += event.skipped_block_calls
        self.hits += int(event.hit)

    def run_phase_blocks(
        self,
        step: int,
        block_input: torch.Tensor,
        run_block: _BlockRunner,
        run_phase_block: _PhaseBlockRunner,
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Run a dense scheduled step or extrapolate every block's three phases."""

        step, block_count = _BlockFeatureCache._validate_run(
            step,
            block_count,
            total_steps,
        )
        all_blocks = tuple(range(block_count))
        if torch.is_grad_enabled():
            hidden = _BlockFeatureCache._run_block_ids(
                block_input,
                all_blocks,
                run_block,
            )
            self._record(
                step=step,
                reason="autograd",
                dense_blocks=all_blocks,
                skipped_blocks=(),
            )
            return hidden

        count = self.total_steps if total_steps is None else total_steps
        if count is not None and int(count) < 1:
            raise ValueError("BlockTaylorSeer total_steps must be positive")
        dense_boundary = self.dense_last > 0 and count is not None and step >= max(
            int(count) - self.dense_last,
            0,
        )
        if self.dense_last > 0 and count is None:
            raise ValueError("BlockTaylorSeer dense_last requires total_steps")

        block_count_changed = self._block_count not in (None, block_count)
        non_monotonic = self._last_dense_step is not None and step <= self._last_dense_step
        if step == 0 or block_count_changed or non_monotonic:
            self._clear_cache_state()
        compatible = self._state_is_compatible(block_input, block_count)
        scheduled_dense = self.dense_pattern[step % len(self.dense_pattern)]
        can_hit = not scheduled_dense and not dense_boundary and compatible

        if can_hit:
            assert self._last_dense_step is not None
            distance = step - self._last_dense_step
            hidden = block_input
            for block_id in all_blocks:
                predicted: _PhaseOutputs = {}
                for name, state in self._blocks[block_id].items():
                    value = state.value
                    if state.derivative is not None:
                        value = value + state.derivative * distance
                    predicted[name] = value
                hidden, _ = run_phase_block(block_id, hidden, predicted)
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("BlockTaylorSeer phase callback must return a tensor hidden state")
            self._record(
                step=step,
                reason="taylor-extrapolation",
                dense_blocks=(),
                skipped_blocks=all_blocks,
                distance=float(distance),
            )
            return hidden

        previous_blocks = self._blocks
        previous_dense_step = self._last_dense_step
        dense_gap = (
            max(step - previous_dense_step, 1)
            if previous_dense_step is not None
            else None
        )
        next_blocks: dict[int, dict[str, _TaylorPhaseState]] = {}
        hidden = block_input
        for block_id in all_blocks:
            hidden, raw_outputs = run_phase_block(block_id, hidden, None)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("BlockTaylorSeer phase callback must return a tensor hidden state")
            outputs = self._validate_phase_outputs(raw_outputs)
            next_phases: dict[str, _TaylorPhaseState] = {}
            for name, output in outputs.items():
                detached = output.detach()
                derivative = None
                previous = previous_blocks.get(block_id, {}).get(name)
                if previous is not None and dense_gap is not None:
                    derivative = (detached - previous.value) / dense_gap
                    derivative = derivative.detach()
                next_phases[name] = _TaylorPhaseState(detached, derivative)
            next_blocks[block_id] = next_phases

        self._blocks = next_blocks
        self._block_count = block_count
        self._input_signature = _tensor_signature(block_input)
        self._last_dense_step = step
        if step == 0:
            reason = "seed"
        elif dense_boundary:
            reason = "dense-boundary"
        elif block_count_changed:
            reason = "block-count-changed"
        elif non_monotonic:
            reason = "non-monotonic-step"
        elif not compatible and not scheduled_dense:
            reason = "incompatible-state"
        else:
            reason = "scheduled-dense"
        self._record(
            step=step,
            reason=reason,
            dense_blocks=all_blocks,
            skipped_blocks=(),
        )
        return hidden

    def receipt(self) -> dict[str, object]:
        """Return request-local schedule, counters, and exact per-step decisions."""

        receipt: dict[str, object] = {
            "algorithm": self.algorithm,
            "prediction_scope": "per-block-self-cross-ffn",
            "taylor_order": 1,
            "dense_pattern": list(self.dense_pattern),
            "dense_last": self.dense_last,
            "events": len(self.events),
            "hits": self.hits,
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
            "event_receipts": [event.receipt() for event in self.events],
        }
        branch = getattr(self, "branch", None)
        if branch is not None:
            receipt["branch"] = str(branch)
        request_epoch = getattr(self, "request_epoch", None)
        if request_epoch is not None:
            receipt["request_epoch"] = int(request_epoch)
        request_id = getattr(self, "request_id", None)
        if request_id is not None:
            receipt["request_id"] = str(request_id)
        return receipt


# ──────────────────────────────────────────────────────────────────────────
# Published residual policies — TeaCache / MagCache / AdaCache / TaylorSeer
# ──────────────────────────────────────────────────────────────────────────


def _polyval(coefficients: tuple[float, ...], value: float) -> float:
    """Evaluate descending-order polynomial coefficients without NumPy."""

    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


class TeaCacheResidualCache:
    """Published TeaCache accumulated polynomial residual-reuse policy.

    Wan TeaCache does not threshold the raw timestep-embedding delta. It first
    rescales relative L1 with a model-calibrated polynomial, accumulates that
    estimate, and recomputes the transformer stack only after the configured
    threshold is crossed. The first ``warmup_steps`` and final ``dense_last``
    steps remain dense, matching the reference implementation.
    """

    algorithm = "teacache"

    def __init__(
        self,
        threshold: float,
        coefficients: Iterable[float],
        *,
        warmup_steps: int = 5,
        dense_last: int = 1,
        total_steps: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Require a non-empty polynomial; empty coeffs would make every step a hit."""
        coefficients = tuple(float(value) for value in coefficients)
        if threshold < 0:
            raise ValueError("TeaCache threshold must be non-negative")
        if not coefficients:
            raise ValueError("TeaCache coefficients must not be empty")
        if warmup_steps < 0 or dense_last < 0:
            raise ValueError("TeaCache dense boundaries must be non-negative")
        self.threshold = float(threshold)
        self.coefficients = coefficients
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self.eps = float(eps)
        self._previous_signal: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._accumulated_change = 0.0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Zero the polynomial accumulator so a new clip cannot inherit the last estimate."""
        self._previous_signal = None
        self._residual = None
        self._accumulated_change = 0.0
        self.events.clear()

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Reuse the residual until the *rescaled* accumulated L1 crosses ``threshold``."""
        step = int(step)
        if torch.is_grad_enabled():
            residual = compute_residual()
            self.events.append(CacheEvent(step, False, "autograd"))
            return residual

        count = self.total_steps if total_steps is None else total_steps
        dense_boundary = step < self.warmup_steps or (
            count is not None and step >= max(int(count) - self.dense_last, 0)
        )
        relative_change = None
        scaled_change = None
        if self._previous_signal is not None:
            relative_change = _relative_l1(signal, self._previous_signal, self.eps)
            scaled_change = _polyval(self.coefficients, relative_change)
            self._accumulated_change += scaled_change
        self._previous_signal = signal.detach().clone()

        can_hit = (
            not dense_boundary
            and self._residual is not None
            and relative_change is not None
            and self._accumulated_change < self.threshold
        )
        if can_hit:
            self.events.append(
                CacheEvent(step, True, "polynomial-below-threshold", self._accumulated_change)
            )
            return self._residual

        residual = compute_residual()
        self._residual = residual.detach()
        self._accumulated_change = 0.0
        if relative_change is None:
            reason = "seed"
        elif dense_boundary:
            reason = "dense-boundary"
        else:
            reason = "polynomial-threshold"
        self.events.append(CacheEvent(step, False, reason, scaled_change))
        return residual


class CustomTaylorResidualCache:
    """Pinned LightX2V ``CustomCaching`` policy for Wan.

    TeaCache's calibrated polynomial decides whether the full Wan stack runs.
    A hit does not replay a constant residual: it first-order extrapolates the
    latest dense whole-stack residual by the number of steps since that dense
    sample.  This is intentionally distinct from both plain TeaCache and the
    fixed-cadence legacy ``taylorseer`` implementation.
    """

    algorithm = "custom"

    def __init__(
        self,
        threshold: float,
        coefficients: Iterable[float],
        *,
        warmup_steps: int = 5,
        dense_last: int = 0,
        total_steps: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        coefficients = tuple(float(value) for value in coefficients)
        if not math.isfinite(float(threshold)) or threshold < 0:
            raise ValueError("Custom cache threshold must be finite and non-negative")
        if not coefficients or any(not math.isfinite(value) for value in coefficients):
            raise ValueError("Custom cache coefficients must be finite and non-empty")
        if warmup_steps < 0 or dense_last < 0:
            raise ValueError("Custom cache dense boundaries must be non-negative")
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError("Custom cache total_steps must be positive")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("Custom cache eps must be finite and positive")
        self.threshold = float(threshold)
        self.coefficients = coefficients
        self.warmup_steps = int(warmup_steps)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self.eps = float(eps)
        self._previous_signal: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._derivative: torch.Tensor | None = None
        self._last_dense_step: int | None = None
        self._accumulated_change = 0.0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Drop Tea decision state and Taylor history for a new request/branch."""

        self._previous_signal = None
        self._residual = None
        self._derivative = None
        self._last_dense_step = None
        self._accumulated_change = 0.0
        self.events.clear()

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Apply TeaCache's gate and Taylor-predict the residual on a real hit."""

        step = int(step)
        if step < 0:
            raise ValueError("Custom cache step must be non-negative")
        if torch.is_grad_enabled():
            residual = compute_residual()
            self.events.append(CacheEvent(step, False, "autograd"))
            return residual

        count = self.total_steps if total_steps is None else total_steps
        if count is not None and int(count) < 1:
            raise ValueError("Custom cache total_steps must be positive")
        dense_boundary = step < self.warmup_steps or (
            count is not None and step >= max(int(count) - self.dense_last, 0)
        )
        relative_change = None
        scaled_change = None
        incompatible_signal = False
        if self._previous_signal is not None:
            if _tensor_signature(signal) != _tensor_signature(self._previous_signal):
                dense_boundary = True
                incompatible_signal = True
            else:
                relative_change = _relative_l1(signal, self._previous_signal, self.eps)
                scaled_change = _polyval(self.coefficients, relative_change)
                self._accumulated_change += scaled_change
        self._previous_signal = signal.detach().clone()
        non_monotonic = self._last_dense_step is not None and step <= self._last_dense_step

        finite_metric = (
            relative_change is None
            or (
                math.isfinite(relative_change)
                and scaled_change is not None
                and math.isfinite(scaled_change)
                and math.isfinite(self._accumulated_change)
            )
        )
        can_hit = (
            not dense_boundary
            and finite_metric
            and self._residual is not None
            and self._last_dense_step is not None
            and step > self._last_dense_step
            and relative_change is not None
            and self._accumulated_change < self.threshold
        )
        if can_hit:
            distance = step - self._last_dense_step
            residual = self._residual
            if self._derivative is not None:
                residual = residual + self._derivative * distance
            self.events.append(
                CacheEvent(
                    step,
                    True,
                    "polynomial-taylor-extrapolation",
                    self._accumulated_change,
                )
            )
            return residual

        residual = compute_residual()
        detached = residual.detach()
        if (
            self._residual is not None
            and self._last_dense_step is not None
            and step > self._last_dense_step
            and _tensor_signature(self._residual) == _tensor_signature(detached)
        ):
            dense_gap = step - self._last_dense_step
            self._derivative = ((detached - self._residual) / dense_gap).detach()
        else:
            self._derivative = None
        self._residual = detached
        self._last_dense_step = step
        self._accumulated_change = 0.0
        if relative_change is None:
            reason = "seed"
        elif not finite_metric:
            reason = "non-finite-metric"
        elif incompatible_signal:
            reason = "incompatible-state"
        elif dense_boundary:
            reason = "dense-boundary"
        elif non_monotonic:
            reason = "non-monotonic-step"
        else:
            reason = "polynomial-threshold"
        self.events.append(CacheEvent(step, False, reason, scaled_change))
        return residual

    def receipt(self) -> dict[str, object]:
        """Return policy identity plus request-local events for runtime certification."""

        hits = sum(int(event.hit) for event in self.events)
        receipt: dict[str, object] = {
            "algorithm": self.algorithm,
            "decision": "teacache-polynomial",
            "prediction": "first-order-stack-residual",
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "dense_last": self.dense_last,
            "events": len(self.events),
            "hits": hits,
            "event_receipts": [
                {
                    "step": event.step,
                    "hit": event.hit,
                    "reason": event.reason,
                    "accumulated_change": event.accumulated_change,
                }
                for event in self.events
            ],
        }
        branch = getattr(self, "branch", None)
        if branch is not None:
            receipt["branch"] = str(branch)
        request_epoch = getattr(self, "request_epoch", None)
        if request_epoch is not None:
            receipt["request_epoch"] = int(request_epoch)
        request_id = getattr(self, "request_id", None)
        if request_id is not None:
            receipt["request_id"] = str(request_id)
        return receipt


class MagCacheResidualCache:
    """MagCache residual reuse driven by calibrated per-step magnitude ratios."""

    algorithm = "magcache"

    def __init__(
        self,
        ratios: Iterable[float],
        *,
        threshold: float,
        max_skip_steps: int,
        retention_ratio: float = 0.2,
        total_steps: int | None = None,
    ) -> None:
        """Need one calibrated ratio per step; ``max_skip_steps`` must be at least 1."""
        ratios = tuple(float(value) for value in ratios)
        if not ratios:
            raise ValueError("MagCache ratios must not be empty")
        if threshold < 0:
            raise ValueError("MagCache threshold must be non-negative")
        if max_skip_steps < 1:
            raise ValueError("MagCache max_skip_steps must be positive")
        if not 0.0 <= retention_ratio <= 1.0:
            raise ValueError("MagCache retention_ratio must be in [0, 1]")
        self.ratios = ratios
        self.threshold = float(threshold)
        self.max_skip_steps = int(max_skip_steps)
        self.retention_ratio = float(retention_ratio)
        self.total_steps = total_steps
        self._residual: torch.Tensor | None = None
        self._accumulated_error = 0.0
        self._accumulated_steps = 0
        self._accumulated_ratio = 1.0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Reset the product of magnitude ratios so a new clip cannot skip from the last error."""
        self._residual = None
        self._accumulated_error = 0.0
        self._accumulated_steps = 0
        self._accumulated_ratio = 1.0
        self.events.clear()

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Skip from calibrated |1 - Π ratios| until the error or skip-count cap trips.

        ``signal`` is unused: MagCache is schedule-driven, not feature-driven.
        The first ``retention_ratio`` fraction of steps stays dense so early
        transients cannot exhaust the skip budget.
        """
        del signal
        step = int(step)
        if torch.is_grad_enabled():
            residual = compute_residual()
            self.events.append(CacheEvent(step, False, "autograd"))
            return residual

        count = self.total_steps if total_steps is None else total_steps
        if count is None:
            count = len(self.ratios)
        if len(self.ratios) < int(count):
            raise ValueError(
                f"MagCache has {len(self.ratios)} calibrated ratios for {count} steps"
            )
        retained_steps = int(int(count) * self.retention_ratio)
        can_consider = step >= retained_steps and self._residual is not None
        skip_error = None
        if can_consider:
            self._accumulated_ratio *= self.ratios[step]
            self._accumulated_steps += 1
            skip_error = abs(1.0 - self._accumulated_ratio)
            self._accumulated_error += skip_error
            if (
                self._accumulated_error < self.threshold
                and self._accumulated_steps <= self.max_skip_steps
            ):
                self.events.append(
                    CacheEvent(step, True, "calibrated-magnitude", self._accumulated_error)
                )
                return self._residual

        residual = compute_residual()
        self._residual = residual.detach()
        self._accumulated_error = 0.0
        self._accumulated_steps = 0
        self._accumulated_ratio = 1.0
        reason = "seed" if len(self.events) == 0 else "retention-window"
        if can_consider:
            reason = "magnitude-threshold"
        self.events.append(CacheEvent(step, False, reason, skip_error))
        return residual


class AdaCacheResidualCache:
    """AdaCache dynamic skip scheduling using a mid-stack residual probe.

    The Wan model calls :meth:`observe_probe` at its decisive middle block on
    dense steps. The change of that cheap probe selects the next dense-step
    distance from the published AdaCache codebook; intervening steps reuse the
    last full transformer residual.
    """

    algorithm = "adacache"
    DEFAULT_CODEBOOK = (
        (0.03, 12),
        (0.05, 10),
        (0.07, 8),
        (0.09, 6),
        (0.11, 4),
        (1.0, 3),
    )

    def __init__(
        self,
        *,
        codebook: Iterable[tuple[float, int]] | None = None,
        dense_last: int = 1,
        total_steps: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Codebook entries must be positive (limit, rate) pairs; empty would pin rate at 1."""
        values = tuple(codebook or self.DEFAULT_CODEBOOK)
        if not values or any(float(limit) <= 0 or int(rate) < 1 for limit, rate in values):
            raise ValueError("AdaCache codebook entries must be positive")
        self.codebook = tuple((float(limit), int(rate)) for limit, rate in values)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self.eps = float(eps)
        self._residual: torch.Tensor | None = None
        self._previous_probe: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._last_dense_step: int | None = None
        self._next_dense_step = 0
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Forget the schedule so the next clip starts at dense step 0."""
        self._residual = None
        self._previous_probe = None
        self._current_probe = None
        self._last_dense_step = None
        self._next_dense_step = 0
        self.events.clear()

    def observe_probe(self, probe: torch.Tensor) -> None:
        """Record the mid-stack probe from a dense step; ignored on cached steps."""
        self._current_probe = probe.detach()

    def _rate(self, metric: float) -> int:
        """First codebook limit strictly above ``metric``; last rate if none match."""
        for limit, rate in self.codebook:
            if metric < limit:
                return rate
        return self.codebook[-1][1]

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Hit until ``_next_dense_step``; a dense call must invoke :meth:`observe_probe`."""
        del signal
        step = int(step)
        count = self.total_steps if total_steps is None else total_steps
        dense_boundary = count is not None and step >= max(int(count) - self.dense_last, 0)
        if (
            not torch.is_grad_enabled()
            and not dense_boundary
            and self._residual is not None
            and step < self._next_dense_step
        ):
            self.events.append(CacheEvent(step, True, "adaptive-schedule"))
            return self._residual

        # Clear so a missed observe_probe cannot reuse the previous clip's probe.
        self._current_probe = None
        residual = compute_residual()
        probe = self._current_probe if self._current_probe is not None else residual.detach()
        metric = None
        rate = 1
        if self._previous_probe is not None:
            metric = _relative_l1(probe, self._previous_probe, self.eps)
            dense_gap = max(step - int(self._last_dense_step or 0), 1)
            metric /= dense_gap
            rate = self._rate(metric)
        self._previous_probe = probe.detach().clone()
        self._residual = residual.detach()
        self._last_dense_step = step
        self._next_dense_step = step + rate
        if torch.is_grad_enabled():
            reason = "autograd"
        elif dense_boundary:
            reason = "dense-boundary"
        elif metric is None:
            reason = "seed"
        else:
            reason = f"adaptive-rate-{rate}"
        self.events.append(CacheEvent(step, False, reason, metric))
        return residual


class TaylorSeerResidualCache:
    """First-order Taylor residual extrapolation on a fixed dense cadence."""

    algorithm = "taylorseer"

    def __init__(
        self,
        *,
        interval: int = 4,
        dense_last: int = 1,
        total_steps: int | None = None,
    ) -> None:
        """Interval must be ≥ 2; interval 1 would never skip and still store a derivative."""
        if interval < 2:
            raise ValueError("TaylorSeer interval must be at least 2")
        self.interval = int(interval)
        self.dense_last = int(dense_last)
        self.total_steps = total_steps
        self._residual: torch.Tensor | None = None
        self._derivative: torch.Tensor | None = None
        self._last_dense_step: int | None = None
        self.events: list[CacheEvent] = []

    def reset(self) -> None:
        """Drop residual and derivative; a stale slope would extrapolate the wrong clip."""
        self._residual = None
        self._derivative = None
        self._last_dense_step = None
        self.events.clear()

    def run(
        self,
        step: int,
        signal: torch.Tensor,
        compute_residual: Callable[[], torch.Tensor],
        *,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Replay ``residual + derivative * Δt`` on off-cadence steps; ``signal`` is unused."""
        del signal
        step = int(step)
        count = self.total_steps if total_steps is None else total_steps
        dense_boundary = count is not None and step >= max(int(count) - self.dense_last, 0)
        scheduled_dense = step % self.interval == 0
        can_hit = (
            not torch.is_grad_enabled()
            and not dense_boundary
            and not scheduled_dense
            and self._residual is not None
        )
        if can_hit:
            distance = step - int(self._last_dense_step or 0)
            residual = self._residual
            if self._derivative is not None:
                residual = residual + self._derivative * distance
            self.events.append(CacheEvent(step, True, "taylor-extrapolation", float(distance)))
            return residual

        residual = compute_residual()
        if self._residual is not None and self._last_dense_step is not None:
            distance = max(step - self._last_dense_step, 1)
            self._derivative = (residual.detach() - self._residual) / distance
        self._residual = residual.detach()
        self._last_dense_step = step
        if torch.is_grad_enabled():
            reason = "autograd"
        elif dense_boundary:
            reason = "dense-boundary"
        elif len(self.events) == 0:
            reason = "seed"
        else:
            reason = "scheduled-dense"
        self.events.append(CacheEvent(step, False, reason))
        return residual


__all__ = [
    "AdaCacheResidualCache",
    "AdaptiveResidualCache",
    "BlockTaylorSeerCache",
    "BlockCacheEvent",
    "CacheEvent",
    "CustomTaylorResidualCache",
    "DualBlockFeatureCache",
    "DynamicBlockFeatureCache",
    "FirstBlockFeatureCache",
    "FixedStepCache",
    "MagCacheResidualCache",
    "TaylorSeerResidualCache",
    "TeaCacheResidualCache",
]
