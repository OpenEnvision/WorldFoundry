"""Online EasyCache transform-vector reuse at a tensor block-stack seam.

The change factor is learned from successive dense input/output pairs, then
predicts accumulated relative output drift from per-step input changes. This
is the EasyCache controller, without TeaCache's fitted polynomial. Adapters
choose the tensor seam; using embedded tokens differs from raw-latent variants.
One instance belongs to one request, stage/expert and CFG branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field

import torch

from worldfoundry.core.acceleration.cache import BlockCacheEvent


def _snapshot_condition(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_condition(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot_condition(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"EasyCache cannot track conditioning of type {type(value).__name__}")


def _same_condition(value, snapshot) -> bool:
    if isinstance(value, torch.Tensor):
        return (
            isinstance(snapshot, torch.Tensor)
            and value.device == snapshot.device
            and value.dtype == snapshot.dtype
            and torch.equal(value, snapshot)
        )
    if isinstance(value, Mapping):
        return (
            isinstance(snapshot, Mapping)
            and value.keys() == snapshot.keys()
            and all(_same_condition(item, snapshot[key]) for key, item in value.items())
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value) is type(snapshot)
            and len(value) == len(snapshot)
            and all(_same_condition(item, saved) for item, saved in zip(value, snapshot, strict=True))
        )
    return type(value) is type(snapshot) and value == snapshot


@dataclass(frozen=True, slots=True)
class EasyCacheConfig:
    """Explicit approximation budget; zero threshold runs every block exactly."""

    threshold: float
    warmup_steps: int = 2
    dense_last: int = 1
    max_skip_steps: int = 4
    subsample_stride: int = 1
    eps: float = 1e-8
    algorithm: str = field(default="easycache", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("EasyCache threshold must be finite and non-negative")
        for name in ("warmup_steps", "dense_last", "max_skip_steps", "subsample_stride"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name in {"max_skip_steps", "subsample_stride"} else 0)
            ):
                raise ValueError(f"EasyCache {name} has an invalid integer value")
        if isinstance(self.eps, bool) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("EasyCache eps must be finite and positive")

    @property
    def options(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "algorithm"}


class EasyCache:
    """Reuse ``current_input + last_dense_transform`` below an online budget."""

    algorithm = "easycache"

    def __init__(self, config: EasyCacheConfig, *, total_steps: int) -> None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("EasyCache total_steps must be a positive integer")
        self.config = config
        self.total_steps = total_steps
        self.reset()

    def _clear_tensors(self) -> None:
        self._previous_input = None
        self._dense_input = None
        self._dense_output = None
        self._residual = None
        self._factor = None
        self._accumulator = 0.0
        self._skips = 0
        self._layout = None

    def reset(self) -> None:
        self._clear_tensors()
        self._previous_step = -1
        self._conditioning = None
        self._invalidation_reason = None
        self.events: list[BlockCacheEvent] = []

    def observe_conditioning(self, values: Mapping[str, object]) -> None:
        """Invalidate reuse on changed conditions, including in-place tensor edits.

        Inference tensors do not have version counters, so snapshots are
        compared by value. Conditions stay fixed in the common native path.
        """
        if self._conditioning is None:
            self._conditioning = _snapshot_condition(values)
        elif not _same_condition(values, self._conditioning):
            snapshot = _snapshot_condition(values)
            self._clear_tensors()
            self._invalidation_reason = "conditioning-change"
            self._conditioning = snapshot

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        # Snapshot even FP32 views: schedulers/callers may mutate their inputs.
        return value.detach().reshape(value.shape[0], -1)[:, :: self.config.subsample_stride].float().clone()

    def run_blocks(
        self,
        step: int,
        value: torch.Tensor,
        run_block: Callable[[int, torch.Tensor], torch.Tensor],
        *,
        block_count: int,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """Run or reuse the whole stack; dense execution returns its original output."""
        count = self.total_steps if total_steps is None else total_steps
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or count != self.total_steps
            or not 0 <= step < count
            or step != self._previous_step + 1
        ):
            raise ValueError("EasyCache steps must be unique, contiguous and use a fixed total_steps")
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, int)
            or block_count <= 0
            or value.ndim < 2
            or value.numel() == 0
            or value.shape[0] <= 0
            or not value.is_floating_point()
        ):
            raise ValueError("EasyCache requires a non-empty tensor block stack")
        layout = (tuple(value.shape), value.dtype, value.device, block_count)
        reason = self._invalidation_reason
        self._invalidation_reason = None
        if torch.is_grad_enabled():
            self._clear_tensors()
            reason = "autograd"
        elif self._layout is not None and layout != self._layout:
            self._clear_tensors()
            reason = "layout-change"
        probe = self._probe(value)
        finite = bool(torch.isfinite(value if self.config.subsample_stride > 1 else probe).all())
        if not finite:
            self._clear_tensors()
            reason = "nonfinite-input"
        if reason is None:
            if step < self.config.warmup_steps:
                reason = "warmup"
            elif step >= count - self.config.dense_last:
                reason = "dense-last"
            elif self._residual is None or self._factor is None:
                reason = "seed"
            elif self._skips >= self.config.max_skip_steps:
                reason = "max-skip"
        if reason is None:
            delta = (probe - self._previous_input).abs().mean(dim=1)
            norm = self._dense_output.abs().mean(dim=1).clamp_min(self.config.eps)
            estimate = float((self._factor * delta / norm).max().item())
            self._accumulator += estimate
            if not math.isfinite(self._accumulator) or self._accumulator >= self.config.threshold:
                reason = "threshold"
        hit = reason is None
        if hit:
            output = value + self._residual
            self._skips += 1
        else:
            output = value
            for index in range(block_count):
                output = run_block(index, output)
            if output.shape != value.shape or output.device != value.device or output.dtype != value.dtype:
                raise ValueError("EasyCache block output must preserve input shape, device and dtype")
            output_probe = self._probe(output)
            output_finite = bool(torch.isfinite(output if self.config.subsample_stride > 1 else output_probe).all())
            if finite and not torch.is_grad_enabled() and output_finite:
                if self._dense_input is not None:
                    input_delta = (probe - self._dense_input).abs().mean(dim=1).clamp_min(self.config.eps)
                    self._factor = (output_probe - self._dense_output).abs().mean(dim=1) / input_delta
                self._dense_input, self._dense_output = probe, output_probe
                self._residual = (output - value).detach().clone()
                self._layout = layout
            else:
                self._clear_tensors()
            self._accumulator = 0.0
            self._skips = 0
        self._previous_input = probe if finite and not torch.is_grad_enabled() else None
        self._previous_step = step
        indices = tuple(range(block_count))
        self.events.append(
            BlockCacheEvent(
                self.algorithm,
                step,
                hit,
                reason or "below-threshold",
                () if hit else indices,
                indices if hit else (),
                self._accumulator,
            )
        )
        return output

    @property
    def dense_block_calls(self) -> int:
        return sum(event.dense_block_calls for event in self.events)

    @property
    def skipped_block_calls(self) -> int:
        return sum(event.skipped_block_calls for event in self.events)

    def receipt(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "variant": "embedded-token-block-stack",
            "parameters": self.config.options,
            "events": [event.receipt() for event in self.events],
            "dense_block_calls": self.dense_block_calls,
            "skipped_block_calls": self.skipped_block_calls,
        }
