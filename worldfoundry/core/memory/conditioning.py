"""Contracts for conditioning diffusion models with stateful memory.

The storage lifecycle in :mod:`worldfoundry.core.memory.base` deliberately does
not prescribe how retrieved memory participates in denoising.  This module is
the corresponding inference-side contract: it describes sequence ownership and
the five places where a memory method may alter a sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

# ──────────────────────────────────────────────────────────────────────────
# Sequence ownership — who owns a span, how it updates, how CFG applies
# ──────────────────────────────────────────────────────────────────────────


class SequenceRole(str, Enum):
    """Semantic ownership of one contiguous latent sequence segment."""

    TARGET = "target"
    MEMORY = "memory"


class SequenceUpdate(str, Enum):
    """How a sampler updates a sequence segment after each solver step."""

    DENOISE = "denoise"
    FROZEN = "frozen"


class GuidanceMode(str, Enum):
    """Classifier-free-guidance behavior for a sequence segment."""

    CLASSIFIER_FREE = "classifier_free"
    CONDITIONAL_ONLY = "conditional_only"


# ──────────────────────────────────────────────────────────────────────────
# Layout contracts — contiguous, ordered segments; empty name / inverted span fail
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SequenceSegment:
    """One contiguous temporal range in a model's latent input."""

    name: str
    start: int
    stop: int
    role: SequenceRole
    update: SequenceUpdate
    guidance: GuidanceMode

    def __post_init__(self) -> None:
        """Refuse empty names and inverted ``[start, stop)`` half-open spans."""
        if not self.name.strip():
            raise ValueError("sequence segment name must be non-empty")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"invalid sequence segment {self.name!r}: [{self.start}, {self.stop})"
            )

    @property
    def length(self) -> int:
        """Return the number of temporal latent tokens in this segment."""

        return self.stop - self.start

    @property
    def slice(self) -> slice:
        """Return the Python slice represented by this segment."""

        return slice(self.start, self.stop)


@dataclass(frozen=True)
class DenoisingLayout:
    """Validated temporal layout passed through a memory-aware denoiser."""

    segments: tuple[SequenceSegment, ...]
    temporal_dim: int = 1

    def __post_init__(self) -> None:
        """Require at least one segment and a contiguous, ordered cover from 0."""
        if not self.segments:
            raise ValueError("denoising layout must contain at least one segment")
        cursor = 0
        for segment in self.segments:
            if segment.start != cursor:
                raise ValueError(
                    "denoising layout segments must be contiguous and ordered; "
                    f"expected start {cursor}, got {segment.start} for {segment.name!r}"
                )
            cursor = segment.stop

    @property
    def length(self) -> int:
        """Return the complete temporal sequence length."""

        return self.segments[-1].stop

    def by_role(self, role: SequenceRole) -> tuple[SequenceSegment, ...]:
        """Return all segments owned by ``role`` in layout order."""

        return tuple(segment for segment in self.segments if segment.role is role)


# ──────────────────────────────────────────────────────────────────────────
# Sampler extension points — adapter owns layout; base sampler owns encode/step
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryCondition:
    """Retrieved memory after conversion to model-facing values."""

    values: Any
    actions: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DenoisingMemoryAdapter(Protocol):
    """Sampler extension points required by memory-conditioned diffusion.

    Implementations own the layout they create.  The base sampler remains in
    charge of prompt encoding, model calls, scheduling, and decoding.
    """

    @property
    def layout(self) -> DenoisingLayout | None:
        """Return the layout after :meth:`prepare`, or ``None`` beforehand."""

    def prepare(self, target_latents: Sequence[Any]) -> list[Any]:
        """Combine target noise and the selected memory condition."""

    def model_kwargs(self) -> Mapping[str, Any]:
        """Return kwargs forwarded to the denoising model."""

    def merge_guidance(
        self,
        conditional: Sequence[Any],
        unconditional: Sequence[Any],
        scale: float,
    ) -> list[Any]:
        """Merge conditional/unconditional predictions by segment semantics."""

    def after_step(self, latents: Sequence[Any]) -> list[Any]:
        """Apply post-scheduler invariants such as freezing clean memory."""

    def finalize(self, latents: Sequence[Any]) -> list[Any]:
        """Remove conditioning-only segments before decoding."""


__all__ = [
    "DenoisingLayout",
    "DenoisingMemoryAdapter",
    "GuidanceMode",
    "MemoryCondition",
    "SequenceRole",
    "SequenceSegment",
    "SequenceUpdate",
]
