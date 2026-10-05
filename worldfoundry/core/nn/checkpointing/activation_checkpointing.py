"""Selective activation-checkpoint (SAC) configuration for training.

Recompute cheap ops and store expensive attention outputs. Inference
should leave this off — use ``checkpoint_compat`` only when a released
graph still contains checkpoint wrappers.

Not this module:
    Inference-time identity wrappers live in
    :mod:`worldfoundry.core.nn.checkpointing.checkpoint_compat`. The model still owns
    the ``torch.utils.checkpoint`` call site that consumes these contexts.

Public surface:

- :class:`CheckpointMode` — ``none`` / ``mm_only`` / ``block_wise``.
- :class:`SACConfig` — frozen config plus :meth:`SACConfig.get_context_fn`.
- :func:`matrix_ops_checkpoint_policy` / :func:`matrix_ops_context_fn` /
  :func:`block_context_fn` — SAC context factories.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch


# ──────────────────────────────────────────────────────────────────────────
# Policy enum and matrix-op set — what SAC must save vs recompute
# ──────────────────────────────────────────────────────────────────────────


class CheckpointMode(str, Enum):
    """Supported activation-checkpoint policies."""

    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"

    def __str__(self) -> str:
        """Return the on-disk / config string so Hydra can round-trip the enum."""

        return self.value


# Flash / SDPA / GEMM outputs dominate activation memory; everything else is
# cheaper to recompute than to stash for backward.
_MATRIX_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten.addmm.default,
}


# ──────────────────────────────────────────────────────────────────────────
# Context factories — consumed by torch.utils.checkpoint wrappers
# ──────────────────────────────────────────────────────────────────────────


def matrix_ops_checkpoint_policy(context: Any, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Save matrix/attention kernels and recompute inexpensive surrounding ops."""

    del context, args, kwargs
    from torch.utils.checkpoint import CheckpointPolicy

    save = function in _MATRIX_OPS or "flash_attn" in str(function)
    return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE


def matrix_ops_context_fn() -> tuple[Any, Any]:
    """Build the selective-checkpoint context pair for :attr:`CheckpointMode.MM_ONLY`."""

    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(matrix_ops_checkpoint_policy)


def block_context_fn() -> tuple[Any, Any]:
    """Return a no-op context so whole-block checkpointing owns save/recompute."""

    from torch.utils.checkpoint import noop_context_fn

    return noop_context_fn()


# ──────────────────────────────────────────────────────────────────────────
# Per-model config — mode plus block stride
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SACConfig:
    """Configuration consumed by models that optionally wrap whole blocks."""

    mode: CheckpointMode | str = CheckpointMode.MM_ONLY
    every_n_blocks: int = 1

    def __post_init__(self) -> None:
        """Coerce ``mode`` to :class:`CheckpointMode` and reject a non-positive stride.

        Raises:
            ValueError: ``every_n_blocks`` is not strictly positive.
        """

        object.__setattr__(self, "mode", CheckpointMode(self.mode))
        if self.every_n_blocks <= 0:
            raise ValueError("every_n_blocks must be positive")

    def get_context_fn(self) -> Callable[[], tuple[Any, Any]]:
        """Return the factory matching ``mode``.

        Raises:
            ValueError: ``mode`` is :attr:`CheckpointMode.NONE` (no wrapper).
        """

        if self.mode is CheckpointMode.MM_ONLY:
            return matrix_ops_context_fn
        if self.mode is CheckpointMode.BLOCK_WISE:
            return block_context_fn
        raise ValueError("CheckpointMode.NONE does not define a checkpoint context")


__all__ = [
    "CheckpointMode",
    "SACConfig",
    "block_context_fn",
    "matrix_ops_checkpoint_policy",
    "matrix_ops_context_fn",
]
