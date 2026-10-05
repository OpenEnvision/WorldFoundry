"""Generic feature-norm token pruning and reconstruction.

Responsibility: score a token segment, gather the keep-set for expensive
blocks, then scatter processed tokens back and fill drops from the
previous full segment.

This module is not a model-specific DiT hook and does not own the
prunable seam, which steps prune, or CUDA Graph capture. Shape changes
generally cannot enter a graph — callers keep that decision outside.

Public surface:
- :func:`select_token_indices` / :func:`prune_tokens` / :func:`restore_tokens`
- :class:`TokenPruneState` — scatter metadata for one segment
- :class:`TokenPruner` — previous-step compensation across denoising calls

Model code retains control of the prunable token seam, active denoising
steps, and compensation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

TokenScoreMethod = Literal[
    "feat_norm",
    "feat_l1",
    "feat_linf",
    "feat_var",
    "velocity",
    "uniform",
    "random",
]


# ──────────────────────────────────────────────────────────────────────────
# Scoring — keep-set is sorted so gather/scatter stay deterministic
# ──────────────────────────────────────────────────────────────────────────


def _uniform_indices(num_tokens: int, keep: int, device: torch.device) -> torch.Tensor:
    """Stride the sequence so a keep-ratio still covers the full temporal span."""
    positions = torch.arange(keep, device=device, dtype=torch.long)
    return ((positions * num_tokens) // keep).clamp_(max=num_tokens - 1)


def select_token_indices(
    hidden_states: torch.Tensor,
    keep_ratio: float,
    *,
    method: TokenScoreMethod | str = "feat_norm",
    seq_dim: int = 1,
    previous_velocity: torch.Tensor | None = None,
    random_seed: int = 42,
) -> torch.Tensor:
    """Return ascending indices for tokens retained by a pruning policy."""

    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have a sequence and feature dimension")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    moved = hidden_states.movedim(seq_dim, -2)
    num_tokens = moved.shape[-2]
    keep = max(1, min(num_tokens, int(round(num_tokens * float(keep_ratio)))))
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=hidden_states.device, dtype=torch.long)

    normalized = str(method).strip().casefold()
    score_source = moved
    if normalized in {"velocity", "vel"} and previous_velocity is not None:
        score_source = previous_velocity.movedim(seq_dim, -2)
        normalized = "velocity"

    if normalized in {"feat_norm", "feat", "norm", "feat_l2", "velocity"}:
        scores = score_source.float().pow(2).sum(-1)
    elif normalized == "feat_l1":
        scores = score_source.float().abs().sum(-1)
    elif normalized in {"feat_linf", "feat_max"}:
        scores = score_source.float().abs().amax(-1)
    elif normalized == "feat_var":
        scores = score_source.float().var(-1, unbiased=False)
    elif normalized in {"random", "rand"}:
        generator = torch.Generator(device=hidden_states.device).manual_seed(int(random_seed))
        selected = torch.randperm(num_tokens, generator=generator, device=hidden_states.device)[:keep]
        return torch.sort(selected).values
    else:
        return _uniform_indices(num_tokens, keep, hidden_states.device)

    if scores.ndim > 1:
        # Mean over batch/extra dims so CFG branches share one keep-set.
        reduce_dims = tuple(range(scores.ndim - 1))
        scores = scores.mean(dim=reduce_dims)
    # Sort so gather/scatter stay layout-stable across steps.
    return torch.sort(torch.topk(scores, keep, largest=True).indices).values


# ──────────────────────────────────────────────────────────────────────────
# Compact / expand — only the selected seam changes length
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TokenPruneState:
    """Metadata needed to reconstruct a pruned token segment."""

    indices: torch.Tensor
    start: int
    end: int
    full_length: int
    seq_dim: int

    @property
    def kept_length(self) -> int:
        """Number of retained tokens; used to slice the compacted sequence."""
        return int(self.indices.numel())


def prune_tokens(
    hidden_states: torch.Tensor,
    keep_ratio: float,
    *,
    method: TokenScoreMethod | str = "feat_norm",
    seq_dim: int = 1,
    start: int = 0,
    end: int | None = None,
    previous_velocity: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TokenPruneState]:
    """Gather the selected part of a token segment before expensive blocks."""

    normalized_dim = seq_dim % hidden_states.ndim
    moved = hidden_states.movedim(normalized_dim, -2)
    sequence_length = moved.shape[-2]
    stop = sequence_length if end is None else int(end)
    begin = int(start)
    if not 0 <= begin <= stop <= sequence_length:
        raise ValueError(f"invalid token segment [{begin}:{stop}] for length {sequence_length}")
    segment = moved[..., begin:stop, :]
    velocity_segment = None
    if previous_velocity is not None:
        velocity_moved = previous_velocity.movedim(normalized_dim, -2)
        velocity_segment = velocity_moved[..., begin:stop, :]
    indices = select_token_indices(
        segment,
        keep_ratio,
        method=method,
        seq_dim=-2,
        previous_velocity=velocity_segment,
    )
    kept = segment.index_select(-2, indices)
    compact = torch.cat((moved[..., :begin, :], kept, moved[..., stop:, :]), dim=-2)
    state = TokenPruneState(
        indices=indices,
        start=begin,
        end=stop,
        full_length=stop - begin,
        seq_dim=normalized_dim,
    )
    return compact.movedim(-2, normalized_dim), state


# ──────────────────────────────────────────────────────────────────────────
# Restore — dropped tokens come from the prior full segment, not zeros
# ──────────────────────────────────────────────────────────────────────────


def restore_tokens(
    processed: torch.Tensor,
    state: TokenPruneState,
    *,
    compensation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter processed tokens and restore dropped tokens from prior state.

    ``compensation`` is the previous full segment in the original tensor
    layout. If omitted, dropped tokens are zero-filled.
    """

    moved = processed.movedim(state.seq_dim, -2)
    kept_end = state.start + state.kept_length
    kept = moved[..., state.start:kept_end, :]
    expected_shape = (*kept.shape[:-2], state.full_length, kept.shape[-1])
    if compensation is None:
        full = kept.new_zeros(expected_shape)
    else:
        full = compensation.movedim(state.seq_dim, -2).to(device=kept.device, dtype=kept.dtype)
        if full.shape != expected_shape:
            raise ValueError(f"compensation shape {tuple(full.shape)} does not match {expected_shape}")
        full = full.clone()
    full = full.index_copy(-2, state.indices, kept)
    restored = torch.cat((moved[..., : state.start, :], full, moved[..., kept_end:, :]), dim=-2)
    return restored.movedim(-2, state.seq_dim)


# ──────────────────────────────────────────────────────────────────────────
# Stateful pruner — first call is dense so later drops have compensation
# ──────────────────────────────────────────────────────────────────────────


class TokenPruner:
    """Stateful previous-step compensation for repeated denoising calls."""

    def __init__(self, keep_ratio: float, *, method: TokenScoreMethod | str = "feat_norm") -> None:
        """Reject a non-positive keep ratio; first ``prune`` is always a dense seed."""
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = float(keep_ratio)
        self.method = str(method)
        self._previous: dict[object, torch.Tensor] = {}
        self._pending_dense: dict[object, tuple[int, int, int]] = {}

    def reset(self, key: object | None = None) -> None:
        """Drop compensation for one branch, or every branch when ``key`` is omitted."""
        if key is None:
            self._previous.clear()
            self._pending_dense.clear()
        else:
            self._previous.pop(key, None)
            self._pending_dense.pop(key, None)

    def prune(
        self,
        hidden_states: torch.Tensor,
        *,
        key: object = "default",
        seq_dim: int = 1,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[torch.Tensor, TokenPruneState | None]:
        """Prune after a dense seed call; first use only records compensation."""

        normalized_dim = seq_dim % hidden_states.ndim
        moved = hidden_states.movedim(normalized_dim, -2)
        stop = moved.shape[-2] if end is None else int(end)
        if key not in self._previous or self.keep_ratio >= 1.0:
            self._pending_dense[key] = (normalized_dim, int(start), stop)
            return hidden_states, None
        return prune_tokens(
            hidden_states,
            self.keep_ratio,
            method=self.method,
            seq_dim=normalized_dim,
            start=start,
            end=stop,
        )

    def restore(
        self,
        processed: torch.Tensor,
        state: TokenPruneState | None,
        *,
        key: object = "default",
    ) -> torch.Tensor:
        """Scatter a pruned step, or snapshot the dense seed as next compensation."""
        if state is None:
            pending = self._pending_dense.pop(key, None)
            if pending is not None:
                seq_dim, start, end = pending
                moved = processed.movedim(seq_dim, -2)
                self._previous[key] = moved[..., start:end, :].movedim(-2, seq_dim).detach()
            return processed
        restored = restore_tokens(processed, state, compensation=self._previous.get(key))
        moved = restored.movedim(state.seq_dim, -2)
        self._previous[key] = (
            moved[..., state.start:state.end, :].movedim(-2, state.seq_dim).detach()
        )
        return restored


__all__ = [
    "TokenPruneState",
    "TokenPruner",
    "prune_tokens",
    "restore_tokens",
    "select_token_indices",
]
