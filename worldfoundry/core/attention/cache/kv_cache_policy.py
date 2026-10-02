"""Pluggable eviction policies for causal video KV caches.

Streaming video models keep a KV cache that grows with every generated chunk, so
cache memory — not compute — is what bounds the horizon. The training-free
mitigations published for this all share one shape: the sampler is untouched and
only the *cache-management policy* changes. Each policy answers two questions
after a chunk is committed:

1. which cached tokens survive, and
2. what temporal position each survivor claims for RoPE.

This module makes that pair explicit so a runtime can swap policies without
touching its attention or sampling code. Policies here are **structural**: they
decide from the block schedule alone, which keeps them pure index arithmetic,
cheap, and testable without a model. Policies that score cached tokens against
live queries (attention Top-K, similarity selection, EMA memory compression)
need an attention observer; :class:`CacheDecision` already carries what they
would emit, so they can be added behind the same protocol.

Positions are expressed in *frames*, matching video RoPE, and expanded to tokens
via :attr:`CacheState.frame_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────
# Policy I/O — blocks, state, and the decision a policy must return
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CachedBlock:
    """One committed chunk inside the cache.

    Attributes:
        block_idx: Index of the chunk in the generated sequence.
        frame_start: Absolute first frame this chunk produced.
        frame_count: Frames in this chunk.
    """

    block_idx: int
    frame_start: int
    frame_count: int

    @property
    def frame_end(self) -> int:
        """One past the last absolute frame in this chunk."""
        return self.frame_start + self.frame_count


@dataclass(frozen=True)
class CacheState:
    """Everything a structural policy needs to decide what to keep.

    Attributes:
        blocks: Committed chunks, oldest first, contiguous in cache order.
        frame_tokens: Tokens per latent frame.
        current_block_idx: Index of the chunk just committed.
    """

    blocks: tuple[CachedBlock, ...]
    frame_tokens: int
    current_block_idx: int

    @property
    def cached_frames(self) -> int:
        """Total frames currently held."""
        return sum(block.frame_count for block in self.blocks)

    @property
    def cached_tokens(self) -> int:
        """Total tokens currently held."""
        return self.cached_frames * self.frame_tokens

    def token_span(self, position: int) -> tuple[int, int]:
        """Return the ``[start, end)`` token span of the block at ``position``.

        Args:
            position: Index into :attr:`blocks`, not a generation block index.
        """
        start = sum(block.frame_count for block in self.blocks[:position]) * self.frame_tokens
        return start, start + self.blocks[position].frame_count * self.frame_tokens


@dataclass(frozen=True)
class CacheDecision:
    """What a policy decided for one committed chunk.

    Attributes:
        keep_indices: Token indices to retain, ascending. ``None`` keeps
            everything, which lets a no-op policy avoid a gather entirely.
        frame_positions: Per-surviving-token frame index for RoPE. ``None``
            leaves positions untouched — correct whenever survivors keep their
            original temporal layout. Policies that renumber frames to stay
            inside the trained window populate it.
        kept_blocks: Positions into :attr:`CacheState.blocks` that survived
            whole. Structural policies fill this so the cache can keep its block
            bookkeeping; token-level policies leave it ``None`` and the cache
            collapses survivors into one synthetic block.
    """

    keep_indices: Tensor | None = None
    frame_positions: Tensor | None = None
    kept_blocks: tuple[int, ...] | None = None

    @property
    def evicts(self) -> bool:
        """Whether this decision drops anything."""
        return self.keep_indices is not None

    @property
    def remaps_positions(self) -> bool:
        """Whether survivors are renumbered for RoPE."""
        return self.frame_positions is not None


@runtime_checkable
class KVCachePolicy(Protocol):
    """Decides which cached tokens survive after a chunk is committed."""

    def decide(self, state: CacheState) -> CacheDecision:
        """Return the surviving tokens and their RoPE positions."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Index helpers — expand kept blocks into token / frame vectors
# ──────────────────────────────────────────────────────────────────────────


def _token_indices(state: CacheState, positions: list[int], *, device: torch.device | str = "cpu") -> Tensor:
    """Expand kept block positions into ascending token indices."""
    if not positions:
        return torch.empty(0, dtype=torch.long, device=device)
    spans = [state.token_span(position) for position in sorted(positions)]
    return torch.cat([torch.arange(start, end, dtype=torch.long, device=device) for start, end in spans])


def _frame_positions(
    state: CacheState,
    positions: list[int],
    frame_starts: list[int],
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Expand per-block frame starts into a per-token frame index vector."""
    chunks = []
    for position, frame_start in zip(sorted(positions), frame_starts):
        block = state.blocks[position]
        frames = torch.arange(frame_start, frame_start + block.frame_count, dtype=torch.long, device=device)
        chunks.append(frames.repeat_interleave(state.frame_tokens))
    if not chunks:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(chunks)


# ──────────────────────────────────────────────────────────────────────────
# Structural policies — decide from the block schedule alone
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KeepAllPolicy:
    """Never evict. The baseline: exact attention at unbounded cache growth."""

    def decide(self, state: CacheState) -> CacheDecision:
        """Keep every cached token."""
        del state
        return CacheDecision()


@dataclass(frozen=True)
class SlidingWindowPolicy:
    """Attention sink plus a FIFO window over the most recent blocks.

    The oldest ``sink_blocks`` are pinned — dropping them is what makes naive
    FIFO windows drift — and the remaining budget holds the newest blocks.
    ``sink_blocks=0`` gives a pure FIFO window.

    Attributes:
        window_blocks: Total block budget, sinks included.
        sink_blocks: Leading blocks that are never evicted.
    """

    window_blocks: int = 6
    sink_blocks: int = 0

    def __post_init__(self) -> None:
        """Require a positive window and sinks strictly inside that budget."""
        if self.window_blocks < 1:
            raise ValueError(f"window_blocks must be >= 1, got {self.window_blocks}")
        if self.sink_blocks < 0:
            raise ValueError(f"sink_blocks must be non-negative, got {self.sink_blocks}")
        if self.sink_blocks >= self.window_blocks:
            raise ValueError(f"sink_blocks ({self.sink_blocks}) must be < window_blocks ({self.window_blocks})")

    def decide(self, state: CacheState) -> CacheDecision:
        """Keep the sinks plus as many recent blocks as the budget allows."""
        total = len(state.blocks)
        if total <= self.window_blocks:
            return CacheDecision()

        recent_budget = self.window_blocks - self.sink_blocks
        kept = list(range(self.sink_blocks)) + list(range(total - recent_budget, total))
        return CacheDecision(keep_indices=_token_indices(state, kept), kept_blocks=tuple(kept))


@dataclass(frozen=True)
class BankedSinkPolicy:
    """Pin an early bank plus the newest blocks and close the temporal gap.

    The first ``bank_blocks`` are frozen as a long-term anchor and the newest
    ``recent_blocks`` carry short-term continuity; everything between them is
    dropped. Survivors are then renumbered into one contiguous frame range, so
    RoPE never sees the hole eviction opened — which is what separates this from
    :class:`SlidingWindowPolicy`, whose survivors keep their original, gapped
    positions.

    Because the bank is pinned rather than sampled, positions stay stable across
    steps and the model sees the whole anchor every step. Cycling a *subset* of
    the bank, as Rolling Sink does, needs per-step visibility masking, which this
    layer does not express — eviction alone cannot hide a resident block.

    Attributes:
        bank_blocks: Leading blocks kept permanently.
        recent_blocks: Newest blocks always retained.
    """

    bank_blocks: int = 4
    recent_blocks: int = 2

    def __post_init__(self) -> None:
        """Require at least one bank block and one recent block."""
        if self.bank_blocks < 1:
            raise ValueError(f"bank_blocks must be >= 1, got {self.bank_blocks}")
        if self.recent_blocks < 1:
            raise ValueError(f"recent_blocks must be >= 1, got {self.recent_blocks}")

    @property
    def resident_blocks(self) -> int:
        """Upper bound on blocks held at any time."""
        return self.bank_blocks + self.recent_blocks

    def decide(self, state: CacheState) -> CacheDecision:
        """Keep bank plus newest blocks, renumbered into a contiguous frame range."""
        total = len(state.blocks)
        if total <= self.resident_blocks:
            return CacheDecision()

        kept = list(range(self.bank_blocks)) + list(range(total - self.recent_blocks, total))

        frame_start = 0
        frame_starts = []
        for position in kept:
            frame_starts.append(frame_start)
            frame_start += state.blocks[position].frame_count

        return CacheDecision(
            keep_indices=_token_indices(state, kept),
            frame_positions=_frame_positions(state, kept, frame_starts),
            kept_blocks=tuple(kept),
        )


@dataclass(frozen=True)
class BlockRelativeRoPEPolicy:
    """Bounded cache that pins the newest block at a fixed frame index.

    Every step the newest block is placed at ``frame_limit`` and older blocks are
    shifted backwards from there, so the model always sees positions inside its
    trained window no matter how long the rollout runs. Blocks that would shift
    below zero are clamped to frame 0, collapsing the distant past into a single
    anchor rather than extrapolating RoPE past what training covered.

    Attributes:
        cache_blocks: Block budget, sinks included.
        sink_blocks: Leading blocks that are never evicted.
        frame_limit: Frame index assigned to the newest block's first frame.
    """

    cache_blocks: int = 6
    sink_blocks: int = 1
    frame_limit: int = 21

    def __post_init__(self) -> None:
        """Require a positive budget, sinks inside it, and ``frame_limit >= 2``."""
        if self.cache_blocks < 1:
            raise ValueError(f"cache_blocks must be >= 1, got {self.cache_blocks}")
        if self.sink_blocks < 0:
            raise ValueError(f"sink_blocks must be non-negative, got {self.sink_blocks}")
        if self.sink_blocks >= self.cache_blocks:
            raise ValueError(f"sink_blocks ({self.sink_blocks}) must be < cache_blocks ({self.cache_blocks})")
        if self.frame_limit < 2:
            raise ValueError(f"frame_limit must be >= 2, got {self.frame_limit}")

    def decide(self, state: CacheState) -> CacheDecision:
        """Evict past the budget and shift survivors back from ``frame_limit``."""
        total = len(state.blocks)
        recent_budget = self.cache_blocks - self.sink_blocks
        kept = (
            list(range(total))
            if total <= self.cache_blocks
            else list(range(self.sink_blocks)) + list(range(total - recent_budget, total))
        )

        if not kept:
            return CacheDecision()

        # Anchor the newest block at frame_limit, then walk backwards so each
        # older block ends exactly where the next one begins. Blocks pushed past
        # frame 0 clamp there rather than extrapolating RoPE below the trained range.
        frame_starts = [0] * len(kept)
        cursor = self.frame_limit
        frame_starts[-1] = cursor
        for offset in range(len(kept) - 2, -1, -1):
            cursor = max(0, cursor - state.blocks[kept[offset]].frame_count)
            frame_starts[offset] = cursor

        decision_indices = None if total <= self.cache_blocks else _token_indices(state, kept)
        return CacheDecision(
            keep_indices=decision_indices,
            frame_positions=_frame_positions(state, kept, frame_starts),
            kept_blocks=tuple(kept),
        )


# ──────────────────────────────────────────────────────────────────────────
# Registry — construct a policy by name without importing the class
# ──────────────────────────────────────────────────────────────────────────

POLICY_REGISTRY: dict[str, type] = {
    "keep_all": KeepAllPolicy,
    "sliding_window": SlidingWindowPolicy,
    "banked_sink": BankedSinkPolicy,
    "block_relative_rope": BlockRelativeRoPEPolicy,
}


def build_policy(name: str, **kwargs) -> KVCachePolicy:
    """Construct a registered policy by name.

    Args:
        name: Registry key, e.g. ``"sliding_window"``.
        **kwargs: Policy-specific configuration.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        policy_cls = POLICY_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_REGISTRY))
        raise KeyError(f"Unknown KV cache policy {name!r}. Known policies: {known}") from exc
    return policy_cls(**kwargs)


__all__ = [
    "POLICY_REGISTRY",
    "BankedSinkPolicy",
    "BlockRelativeRoPEPolicy",
    "CacheDecision",
    "CacheState",
    "CachedBlock",
    "KVCachePolicy",
    "KeepAllPolicy",
    "SlidingWindowPolicy",
    "build_policy",
]
