"""Compatibility helpers removed from recent Transformers releases.

Keeps older ``from_pretrained`` graphs working when Hugging Face drops
a utility. Do not add new model logic here.
"""

from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────────
# Head mask / prune indices — keep older from_pretrained graphs loading
# ──────────────────────────────────────────────────────────────────────────


def prepare_head_mask(
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
) -> list[None] | torch.Tensor:
    """Expand a one- or two-dimensional attention head mask for model layers."""

    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None].expand(
            num_hidden_layers, -1, -1, -1, -1
        )
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    else:
        raise ValueError(f"head_mask must have dimension 1 or 2, got {head_mask.dim()}")
    return head_mask.to(dtype=torch.float32)


def find_pruneable_heads_and_indices(
    heads: list[int] | set[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    """Return attention heads and flattened indices retained after pruning."""

    requested = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in requested:
        # Indices shift left as earlier heads were already removed from the matrix.
        shifted_head = head - sum(1 for pruned in already_pruned_heads if pruned < head)
        mask[shifted_head] = 0
    keep = mask.view(-1).contiguous().eq(1)
    indices = torch.arange(keep.numel(), dtype=torch.long)[keep]
    return requested, indices
