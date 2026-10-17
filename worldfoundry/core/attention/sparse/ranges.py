# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact attention over unions of query/key ranges without a dense mask."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldfoundry.core.attention.backends.native import scaled_dot_product_attention


@dataclass(frozen=True)
class AttentionRanges:
    """Disjoint query intervals and their merged, non-overlapping key intervals."""

    groups: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]


def attention_range_plan(q_ranges: torch.Tensor, k_ranges: torch.Tensor) -> AttentionRanges:
    """Partition overlapping query ranges and merge their allowed key unions."""
    rows = [
        (int(qs), int(qe), int(ks), int(ke))
        for (qs, qe), (ks, ke) in zip(q_ranges.tolist(), k_ranges.tolist(), strict=True)
        if qe > qs and ke > ks
    ]
    events: dict[int, list[tuple[int, bool]]] = {}
    for index, (qs, qe, _, _) in enumerate(rows):
        events.setdefault(qs, []).append((index, True))
        events.setdefault(qe, []).append((index, False))
    bounds = sorted(events)
    active: set[int] = set()
    groups = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        for index, added in events[start]:
            if added:
                active.add(index)
            else:
                active.discard(index)
        merged: list[tuple[int, int]] = []
        for left, right in sorted((rows[index][2], rows[index][3]) for index in active):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        if merged:
            groups.append((start, stop, tuple(merged)))
    return AttentionRanges(tuple(groups))


def range_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: AttentionRanges,
    *,
    scale: float | None = None,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Attend ``[tokens, heads, channels]`` tensors to exact allowed key unions.

    Query chunking bounds portable SDPA score storage; it never prunes allowed
    keys or changes the softmax domain. The head counts may use grouped queries.
    """
    output = torch.zeros_like(q)
    for start, stop, keys in plan.groups:
        selected_k = torch.cat([k[left:right] for left, right in keys]).transpose(0, 1).unsqueeze(0)
        selected_v = torch.cat([v[left:right] for left, right in keys]).transpose(0, 1).unsqueeze(0)
        for left in range(start, stop, query_chunk_size):
            right = min(left + query_chunk_size, stop)
            result = scaled_dot_product_attention(
                q[left:right].transpose(0, 1).unsqueeze(0),
                selected_k,
                selected_v,
                scale=scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            output[left:right] = result.squeeze(0).transpose(0, 1)
    return output
