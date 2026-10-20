# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Shared Human/Object phase allocation from target-presence runs.

The protocol has ``K`` possible departure gaps::

    V [D?] R1 [D?] R2 ... [D?] RK

Presence components are matched to the fixed revisit boundaries.  A component
that covers several consecutive boundaries is split evenly across those
revisits.  This represents all ``2**K`` departure-insertion patterns without a
separate never-departed fallback.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import combinations

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

MAX_SHORT_MISSED_FRAMES = 2
MIN_REAPPEAR_MATCH_RUN = 3
MAX_EXPECTED_REVISITS = 3
REVISIT_BOUNDARY_BUFFER_FRAMES = 30


@dataclass(frozen=True)
class TargetPhase:
    target_id: int
    visible_indices: tuple[int, ...]
    departure_indices: tuple[int, ...]
    return_indices: tuple[int, ...]
    status: str
    revisit_indices: tuple[tuple[int, ...], ...] = ()
    revisit_slots: tuple[tuple[int, int], ...] = ()
    revisit_nominal_boundaries: tuple[int, ...] = ()
    revisit_search_windows: tuple[tuple[int, int], ...] = ()
    revisit_sources: tuple[str, ...] = ()
    revisit_departure_evidence: tuple[bool, ...] = ()

    @property
    def is_scorable(self) -> bool:
        return self.status == "ok" or (self.status == "not_departed" and bool(self.return_indices))


def validate_expected_revisits(value: object) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_EXPECTED_REVISITS
    ):
        raise DataInvalidError("expected_revisits must be null (automatic) or an integer in [1, 3]")
    return value


def _balanced_ranges(
    start: int,
    end: int,
    parts: int,
) -> tuple[tuple[int, ...], ...]:
    """Split ``[start, end)`` exactly once, with earlier parts one frame longer."""

    if parts < 1:
        raise ValueError("parts must be positive")
    length = max(0, end - start)
    base, remainder = divmod(length, parts)
    ranges: list[tuple[int, ...]] = []
    cursor = start
    for part_index in range(parts):
        size = base + int(part_index < remainder)
        ranges.append(tuple(range(cursor, cursor + size)))
        cursor += size
    return tuple(ranges)


def _close_short_internal_gaps(
    presence: list[bool],
    *,
    max_gap: int,
) -> list[bool]:
    smoothed = list(presence)
    index = 0
    while index < len(smoothed):
        if smoothed[index]:
            index += 1
            continue
        start = index
        while index < len(smoothed) and not smoothed[index]:
            index += 1
        if start > 0 and index < len(smoothed) and index - start <= max_gap:
            smoothed[start:index] = [True] * (index - start)
    return smoothed


def _true_runs(values: Sequence[bool], *, start: int = 0) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = start
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        run_start = index
        while index < len(values) and values[index]:
            index += 1
        runs.append((run_start, index - 1))
    return runs


def _merge_nearby_runs(
    runs: list[tuple[int, int]],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _slot_geometry(
    total_frames: int,
    expected_revisits: int,
    buffer_frames: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    slots = tuple((indices[0], indices[-1] + 1) for indices in _balanced_ranges(0, total_frames, expected_revisits))
    boundaries = tuple(end for _, end in slots)
    if expected_revisits == 1:
        return slots, boundaries, ((0, total_frames),)

    effective_buffer = min(
        buffer_frames,
        min(end - start for start, end in slots) // 2,
    )
    windows = tuple(
        (
            max(slot_start, slot_end - effective_buffer),
            min(total_frames, slot_end + effective_buffer),
        )
        for slot_start, slot_end in slots
    )
    return slots, boundaries, windows


def _overlap_length(
    component: tuple[int, int],
    window: tuple[int, int],
) -> int:
    start, end = component
    window_start, window_end = window
    return max(0, min(end + 1, window_end) - max(start, window_start))


def _boundary_distance(component: tuple[int, int], boundary: int) -> int:
    start, end = component
    if start <= boundary <= end + 1:
        return 0
    return min(abs(start - boundary), abs(end + 1 - boundary))


def _is_monotonic_assignment(
    assignments: dict[int, int],
) -> bool:
    components = [assignments[index] for index in sorted(assignments)]
    return components == sorted(components)


def _assign_components(
    components: Sequence[tuple[int, int]],
    *,
    boundaries: Sequence[int],
    windows: Sequence[tuple[int, int]],
    min_overlap_frames: int,
) -> dict[int, int]:
    """Map revisit boundaries to ordered visible components.

    Boundary-window overlap is authoritative.  Remaining post-departure
    components are matched to the nearest free boundaries in monotonic order.
    The initial component is never force-matched: without boundary overlap it
    remains the full V reference.
    """

    assignments: dict[int, int] = {}
    last_component = 0
    for boundary_index, (boundary, window) in enumerate(zip(boundaries, windows)):
        candidates = []
        for component_index in range(last_component, len(components)):
            component = components[component_index]
            overlap = _overlap_length(component, window)
            if overlap < min_overlap_frames:
                continue
            candidates.append(
                (
                    overlap,
                    -_boundary_distance(component, boundary),
                    -component_index,
                    component_index,
                )
            )
        if candidates:
            component_index = max(candidates)[-1]
            assignments[boundary_index] = component_index
            last_component = component_index

    used_components = set(assignments.values())
    unmatched_post = [index for index in range(1, len(components)) if index not in used_components]
    free_boundaries = [index for index in range(len(boundaries)) if index not in assignments]
    pair_count = min(len(unmatched_post), len(free_boundaries))
    if not pair_count:
        return assignments

    best: tuple | None = None
    for component_indices in combinations(unmatched_post, pair_count):
        for boundary_indices in combinations(free_boundaries, pair_count):
            proposed = dict(assignments)
            proposed.update(zip(boundary_indices, component_indices))
            if not _is_monotonic_assignment(proposed):
                continue
            distances = tuple(
                _boundary_distance(
                    components[component_index],
                    boundaries[boundary_index],
                )
                for boundary_index, component_index in zip(
                    boundary_indices,
                    component_indices,
                )
            )
            total_length = sum(components[index][1] - components[index][0] + 1 for index in component_indices)
            key = (
                sum(distances),
                max(distances),
                -total_length,
                boundary_indices,
                component_indices,
            )
            if best is None or key < best[0]:
                best = (key, proposed)
    return best[1] if best is not None else assignments


def _flatten_revisits(
    revisits: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(index for revisit in revisits for index in revisit)


def phase_from_revisit_runs(
    *,
    target_id: int,
    visible_indices: tuple[int, ...],
    total_frames: int,
    runs: list[tuple[int, int]],
    expected_revisits: int | None,
    revisit_boundary_buffer_frames: int,
) -> TargetPhase:
    """Allocate initial visibility and post-departure runs to R1..RK."""

    if not visible_indices or visible_indices[0] != 0:
        raise ValueError("visible_indices must be a non-empty frame-0 run")
    initial_component = (0, visible_indices[-1])
    post_components = sorted(
        (max(start, initial_component[1] + 1), end) for start, end in runs if end > initial_component[1]
    )
    post_components = [run for run in post_components if run[0] <= run[1]]
    if expected_revisits is None:
        # Keep observed return components intact, without fixed-time boundary
        # allocation. The reference's no-departure/no-return scoring stays intact.
        not_departed = initial_component[1] == total_frames - 1
        reference = (0,) if not_departed else visible_indices
        revisits = (
            (tuple(range(1, total_frames)),)
            if not_departed
            else tuple(tuple(range(start, end + 1)) for start, end in post_components)
        )
        returned = _flatten_revisits(revisits)
        used = set(reference) | set(returned)
        return TargetPhase(
            target_id=target_id,
            visible_indices=reference,
            departure_indices=tuple(index for index in range(total_frames) if index not in used),
            return_indices=returned,
            status="not_departed" if not_departed else "ok" if returned else "not_returned",
            revisit_indices=revisits,
            revisit_sources=tuple("uniform_not_departed" if not_departed else "observed_cluster" for _ in revisits),
            revisit_departure_evidence=tuple(not not_departed for _ in revisits),
        )
    components = [initial_component, *post_components]
    slots, boundaries, windows = _slot_geometry(
        total_frames,
        expected_revisits,
        revisit_boundary_buffer_frames,
    )

    if expected_revisits == 1:
        if post_components:
            component_index = min(
                range(1, len(components)),
                key=lambda index: (
                    _boundary_distance(components[index], boundaries[0]),
                    -components[index][1],
                ),
            )
            assignments = {0: component_index}
        elif initial_component[1] == total_frames - 1:
            assignments = {0: 0}
        else:
            assignments = {}
    else:
        assignments = _assign_components(
            components,
            boundaries=boundaries,
            windows=windows,
            min_overlap_frames=MIN_REAPPEAR_MATCH_RUN,
        )

    component_boundaries: dict[int, list[int]] = {}
    for boundary_index, component_index in assignments.items():
        component_boundaries.setdefault(component_index, []).append(boundary_index)

    revisits: list[tuple[int, ...]] = [() for _ in range(expected_revisits)]
    sources = ["missing" for _ in range(expected_revisits)]
    departure_evidence = [False for _ in range(expected_revisits)]
    initial_carries_revisit = 0 in component_boundaries
    for component_index, assigned_boundaries in component_boundaries.items():
        assigned_boundaries.sort()
        start, end = components[component_index]
        if component_index == 0:
            start = max(1, start)
        segments = _balanced_ranges(start, end + 1, len(assigned_boundaries))
        for local_index, (boundary_index, segment) in enumerate(zip(assigned_boundaries, segments)):
            revisits[boundary_index] = segment
            if component_index == 0:
                sources[boundary_index] = (
                    "uniform_not_departed"
                    if initial_component[1] == total_frames - 1
                    else "continuous_presence_until_departure"
                )
            else:
                sources[boundary_index] = "observed_cluster" if local_index == 0 else "continuous_presence_split"
                departure_evidence[boundary_index] = local_index == 0

    reference_indices = (
        (0,) if initial_carries_revisit else tuple(range(initial_component[0], initial_component[1] + 1))
    )
    return_indices = _flatten_revisits(revisits)
    used = set(reference_indices) | set(return_indices)
    departure_indices = tuple(index for index in range(total_frames) if index not in used)
    if initial_component[1] == total_frames - 1:
        status = "not_departed"
    elif return_indices:
        status = "ok"
    else:
        status = "not_returned"
    return TargetPhase(
        target_id=target_id,
        visible_indices=reference_indices,
        departure_indices=departure_indices,
        return_indices=return_indices,
        status=status,
        revisit_indices=tuple(revisits),
        revisit_slots=slots,
        revisit_nominal_boundaries=boundaries,
        revisit_search_windows=windows,
        revisit_sources=tuple(sources),
        revisit_departure_evidence=tuple(departure_evidence),
    )


def infer_target_phase(
    *,
    target_id: int,
    total_frames: int,
    matched_frame_indices: Collection[int],
    max_short_missed_frames: int = MAX_SHORT_MISSED_FRAMES,
    min_reappear_match_run: int = MIN_REAPPEAR_MATCH_RUN,
    max_revisit_gap_frames: int = MAX_SHORT_MISSED_FRAMES,
    expected_revisits: int | None = None,
    revisit_boundary_buffer_frames: int | None = None,
) -> TargetPhase:
    """Infer V/D/R1..RK from one modality-specific presence sequence."""

    if max_short_missed_frames < 0 or max_revisit_gap_frames < 0:
        raise ValueError("missed-frame gap limits must be non-negative")
    if min_reappear_match_run < 1:
        raise ValueError("min_reappear_match_run must be positive")
    validate_expected_revisits(expected_revisits)
    buffer_frames = (
        REVISIT_BOUNDARY_BUFFER_FRAMES if revisit_boundary_buffer_frames is None else revisit_boundary_buffer_frames
    )
    if buffer_frames < 0:
        raise ValueError("revisit_boundary_buffer_frames must be non-negative")

    matched = {int(index) for index in matched_frame_indices}
    matched.add(0)
    presence = [index in matched for index in range(total_frames)]
    smoothed = _close_short_internal_gaps(
        presence,
        max_gap=max_short_missed_frames,
    )
    initial_runs = _true_runs(smoothed)
    if not initial_runs or initial_runs[0][0] != 0:
        raise AssertionError("the seeded first-frame presence run is missing")
    visible_end = initial_runs[0][1]
    visible_indices = tuple(range(visible_end + 1))
    post_runs = [
        (start, end)
        for start, end in _true_runs(presence, start=visible_end + 1)
        if end - start + 1 >= min_reappear_match_run
    ]
    post_runs = _merge_nearby_runs(post_runs, max_gap=max_revisit_gap_frames)
    return phase_from_revisit_runs(
        target_id=target_id,
        visible_indices=visible_indices,
        total_frames=total_frames,
        runs=post_runs,
        expected_revisits=expected_revisits,
        revisit_boundary_buffer_frames=buffer_frames,
    )
