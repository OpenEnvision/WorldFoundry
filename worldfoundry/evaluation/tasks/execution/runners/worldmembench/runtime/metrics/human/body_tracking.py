# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Target-specific body phases with frozen-face reacquisition and local SAM2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.revisit_protocol import (
    MAX_SHORT_MISSED_FRAMES,
    MIN_REAPPEAR_MATCH_RUN,
    REVISIT_BOUNDARY_BUFFER_FRAMES,
    _close_short_internal_gaps,
    _merge_nearby_runs,
    _true_runs,
    infer_target_phase,
    phase_from_revisit_runs,
)

from .face_identity import (
    _frame_matches,
    build_frozen_reference_libraries,
)
from .person_mask_tracker import MaskPropagator, PersonDetector
from .schemas import (
    BodyTrackAnalysis,
    BoundingBox,
    FaceObservation,
    FaceTrackAnalysis,
    HumanEntityProtocol,
    TargetPhase,
)
from .target_association import assign_faces_to_person_boxes, hungarian_matches

BODY_MASK_BOX_IOU_THRESHOLD = 0.1
BODY_REACQUIRE_PROMPT_STRIDE = 8
BODY_REACQUISITION_FACE_THRESHOLD = 0.35


class BodyTrackerBackend(Protocol):
    def analyze(
        self,
        protocol: HumanEntityProtocol,
        decoded_frames: dict[int, np.ndarray],
        *,
        max_revisit_gap_frames: int,
        revisit_boundary_buffer_frames: int = REVISIT_BOUNDARY_BUFFER_FRAMES,
        face_tracks: FaceTrackAnalysis | None = None,
        face_reacquisition_threshold: float = BODY_REACQUISITION_FACE_THRESHOLD,
    ) -> BodyTrackAnalysis: ...


def _mask_bbox(mask: np.ndarray) -> BoundingBox | None:
    y_coordinates, x_coordinates = np.where(mask)
    if not len(x_coordinates):
        return None
    return BoundingBox(
        x1=float(x_coordinates.min()),
        y1=float(y_coordinates.min()),
        x2=float(x_coordinates.max() + 1),
        y2=float(y_coordinates.max() + 1),
    )


def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


def _initial_phase(
    *,
    target_id: int,
    total_frames: int,
    assignments: dict[int, dict[int, BoundingBox]],
) -> TargetPhase:
    presence = [
        frame_index == 0 or target_id in assignments.get(frame_index, {}) for frame_index in range(total_frames)
    ]
    smoothed = _close_short_internal_gaps(
        presence,
        max_gap=MAX_SHORT_MISSED_FRAMES,
    )
    initial_runs = _true_runs(smoothed)
    visible_end = initial_runs[0][1]
    visible = tuple(range(visible_end + 1))
    if visible_end == total_frames - 1:
        return TargetPhase(
            target_id=target_id,
            visible_indices=visible,
            departure_indices=(),
            return_indices=(),
            status="not_departed",
        )
    return TargetPhase(
        target_id=target_id,
        visible_indices=visible,
        departure_indices=tuple(range(visible_end + 1, total_frames)),
        return_indices=(),
        status="not_returned",
    )


def _stable_face_runs(
    *,
    target_id: int,
    visible_end: int,
    total_frames: int,
    matches_by_frame: dict[int, dict[int, FaceObservation]],
    max_revisit_gap_frames: int,
) -> list[tuple[int, int]]:
    presence = [target_id in matches_by_frame.get(frame_index, {}) for frame_index in range(total_frames)]
    runs = [
        (start, end)
        for start, end in _true_runs(presence, start=visible_end + 1)
        if end - start + 1 >= MIN_REAPPEAR_MATCH_RUN
    ]
    runs = _merge_nearby_runs(
        runs,
        max_gap=max_revisit_gap_frames,
    )
    return runs


def _segment_bounds(
    runs: Sequence[tuple[int, int]],
    *,
    post_start: int,
    final_frame: int,
) -> list[tuple[int, int]]:
    bounds = []
    for index, (start, end) in enumerate(runs):
        segment_start = post_start if index == 0 else (runs[index - 1][1] + start) // 2 + 1
        segment_end = final_frame if index == len(runs) - 1 else (end + runs[index + 1][0]) // 2
        bounds.append((segment_start, segment_end))
    return bounds


def _sparse_prompts(
    assignments: dict[int, BoundingBox],
    *,
    start: int,
    end: int,
) -> list[tuple[int, BoundingBox]]:
    frames = sorted(frame_index for frame_index in assignments if start <= frame_index <= end)
    if not frames:
        return []
    selected = [frames[0]]
    for frame_index in frames[1:-1]:
        if frame_index - selected[-1] >= BODY_REACQUIRE_PROMPT_STRIDE:
            selected.append(frame_index)
    if frames[-1] != selected[-1]:
        selected.append(frames[-1])
    return [(frame_index, assignments[frame_index]) for frame_index in selected]


def _anchored_component(
    candidate_frames: set[int],
    *,
    segment_start: int,
    segment_end: int,
    anchor_start: int,
    anchor_end: int,
    max_gap: int,
) -> tuple[int, int]:
    presence = [frame_index in candidate_frames for frame_index in range(segment_start, segment_end + 1)]
    smoothed = _close_short_internal_gaps(presence, max_gap=max_gap)
    components = [(segment_start + start, segment_start + end) for start, end in _true_runs(smoothed)]
    anchored = [(start, end) for start, end in components if not (end < anchor_start or start > anchor_end)]
    if not anchored:
        return anchor_start, anchor_end
    return (
        min(start for start, _ in anchored),
        max(end for _, end in anchored),
    )


class YoloSamBodyTracker:
    """Use initial SAM2 presence, then face-anchored independent local tracks."""

    def __init__(
        self,
        *,
        person_detector: PersonDetector,
        mask_propagator: MaskPropagator,
        mask_box_iou_threshold: float = BODY_MASK_BOX_IOU_THRESHOLD,
    ) -> None:
        self.person_detector = person_detector
        self.mask_propagator = mask_propagator
        self.mask_box_iou_threshold = mask_box_iou_threshold

    def _initial_tracks(
        self,
        protocol: HumanEntityProtocol,
        decoded_frames: dict[int, np.ndarray],
    ) -> tuple[
        dict[int, dict[int, BoundingBox]],
        dict[int, list[tuple[BoundingBox, float]]],
    ]:
        frame_indices = list(protocol.required_frame_indices)
        frames = [decoded_frames[index] for index in frame_indices]
        first_people = self.person_detector.detect(frames[0])
        people_by_frame = {0: first_people}
        first_assignments = assign_faces_to_person_boxes(
            {target_id: box for target_id, box in enumerate(protocol.target_face_boxes)},
            first_people,
        )
        masks = self.mask_propagator.track(
            frames,
            {target_id: [(0, box)] for target_id, box in first_assignments.items()},
        )
        assignments = {0: first_assignments}
        for local_index, frame_index in enumerate(frame_indices[1:], start=1):
            people = self.person_detector.detect(decoded_frames[frame_index])
            people_by_frame[frame_index] = people
            target_ids = []
            mask_boxes = []
            for target_id in range(protocol.n_targets):
                mask = masks.get(target_id, {}).get(local_index)
                box = _mask_bbox(mask) if mask is not None else None
                if box is not None:
                    target_ids.append(target_id)
                    mask_boxes.append(box)
            if not mask_boxes or not people:
                assignments[frame_index] = {}
                continue
            scores = np.asarray(
                [[_box_iou(mask_box, person_box) for person_box, _ in people] for mask_box in mask_boxes],
                dtype=np.float32,
            )
            matches = hungarian_matches(
                scores,
                threshold=self.mask_box_iou_threshold,
            )
            assignments[frame_index] = {target_ids[row]: people[column][0] for row, column in matches.items()}
        return assignments, people_by_frame

    def _face_reacquisition_matches(
        self,
        protocol: HumanEntityProtocol,
        face_tracks: FaceTrackAnalysis,
        initial_phases: Sequence[TargetPhase],
        *,
        face_reacquisition_threshold: float,
    ) -> dict[int, dict[int, FaceObservation]]:
        references = build_frozen_reference_libraries(
            face_tracks,
            initial_phases,
        )
        matches_by_frame: dict[int, dict[int, FaceObservation]] = {0: {}}
        for frame_index in range(1, protocol.total_frames):
            active_target_ids = [
                target_id
                for target_id, phase in enumerate(initial_phases)
                if (phase.status != "not_departed" and frame_index > phase.visible_indices[-1])
            ]
            reference_sets = [references[target_id] for target_id in active_target_ids]
            matches_by_frame[frame_index] = _frame_matches(
                reference_sets,
                face_tracks.candidates_by_frame[frame_index],
                threshold=face_reacquisition_threshold,
                target_ids=active_target_ids,
            )
        return matches_by_frame

    @staticmethod
    def _direct_person_assignments(
        *,
        stable_runs: Sequence[list[tuple[int, int]]],
        matches_by_frame: dict[int, dict[int, FaceObservation]],
        people_by_frame: dict[int, list[tuple[BoundingBox, float]]],
        total_frames: int,
    ) -> list[dict[int, BoundingBox]]:
        direct = [dict() for _ in stable_runs]
        for frame_index in range(total_frames):
            eligible_faces = {}
            for target_id, runs in enumerate(stable_runs):
                if any(start <= frame_index <= end for start, end in runs):
                    observation = matches_by_frame.get(frame_index, {}).get(target_id)
                    if observation is not None:
                        eligible_faces[target_id] = observation.bbox
            if not eligible_faces:
                continue
            mapped = assign_faces_to_person_boxes(
                eligible_faces,
                people_by_frame.get(frame_index, []),
            )
            for target_id, person_box in mapped.items():
                direct[target_id][frame_index] = person_box
        return direct

    def _local_revisit_tracks(
        self,
        *,
        target_id: int,
        visible_end: int,
        runs: Sequence[tuple[int, int]],
        direct_assignments: dict[int, BoundingBox],
        decoded_frames: dict[int, np.ndarray],
        people_by_frame: dict[int, list[tuple[BoundingBox, float]]],
        max_revisit_gap_frames: int,
    ) -> tuple[dict[int, BoundingBox], list[tuple[int, int]]]:
        viable_runs = [
            (start, end)
            for start, end in runs
            if any(start <= frame_index <= end for frame_index in direct_assignments)
        ]
        assignments = {}
        kept_runs = []
        bounds = _segment_bounds(
            viable_runs,
            post_start=visible_end + 1,
            final_frame=len(decoded_frames) - 1,
        )
        for (anchor_start, anchor_end), (segment_start, segment_end) in zip(
            viable_runs,
            bounds,
        ):
            global_prompts = _sparse_prompts(
                direct_assignments,
                start=anchor_start,
                end=anchor_end,
            )
            segment_frames = [decoded_frames[frame_index] for frame_index in range(segment_start, segment_end + 1)]
            masks = self.mask_propagator.track(
                segment_frames,
                {target_id: [(frame_index - segment_start, box) for frame_index, box in global_prompts]},
            ).get(target_id, {})
            candidates = {
                frame_index: box
                for frame_index, box in direct_assignments.items()
                if anchor_start <= frame_index <= anchor_end
            }
            for local_index, mask in masks.items():
                frame_index = segment_start + int(local_index)
                if frame_index < segment_start or frame_index > segment_end or frame_index in candidates:
                    continue
                mask_box = _mask_bbox(np.asarray(mask, dtype=bool))
                people = people_by_frame.get(frame_index, [])
                if mask_box is None or not people:
                    continue
                scores = [_box_iou(mask_box, person_box) for person_box, _ in people]
                best_index = int(np.argmax(scores))
                if scores[best_index] >= self.mask_box_iou_threshold:
                    candidates[frame_index] = people[best_index][0]
            run_start, run_end = _anchored_component(
                set(candidates),
                segment_start=segment_start,
                segment_end=segment_end,
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                max_gap=max_revisit_gap_frames,
            )
            assignments.update(
                {frame_index: box for frame_index, box in candidates.items() if run_start <= frame_index <= run_end}
            )
            kept_runs.append((run_start, run_end))
        return assignments, kept_runs

    def analyze(
        self,
        protocol: HumanEntityProtocol,
        decoded_frames: dict[int, np.ndarray],
        *,
        max_revisit_gap_frames: int,
        revisit_boundary_buffer_frames: int = REVISIT_BOUNDARY_BUFFER_FRAMES,
        face_tracks: FaceTrackAnalysis | None = None,
        face_reacquisition_threshold: float = BODY_REACQUISITION_FACE_THRESHOLD,
    ) -> BodyTrackAnalysis:
        initial_assignments, people_by_frame = self._initial_tracks(
            protocol,
            decoded_frames,
        )
        if face_tracks is None:
            inferred = tuple(
                infer_target_phase(
                    target_id=target_id,
                    total_frames=protocol.total_frames,
                    matched_frame_indices={
                        frame_index for frame_index, matched in initial_assignments.items() if target_id in matched
                    },
                    max_revisit_gap_frames=max_revisit_gap_frames,
                    expected_revisits=protocol.expected_revisits,
                    revisit_boundary_buffer_frames=(revisit_boundary_buffer_frames),
                )
                for target_id in range(protocol.n_targets)
            )
            return BodyTrackAnalysis(
                target_phases=inferred,
                assignments=initial_assignments,
            )

        initial_phases = tuple(
            _initial_phase(
                target_id=target_id,
                total_frames=protocol.total_frames,
                assignments=initial_assignments,
            )
            for target_id in range(protocol.n_targets)
        )
        matches_by_frame = self._face_reacquisition_matches(
            protocol,
            face_tracks,
            initial_phases,
            face_reacquisition_threshold=face_reacquisition_threshold,
        )
        stable_runs = [
            (
                []
                if phase.status == "not_departed"
                else _stable_face_runs(
                    target_id=target_id,
                    visible_end=phase.visible_indices[-1],
                    total_frames=protocol.total_frames,
                    matches_by_frame=matches_by_frame,
                    max_revisit_gap_frames=max_revisit_gap_frames,
                )
            )
            for target_id, phase in enumerate(initial_phases)
        ]
        direct_assignments = self._direct_person_assignments(
            stable_runs=stable_runs,
            matches_by_frame=matches_by_frame,
            people_by_frame=people_by_frame,
            total_frames=protocol.total_frames,
        )

        final_assignments = {frame_index: {} for frame_index in range(protocol.total_frames)}
        target_phases = []
        for target_id, initial_phase in enumerate(initial_phases):
            visible_end = initial_phase.visible_indices[-1]
            if initial_phase.status == "not_departed":
                for frame_index, matched in initial_assignments.items():
                    person_box = matched.get(target_id)
                    if person_box is not None:
                        final_assignments[frame_index][target_id] = person_box
                target_phases.append(
                    phase_from_revisit_runs(
                        target_id=target_id,
                        visible_indices=initial_phase.visible_indices,
                        total_frames=protocol.total_frames,
                        runs=[],
                        expected_revisits=protocol.expected_revisits,
                        revisit_boundary_buffer_frames=(revisit_boundary_buffer_frames),
                    )
                )
                continue

            for frame_index in initial_phase.visible_indices:
                person_box = initial_assignments.get(frame_index, {}).get(target_id)
                if person_box is not None:
                    final_assignments[frame_index][target_id] = person_box
            local_assignments, revisit_runs = self._local_revisit_tracks(
                target_id=target_id,
                visible_end=visible_end,
                runs=stable_runs[target_id],
                direct_assignments=direct_assignments[target_id],
                decoded_frames=decoded_frames,
                people_by_frame=people_by_frame,
                max_revisit_gap_frames=max_revisit_gap_frames,
            )
            for frame_index, person_box in local_assignments.items():
                final_assignments[frame_index][target_id] = person_box
            target_phases.append(
                phase_from_revisit_runs(
                    target_id=target_id,
                    visible_indices=initial_phase.visible_indices,
                    total_frames=protocol.total_frames,
                    runs=revisit_runs,
                    expected_revisits=protocol.expected_revisits,
                    revisit_boundary_buffer_frames=(revisit_boundary_buffer_frames),
                )
            )

        return BodyTrackAnalysis(
            target_phases=tuple(target_phases),
            assignments=final_assignments,
        )
