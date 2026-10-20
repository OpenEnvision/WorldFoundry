# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Planned five-phase protocol for the single-revisit LR-hold trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)

LR_HOLD_TRAJECTORY_ID = "lr_hold_sym_0p5dpf"
LR_HOLD_ACTIONS = ("hold", "yaw_left", "hold", "yaw_right", "hold")


@dataclass(frozen=True)
class FrameRange:
    """Inclusive output-frame range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid inclusive frame range: {self.start}..{self.end}")

    def contains(self, frame_index: int) -> bool:
        return self.start <= frame_index <= self.end

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class PromptAnchors:
    """Sampled-image anchors supplied to the VLM rather than inferred by it."""

    frame_map: tuple[dict[str, Any], ...]
    departure_baseline_images: tuple[int, ...]
    away_hold_images: tuple[int, ...]
    returned_hold_images: tuple[int, ...]
    returned_before_final_images: tuple[int, ...]
    final_stationary_images: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "departure_baseline_images": list(self.departure_baseline_images),
            "away_hold_images": list(self.away_hold_images),
            "returned_hold_images": list(self.returned_hold_images),
            "returned_before_final_images": list(self.returned_before_final_images),
            "final_stationary_images": list(self.final_stationary_images),
        }

    def phase_ranges(self) -> dict[str, list[int]]:
        ranges: dict[str, list[int]] = {}
        for phase in (
            "initial_hold",
            "yaw_left",
            "away_hold",
            "yaw_right",
            "returned_hold",
        ):
            numbers = [int(value["image_number"]) for value in self.frame_map if value["phase"] == phase]
            if numbers:
                ranges[phase] = [numbers[0], numbers[-1]]
        return ranges

    def compact_ranges(self) -> dict[str, list[int]]:
        def bounds(numbers: tuple[int, ...]) -> list[int]:
            return [numbers[0], numbers[-1]]

        return {
            "departure_baseline": bounds(self.departure_baseline_images),
            "away_hold": bounds(self.away_hold_images),
            "returned_hold": bounds(self.returned_hold_images),
            "returned_before_final": bounds(self.returned_before_final_images),
            "final_stationary": bounds(self.final_stationary_images),
        }


@dataclass(frozen=True)
class LRHoldProtocol:
    """Model-native planned LR-hold phases mapped onto one output video."""

    trajectory_id: str
    native_trajectory_id: str
    phase_transitions: tuple[int, int, int, int, int]
    tail_hold_transitions: int
    declared_output_frames: int
    actual_output_frames: int
    initial_hold: FrameRange
    yaw_left: FrameRange
    away_hold: FrameRange
    yaw_right: FrameRange
    returned_hold: FrameRange

    @classmethod
    def from_metadata(
        cls,
        value: Any,
        *,
        actual_output_frames: int,
    ) -> "LRHoldProtocol":
        if not isinstance(value, dict):
            raise DataInvalidError("metadata.lr_hold_protocol must be an object")
        trajectory_id = str(value.get("trajectory_id", ""))
        if trajectory_id != LR_HOLD_TRAJECTORY_ID:
            raise DataInvalidError(f"Dynamic Memory requires {LR_HOLD_TRAJECTORY_ID}, got {trajectory_id!r}")
        native_trajectory_id = str(value.get("native_trajectory_id", ""))
        if not (
            native_trajectory_id == LR_HOLD_TRAJECTORY_ID
            or native_trajectory_id.startswith(LR_HOLD_TRAJECTORY_ID + "_action")
        ):
            raise DataInvalidError(f"invalid native LR-hold trajectory: {native_trajectory_id!r}")
        raw_actions = value.get("actions")
        if tuple(raw_actions or ()) != LR_HOLD_ACTIONS:
            raise DataInvalidError(f"LR-hold actions must be {LR_HOLD_ACTIONS}, got {raw_actions!r}")
        raw_counts = value.get("phase_transitions")
        if (
            not isinstance(raw_counts, list)
            or len(raw_counts) != 5
            or any(isinstance(count, bool) or not isinstance(count, int) for count in raw_counts)
            or any(count <= 0 for count in raw_counts)
        ):
            raise DataInvalidError("lr_hold_protocol.phase_transitions must contain five positive integers")
        counts = tuple(raw_counts)
        tail_hold = value.get("tail_hold_transitions", 0)
        declared_frames = value.get("declared_output_frames")
        if isinstance(tail_hold, bool) or not isinstance(tail_hold, int) or tail_hold < 0:
            raise DataInvalidError("tail_hold_transitions must be a non-negative integer")
        if isinstance(declared_frames, bool) or not isinstance(declared_frames, int) or declared_frames < 2:
            raise DataInvalidError("declared_output_frames must be an integer >= 2")
        if actual_output_frames < 2:
            raise EvaluationError("LR-hold video must contain at least two frames")
        if sum(counts) + tail_hold != declared_frames - 1:
            raise DataInvalidError("phase transitions and tail hold do not cover declared output frames")

        native_boundaries = []
        cumulative = 0
        for count in counts[:4]:
            cumulative += count
            native_boundaries.append(cumulative)

        def map_frame(index: int) -> int:
            return round(index * (actual_output_frames - 1) / (declared_frames - 1))

        b1, b2, b3, b4 = (map_frame(value) for value in native_boundaries)
        last = actual_output_frames - 1
        if not (0 < b1 < b2 < b3 < b4 < last):
            raise EvaluationError(
                f"mapped LR-hold phase boundaries are not strictly increasing: {(b1, b2, b3, b4, last)}"
            )
        return cls(
            trajectory_id=trajectory_id,
            native_trajectory_id=native_trajectory_id,
            phase_transitions=counts,  # type: ignore[arg-type]
            tail_hold_transitions=tail_hold,
            declared_output_frames=declared_frames,
            actual_output_frames=actual_output_frames,
            initial_hold=FrameRange(0, b1),
            yaw_left=FrameRange(b1 + 1, b2 - 1),
            away_hold=FrameRange(b2, b3),
            yaw_right=FrameRange(b3 + 1, b4 - 1),
            returned_hold=FrameRange(b4, last),
        )

    def phase_for_frame(self, frame_index: int) -> str:
        if not 0 <= frame_index < self.actual_output_frames:
            raise ValueError(f"frame {frame_index} outside [0, {self.actual_output_frames - 1}]")
        if self.initial_hold.contains(frame_index):
            return "initial_hold"
        if self.yaw_left.contains(frame_index):
            return "yaw_left"
        if self.away_hold.contains(frame_index):
            return "away_hold"
        if self.yaw_right.contains(frame_index):
            return "yaw_right"
        return "returned_hold"

    def prompt_anchors(self, sampled_frame_indices: Iterable[int]) -> PromptAnchors:
        indices = tuple(int(index) for index in sampled_frame_indices)
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("sampled frame indices must be sorted and unique")
        frame_map = tuple(
            {
                "image_number": image_number,
                "source_frame_index": frame_index,
                "phase": self.phase_for_frame(frame_index),
            }
            for image_number, frame_index in enumerate(indices, start=1)
        )

        def image_numbers(phase: str) -> tuple[int, ...]:
            return tuple(int(value["image_number"]) for value in frame_map if value["phase"] == phase)

        initial = image_numbers("initial_hold")
        away = image_numbers("away_hold")
        returned = image_numbers("returned_hold")
        if len(initial) < 3 or len(away) < 1 or len(returned) < 4:
            raise EvaluationError("stride sampling does not preserve enough LR-hold anchor images")
        return PromptAnchors(
            frame_map=frame_map,
            departure_baseline_images=initial[-3:],
            away_hold_images=away,
            returned_hold_images=returned,
            returned_before_final_images=returned[:-3],
            final_stationary_images=returned[-3:],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "native_trajectory_id": self.native_trajectory_id,
            "phase_transitions": list(self.phase_transitions),
            "tail_hold_transitions": self.tail_hold_transitions,
            "declared_output_frames": self.declared_output_frames,
            "actual_output_frames": self.actual_output_frames,
            "initial_hold": self.initial_hold.as_dict(),
            "yaw_left": self.yaw_left.as_dict(),
            "away_hold": self.away_hold.as_dict(),
            "yaw_right": self.yaw_right.as_dict(),
            "returned_hold": self.returned_hold.as_dict(),
        }
