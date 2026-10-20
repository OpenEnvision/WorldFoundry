# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Typed records shared by the Human Entity Memory metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.revisit_protocol import TargetPhase


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_value(cls, value: Any, *, field_name: str) -> BoundingBox:
        import numpy as np

        if isinstance(value, dict):
            try:
                coordinates = [value[key] for key in ("x1", "y1", "x2", "y2")]
            except KeyError as error:
                raise ValueError(f"{field_name} bbox mappings require x1/y1/x2/y2") from error
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            coordinates = list(value)
        else:
            raise ValueError(f"{field_name} must contain [x1, y1, x2, y2] boxes")
        try:
            box = cls(*(float(coordinate) for coordinate in coordinates))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} contains a non-numeric bbox") from error
        if not np.isfinite(box.as_array()).all():
            raise ValueError(f"{field_name} contains a non-finite bbox")
        if box.x2 <= box.x1 or box.y2 <= box.y1:
            raise ValueError(f"{field_name} contains an empty or inverted bbox")
        return box

    def as_array(self) -> np.ndarray:
        import numpy as np

        return np.asarray([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clipped(self, *, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            x1=max(0.0, min(float(width), self.x1)),
            y1=max(0.0, min(float(height), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )


@dataclass(frozen=True)
class HumanEntityProtocol:
    total_frames: int
    target_face_boxes: tuple[BoundingBox, ...]
    target_descriptions: tuple[str, ...]
    expected_revisits: int | None

    @property
    def n_targets(self) -> int:
        return len(self.target_face_boxes)

    @property
    def required_frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.total_frames))


@dataclass
class BodyTrackAnalysis:
    target_phases: tuple[TargetPhase, ...]
    assignments: dict[int, dict[int, BoundingBox]]


@dataclass(frozen=True)
class FaceObservation:
    bbox: BoundingBox
    embedding: np.ndarray


@dataclass
class FaceTrackAnalysis:
    anchors: tuple[np.ndarray, ...]
    seeded_association: dict[int, FaceObservation]
    candidates_by_frame: dict[int, list[FaceObservation]]
    track_associations: dict[int, dict[int, FaceObservation]]


@dataclass(frozen=True)
class IdentityTargetResult:
    target_id: int
    phase: TargetPhase
    frame_scores: dict[int, float]
    similarity: float
    coverage: float
    n_reference_embeddings: int
    n_return_embeddings: int
    revisit_scores: tuple[float, ...]
    revisit_coverages: tuple[float, ...]
    n_revisit_embeddings: tuple[int, ...]


@dataclass
class IdentityAnalysis:
    protocol: HumanEntityProtocol
    target_phases: tuple[TargetPhase, ...]
    appearance_target_phases: tuple[TargetPhase, ...]
    body_assignments: dict[int, dict[int, BoundingBox]]
    target_results: tuple[IdentityTargetResult, ...]
    max_revisit_gap_frames: int
    revisit_boundary_buffer_frames: int
    decoded_frames: dict[int, np.ndarray] = field(repr=False)
    video_path: str = ""


@dataclass(frozen=True)
class AppearanceTargetResult:
    target_id: int
    phase: TargetPhase
    frame_scores: dict[int, float]
    score: float
    reappearance_coverage: float
    n_reference_features: int
    n_return_features: int
    revisit_scores: tuple[float, ...]
    revisit_coverages: tuple[float, ...]
    n_revisit_features: tuple[int, ...]


@dataclass(frozen=True)
class AppearanceAnalysis:
    target_results: tuple[AppearanceTargetResult, ...]
