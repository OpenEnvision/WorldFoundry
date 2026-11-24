# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Typed records for Object Entity evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..revisit_protocol import TargetPhase


@dataclass(frozen=True)
class ObjectEntityProtocol:
    total_frames: int
    condition_image_path: str
    target_mask_path: str
    target_description: str = ""
    expected_revisits: int | None = None

    @property
    def required_frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.total_frames))


@dataclass(frozen=True)
class ObjectTrackObservation:
    frame_index: int
    mask: np.ndarray | None = field(repr=False)
    presence_logit: float | None
    presence_probability: float
    mask_area_ratio: float
    mask_valid: bool
    present: bool


@dataclass(frozen=True)
class ObjectRedetectionPrompt:
    frame_index: int
    box_xyxy: tuple[float, float, float, float]
    detector_score: float
    reid_similarity: float = 0.0
    runner_up_reid_similarity: float | None = None
    candidate_count: int = 1


@dataclass
class ObjectTrackAnalysis:
    protocol: ObjectEntityProtocol
    phase: TargetPhase
    observations: dict[int, ObjectTrackObservation]
    decoded_frames: dict[int, np.ndarray] = field(repr=False)
    video_path: str = ""
    alignment_method: str = ""
    frame0_alignment_score: float = 0.0
    redetection_prompts: tuple[ObjectRedetectionPrompt, ...] = ()
    visibility_segments: tuple[tuple[int, int], ...] = ()
    max_revisit_gap_frames: int = 0


@dataclass(frozen=True)
class ObjectAppearanceResult:
    phase: TargetPhase
    frame_scores: dict[int, float]
    similarity: float
    appearance_coverage: float
    n_reference_features: int
    n_return_features: int
    revisit_scores: tuple[float, ...]
    revisit_coverages: tuple[float, ...]
    n_revisit_features: tuple[int, ...]


@dataclass
class ObjectAppearanceAnalysis:
    track: ObjectTrackAnalysis
    result: ObjectAppearanceResult


@dataclass(frozen=True)
class ObjectGeometryResult:
    phase: TargetPhase
    reference_frame_indices: tuple[int, ...]
    frame_scores: dict[int, float]
    similarity: float
    geometry_coverage: float
    area_ratio_drift: float
    n_reference_masks: int
    n_return_masks: int
    revisit_scores: tuple[float, ...]
    revisit_coverages: tuple[float, ...]
    n_revisit_masks: tuple[int, ...]
    revisit_area_ratio_drifts: tuple[float, ...]
