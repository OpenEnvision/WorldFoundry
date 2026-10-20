# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Single-target Object Appearance and Geometry with shared tracking."""

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.aggregation import (
    aggregate_video_summary,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    ResultStatus,
    UnitResult,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from ..revisit_protocol import MAX_SHORT_MISSED_FRAMES, MIN_REAPPEAR_MATCH_RUN, REVISIT_BOUNDARY_BUFFER_FRAMES
from .evaluator import object_appearance_analysis_for_item, object_track_analysis_for_item
from .geometry_consistency import DEFAULT_CANONICAL_PADDING, DEFAULT_CANONICAL_SIZE, analyze_object_geometry
from .object_tracker import (
    MAX_MASK_AREA_RATIO,
    MIN_MASK_AREA_RATIO,
    MIN_MASK_AREA_RELATIVE_TO_SEED,
    PRESENCE_LOGIT_THRESHOLD,
)
from .revisit import official_revisit_score, revisit_metadata, revisit_score_fields


def _frame_boundary_contact_ratio(track, indices) -> float:
    contacts = []
    for index in indices:
        observation = track.observations[index]
        if not observation.present or observation.mask is None:
            continue
        mask = observation.mask
        contacts.append(bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
    return float(sum(contacts) / len(contacts)) if contacts else 0.0


class _ObjectMetric(Metric):
    geometry = False

    def __init__(self, *, tracker=None, redetector=None, feature_extractor=None, reader=None):
        self.tracker = tracker
        self.redetector = redetector
        self.feature_extractor = feature_extractor
        self.reader = reader or VideoReader()

    def evaluate(self, sample: dict, context) -> list[UnitResult]:
        if sample.get("category") not in (None, "object"):
            return [UnitResult(unit_id="target_0000", status=ResultStatus.NOT_APPLICABLE)]
        error = sample.get("object_preparation_error")
        if error:
            exception = DataInvalidError if error["status"] == "data_invalid" else EvaluationError
            raise exception(error["message"])
        kwargs = dict(tracker=self.tracker, redetector=self.redetector, reader=self.reader)
        extra_scores, extra_metadata = {}, {}
        if self.geometry:
            track = object_track_analysis_for_item(sample, context, weight_metric=self.name, **kwargs)
            config = context.metric_config(self.name)
            size = int(config.get("canonical_canvas_size", DEFAULT_CANONICAL_SIZE))
            padding = int(config.get("canonical_padding", DEFAULT_CANONICAL_PADDING))
            result = analyze_object_geometry(track, canonical_size=size, canonical_padding=padding)
            coverage = result.geometry_coverage
            counts = result.n_revisit_masks
            count_key = "n_return_masks"
            extra_scores = {
                "area_ratio_drift": result.area_ratio_drift,
                "frame_boundary_contact_ratio": _frame_boundary_contact_ratio(track, result.phase.return_indices),
            }
            extra_metadata = {
                "reference_frame_indices": list(result.reference_frame_indices),
                "n_reference_masks": result.n_reference_masks,
                "n_return_masks": result.n_return_masks,
                "geometry_protocol": {
                    "canonical_canvas_size": size,
                    "canonical_padding": padding,
                    "alignment": "bbox_crop_center_uniform_scale",
                    "reference_matching": "max_iou_over_frozen_v_library",
                    "aggregation": "mean_valid_frames_per_revisit_macro",
                },
            }
        else:
            analysis = object_appearance_analysis_for_item(
                sample, context, feature_extractor=self.feature_extractor, **kwargs
            )
            track, result = analysis.track, analysis.result
            coverage = result.appearance_coverage
            counts = result.n_revisit_features
            count_key = "n_return_features"
            extra_metadata = {
                "n_reference_features": result.n_reference_features,
                "n_return_features": result.n_return_features,
            }

        phase = result.phase
        expected = track.protocol.expected_revisits
        kind = "geometry" if self.geometry else "appearance"
        scores = {
            "score": official_revisit_score(result.revisit_scores, expected_revisits=expected),
            f"{kind}_similarity": result.similarity,
            f"{kind}_coverage": coverage,
            **extra_scores,
            **revisit_score_fields(result.revisit_scores, result.revisit_coverages, expected_revisits=expected),
        }
        revisits = revisit_metadata(
            phase,
            result.revisit_scores,
            result.revisit_coverages,
            counts,
            count_key=count_key,
            expected_revisits=expected,
        )
        if self.geometry:
            for index, revisit in enumerate(revisits):
                indices = phase.revisit_indices[index] if index < len(phase.revisit_indices) else ()
                revisit["frame_boundary_contact_ratio"] = _frame_boundary_contact_ratio(track, indices)
                revisit["area_ratio_drift"] = (
                    result.revisit_area_ratio_drifts[index] if index < len(result.revisit_area_ratio_drifts) else 0.0
                )
        phase_roles = {index: "V" for index in phase.visible_indices}
        for index, indices in enumerate(phase.revisit_indices, 1):
            phase_roles.update({frame: f"R{index}" for frame in indices})
        events = []
        for prompt in track.redetection_prompts:
            outcome = "rejected_or_unstable"
            for index, (start, end) in enumerate(track.visibility_segments):
                if start <= prompt.frame_index <= end:
                    outcome = "already_tracked" if index == 0 else "accepted_reacquisition"
                    break
            events.append(
                {
                    "frame_index": prompt.frame_index,
                    "box_xyxy": list(prompt.box_xyxy),
                    "detector_score": prompt.detector_score,
                    "reid_similarity": prompt.reid_similarity,
                    "runner_up_reid_similarity": prompt.runner_up_reid_similarity,
                    "candidate_count": prompt.candidate_count,
                    "phase_role": phase_roles.get(prompt.frame_index, "D"),
                    "tracking_outcome": outcome,
                }
            )
        target = {
            "target_id": 0,
            "phase_status": phase.status,
            "V": list(phase.visible_indices),
            "D": list(phase.departure_indices),
            "R": list(phase.return_indices),
            "revisits": revisits,
            f"{kind}_score": scores["score"],
            f"{kind}_similarity": result.similarity,
            f"{kind}_coverage": coverage,
            **extra_scores,
            **extra_metadata,
        }
        return [
            UnitResult(
                unit_id="target_0000",
                unit_type=f"target_object_{kind}",
                scores=scores,
                metadata={
                    "video_path": track.video_path,
                    "n_targets": 1,
                    "condition_image_path": track.protocol.condition_image_path,
                    "target_mask_path": track.protocol.target_mask_path,
                    "target_description": track.protocol.target_description,
                    "expected_revisits": expected,
                    "revisit_mode": "auto" if expected is None else "fixed",
                    "alignment_method": track.alignment_method,
                    "frame0_alignment_score": track.frame0_alignment_score,
                    "phase_source": "grounding_dino_clip_redetection_with_multitracklet_sam2",
                    "phase_inference": {
                        "presence_logit_threshold": PRESENCE_LOGIT_THRESHOLD,
                        "min_mask_area_ratio": MIN_MASK_AREA_RATIO,
                        "max_mask_area_ratio": MAX_MASK_AREA_RATIO,
                        "min_mask_area_relative_to_seed": MIN_MASK_AREA_RELATIVE_TO_SEED,
                        "max_short_missed_frames": MAX_SHORT_MISSED_FRAMES,
                        "min_reappear_match_run": MIN_REAPPEAR_MATCH_RUN,
                        "max_revisit_gap_frames": track.max_revisit_gap_frames,
                        "revisit_allocator": "observed_return_components"
                        if expected is None
                        else "boundary_presence_components",
                        "revisit_boundary_buffer_frames": None if expected is None else REVISIT_BOUNDARY_BUFFER_FRAMES,
                    },
                    "visibility_segments": [list(segment) for segment in track.visibility_segments],
                    "redetection_events": events,
                    "per_target": [target],
                },
            )
        ]

    def aggregate_item(self, sample, units):
        if len(units) != 1:
            raise EvaluationError(f"{self.name} requires exactly one target unit")
        unit = units[0]
        return ItemResult(
            item_id=sample["item_id"],
            model_id=sample.get("model_id"),
            scores=dict(unit.scores),
            num_units=1,
            status=unit.status,
            error=unit.error,
            metadata=dict(unit.metadata),
        )

    def aggregate_summary(self, items):
        return aggregate_video_summary(items)


class ObjectAppearanceMetric(_ObjectMetric):
    name = "object_appearance"
    description = "Masked DINOv2-Large appearance consistency across object revisits."


class ObjectGeometryMetric(_ObjectMetric):
    name = "object_geometry"
    description = "Canonical silhouette IoU across object revisits, preserving the reference shape score."
    geometry = True
