# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Human Identity and Human Appearance metric plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
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

if TYPE_CHECKING:
    from .appearance_features import AppearanceBackend
    from .body_tracking import BodyTrackerBackend
    from .face_identity import FaceIdentityBackend
from ..revisit_protocol import (
    MAX_EXPECTED_REVISITS,
    MIN_REAPPEAR_MATCH_RUN,
    validate_expected_revisits,
)
from .schemas import IdentityAnalysis, TargetPhase

_SHARED_ITEM_METADATA = (
    "video_path",
    "n_targets",
    "target_face_bboxes_first_frame",
    "target_descriptions",
    "expected_revisits",
    "revisit_mode",
    "first_frame_match_status",
    "first_frame_match_error",
    "phase_source",
    "phase_inference",
)
_SHARED_TARGET_METADATA = (
    "target_id",
    "phase_status",
    "V",
    "D",
    "R",
    "revisits",
)


def _identity_metadata(
    identity: IdentityAnalysis,
    *,
    phase_source: str,
) -> dict:
    from .body_tracking import BODY_MASK_BOX_IOU_THRESHOLD, BODY_REACQUISITION_FACE_THRESHOLD
    from .face_identity import TRACK_ASSOCIATION_THRESHOLD

    protocol = identity.protocol
    return {
        "video_path": identity.video_path,
        "n_targets": protocol.n_targets,
        "target_face_bboxes_first_frame": [box.as_list() for box in protocol.target_face_boxes],
        "target_descriptions": list(protocol.target_descriptions),
        "expected_revisits": protocol.expected_revisits,
        "revisit_mode": "auto" if protocol.expected_revisits is None else "fixed",
        "phase_source": phase_source,
        "phase_inference": {
            "body_mask_box_iou_threshold": BODY_MASK_BOX_IOU_THRESHOLD,
            "body_presence_model": "first_frame_face_to_yolo_person_then_sam2_track",
            "track_association_threshold": TRACK_ASSOCIATION_THRESHOLD,
            "body_reacquisition_face_threshold": (BODY_REACQUISITION_FACE_THRESHOLD),
            "max_revisit_gap_frames": identity.max_revisit_gap_frames,
            "revisit_boundary_buffer_frames": (identity.revisit_boundary_buffer_frames),
            "identity_revisit_model": "rolling_face_presence_frozen_v_cosine",
            "revisit_partition": (
                "observed_return_components"
                if protocol.expected_revisits is None
                else "boundary_buffer_d_insertion_components"
            ),
            "min_reappear_match_run": MIN_REAPPEAR_MATCH_RUN,
            "continuous_component_split": "balanced_full_component",
        },
    }


def _phase_metadata(phase: TargetPhase) -> dict:
    return {
        "phase_status": phase.status,
        "V": list(phase.visible_indices),
        "D": list(phase.departure_indices),
        "R": list(phase.return_indices),
    }


def _revisit_score_fields(
    scores: tuple[float, ...],
    coverages: tuple[float, ...],
    *,
    expected_revisits: int | None,
) -> dict[str, float]:
    # Automatic mode has a target-dependent number of segments. Keep those
    # scores in per_target.revisits, not misaligned video-level scalar fields.
    if expected_revisits is None:
        return {}
    fields: dict[str, float] = {}
    for index in range(MAX_EXPECTED_REVISITS):
        revisit_id = index + 1
        fields[f"revisit_{revisit_id}_score"] = scores[index] if index < len(scores) else 0.0
        fields[f"revisit_{revisit_id}_coverage"] = coverages[index] if index < len(coverages) else 0.0
    return fields


def _official_revisit_score(
    scores: tuple[float, ...],
    *,
    expected_revisits: int | None,
) -> float:
    if expected_revisits is None:
        return sum(scores) / len(scores) if scores else 0.0
    return sum(scores[:expected_revisits]) / expected_revisits


def _revisit_metadata(
    phase: TargetPhase,
    scores: tuple[float, ...],
    coverages: tuple[float, ...],
    counts: tuple[int, ...],
    *,
    count_key: str,
    expected_revisits: int | None,
) -> list[dict]:
    output: list[dict] = []
    for index in range(len(scores) if expected_revisits is None else MAX_EXPECTED_REVISITS):
        indices = phase.revisit_indices[index] if index < len(phase.revisit_indices) else ()
        slot = phase.revisit_slots[index] if index < len(phase.revisit_slots) else ()
        boundary = phase.revisit_nominal_boundaries[index] if index < len(phase.revisit_nominal_boundaries) else None
        search_window = phase.revisit_search_windows[index] if index < len(phase.revisit_search_windows) else ()
        split_source = phase.revisit_sources[index] if index < len(phase.revisit_sources) else "missing"
        departure_evidence = (
            phase.revisit_departure_evidence[index] if index < len(phase.revisit_departure_evidence) else False
        )
        output.append(
            {
                "revisit_id": index + 1,
                "indices": list(indices),
                "slot": list(slot),
                "nominal_boundary": boundary,
                "search_window": list(search_window),
                "actual_window": ([indices[0], indices[-1] + 1] if indices else []),
                "window_source": split_source,
                "departure_evidence": departure_evidence,
                "score": scores[index] if index < len(scores) else 0.0,
                "coverage": coverages[index] if index < len(coverages) else 0.0,
                count_key: counts[index] if index < len(counts) else 0,
            }
        )
    return output


def _first_frame_match(
    item: dict,
) -> tuple[str, str | None]:
    preparation_error = item.get("human_preparation_error")
    if preparation_error:
        error_type = DataInvalidError if preparation_error["status"] == "data_invalid" else EvaluationError
        raise error_type(preparation_error["message"])
    raw_status = item.get("annotations", {}).get("first_frame_match_status")
    if raw_status not in {"ok", "failed"}:
        raise DataInvalidError(
            "annotations.first_frame_match_status must be 'ok' or 'failed'; "
            "provide the source case files and run eval to prepare annotations"
        )
    raw_error = item.get("annotations", {}).get("first_frame_match_error")
    if raw_error is not None and not isinstance(raw_error, str):
        raise DataInvalidError("annotations.first_frame_match_error must be a string or null")
    error = raw_error.strip() if isinstance(raw_error, str) else None
    if raw_status == "failed" and not error:
        raise DataInvalidError(
            "failed first_frame_match_status requires a non-blank annotations.first_frame_match_error"
        )
    return raw_status, error


def _zero_revisits(*, count_key: str) -> list[dict]:
    return [
        {
            "revisit_id": index + 1,
            "indices": [],
            "slot": [],
            "nominal_boundary": None,
            "search_window": [],
            "actual_window": [],
            "window_source": "first_frame_match_failed",
            "departure_evidence": False,
            "score": 0.0,
            "coverage": 0.0,
            count_key: 0,
        }
        for index in range(MAX_EXPECTED_REVISITS)
    ]


def _zero_units_for_first_frame_failure(
    item: dict,
    *,
    metric_name: str,
    unit_type: str,
    prefix: str,
    diagnostic_scores: tuple[str, str],
    count_metadata: tuple[str, str],
    error: str,
) -> list[UnitResult]:
    boxes = item.get("annotations", {}).get("target_face_bboxes_first_frame")
    descriptions = item.get("annotations", {}).get("target_descriptions")
    expected_revisits = validate_expected_revisits(item.get("annotations", {}).get("expected_revisits"))
    if not isinstance(boxes, (list, tuple)) or not boxes:
        raise DataInvalidError("annotations.target_face_bboxes_first_frame must be non-empty")
    if not isinstance(descriptions, (list, tuple)) or len(descriptions) != len(boxes):
        raise DataInvalidError("annotations.target_descriptions must align with target face boxes")
    video_path = item["video"]
    shared = {
        "video_path": video_path,
        "n_targets": len(boxes),
        "target_face_bboxes_first_frame": list(boxes),
        "target_descriptions": [str(value) for value in descriptions],
        "expected_revisits": expected_revisits,
        "revisit_mode": "auto" if expected_revisits is None else "fixed",
        "first_frame_match_status": "failed",
        "first_frame_match_error": error,
        "phase_source": "first_frame_match_failure",
        "phase_inference": {
            "zero_score_policy": "whole_model_case_trajectory",
            "reason": error,
        },
    }
    zero_scores = {
        "score": 0.0,
        diagnostic_scores[0]: 0.0,
        diagnostic_scores[1]: 0.0,
        **_revisit_score_fields((), (), expected_revisits=expected_revisits),
    }
    return [
        UnitResult(
            unit_type=unit_type,
            unit_id=f"target_{target_id:04d}",
            scores=dict(zero_scores),
            status=ResultStatus.OK,
            metadata={
                **shared,
                "target_id": target_id,
                "phase_status": "first_frame_match_failed",
                "V": [],
                "D": [],
                "R": [],
                f"{prefix}_frame_scores": {},
                count_metadata[0]: 0,
                count_metadata[1]: 0,
                "revisits": _zero_revisits(count_key=count_metadata[1]) if expected_revisits is not None else [],
            },
        )
        for target_id in range(len(boxes))
    ]


def _aggregate_units(
    item: dict,
    units: list[UnitResult],
    *,
    metric_name: str,
    prefix: str,
    diagnostic_scores: tuple[str, ...],
    count_metadata: tuple[str, ...],
) -> ItemResult:
    if not units:
        raise EvaluationError(f"{metric_name} produced no target units")
    non_scored = [unit for unit in units if not unit.status.is_scored]
    if non_scored:
        first = non_scored[0]
        return ItemResult(
            **{key: item.get(key) for key in ("item_id", "model_id", "category", "sample_id", "condition_id")},
            status=first.status,
            error=first.error,
            metadata=first.metadata,
        )
    target_ids = sorted(int(unit.metadata["target_id"]) for unit in units)
    if target_ids != list(range(len(units))):
        raise EvaluationError(f"{metric_name} target units are incomplete")

    scores = {key: sum(unit.scores[key] for unit in units) / len(units) for key in units[0].scores}
    metadata = {key: units[0].metadata[key] for key in _SHARED_ITEM_METADATA}
    metadata["per_target"] = [
        {
            **{key: unit.metadata[key] for key in _SHARED_TARGET_METADATA},
            f"{prefix}_score": unit.scores["score"],
            **{key: unit.scores[key] for key in diagnostic_scores},
            **{key: unit.metadata[key] for key in count_metadata},
        }
        for unit in units
    ]
    return ItemResult(
        **{key: item.get(key) for key in ("item_id", "model_id", "category", "sample_id", "condition_id")},
        scores=scores,
        num_units=len(units),
        status=ResultStatus.OK,
        metadata=metadata,
    )


class HumanIdentityMetric(Metric):
    aggregate_summary = staticmethod(aggregate_video_summary)

    name = "human_identity"
    description = (
        "InsightFace first-frame-seeded Hungarian face presence with "
        "boundary-buffer departure insertion, frozen ArcFace references, "
        "revisit splits, per-revisit official macro, and coverage."
    )
    unit_type = "target_identity"

    def __init__(
        self,
        *,
        backend: FaceIdentityBackend | None = None,
        body_backend: BodyTrackerBackend | None = None,
        reader: VideoReader | None = None,
    ) -> None:
        self.backend = backend
        self.body_backend = body_backend
        self.reader = reader or VideoReader()

    def evaluate(
        self,
        item: dict,
        context: EvaluationContext,
    ) -> list[UnitResult]:
        if item.get("category") not in (None, "human"):
            return [
                UnitResult(
                    unit_id="not_applicable",
                    status=ResultStatus.NOT_APPLICABLE,
                    metadata={"reason": "human metrics apply to human cases"},
                )
            ]
        match_status, match_error = _first_frame_match(item)
        if match_status == "failed":
            assert match_error is not None
            return _zero_units_for_first_frame_failure(
                item,
                metric_name=self.name,
                unit_type=self.unit_type,
                prefix="identity",
                diagnostic_scores=("identity_similarity", "identity_coverage"),
                count_metadata=(
                    "n_reference_embeddings",
                    "n_return_embeddings",
                ),
                error=match_error,
            )
        from .evaluator import identity_analysis_for_item

        identity = identity_analysis_for_item(
            item,
            context,
            backend=self.backend,
            body_backend=self.body_backend,
            reader=self.reader,
        )
        shared = _identity_metadata(
            identity,
            phase_source="rolling_face_tracks",
        )
        shared.update(
            {
                "first_frame_match_status": match_status,
                "first_frame_match_error": match_error,
            }
        )
        return [
            UnitResult(
                unit_type=self.unit_type,
                unit_id=f"target_{result.target_id:04d}",
                scores={
                    "score": _official_revisit_score(
                        result.revisit_scores,
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                    "identity_similarity": result.similarity,
                    "identity_coverage": result.coverage,
                    **_revisit_score_fields(
                        result.revisit_scores,
                        result.revisit_coverages,
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                },
                status=ResultStatus.OK,
                metadata={
                    **shared,
                    "target_id": result.target_id,
                    **_phase_metadata(result.phase),
                    "identity_frame_scores": {str(index): score for index, score in result.frame_scores.items()},
                    "n_reference_embeddings": result.n_reference_embeddings,
                    "n_return_embeddings": result.n_return_embeddings,
                    "revisits": _revisit_metadata(
                        result.phase,
                        result.revisit_scores,
                        result.revisit_coverages,
                        result.n_revisit_embeddings,
                        count_key="n_return_embeddings",
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                },
            )
            for result in identity.target_results
        ]

    def aggregate_item(
        self,
        item: dict,
        units: list[UnitResult],
    ) -> ItemResult:
        return _aggregate_units(
            item,
            units,
            metric_name=self.name,
            prefix="identity",
            diagnostic_scores=("identity_similarity", "identity_coverage"),
            count_metadata=("n_reference_embeddings", "n_return_embeddings"),
        )


class HumanAppearanceMetric(Metric):
    aggregate_summary = staticmethod(aggregate_video_summary)

    name = "human_appearance"
    description = (
        "Seeded-body presence with boundary-buffer departure insertion, "
        "phase-local YOLO-World/SAM2 person masks, component-level revisit "
        "splits, per-revisit official macro, and DINOv2-Large similarity."
    )
    unit_type = "target_appearance"

    def __init__(
        self,
        *,
        face_backend: FaceIdentityBackend | None = None,
        body_backend: BodyTrackerBackend | None = None,
        appearance_backend: AppearanceBackend | None = None,
        reader: VideoReader | None = None,
    ) -> None:
        self.face_backend = face_backend
        self.body_backend = body_backend
        self.appearance_backend = appearance_backend
        self.reader = reader or VideoReader()

    def evaluate(
        self,
        item: dict,
        context: EvaluationContext,
    ) -> list[UnitResult]:
        if item.get("category") not in (None, "human"):
            return [
                UnitResult(
                    unit_id="not_applicable",
                    status=ResultStatus.NOT_APPLICABLE,
                    metadata={"reason": "human metrics apply to human cases"},
                )
            ]
        match_status, match_error = _first_frame_match(item)
        if match_status == "failed":
            assert match_error is not None
            return _zero_units_for_first_frame_failure(
                item,
                metric_name=self.name,
                unit_type=self.unit_type,
                prefix="appearance",
                diagnostic_scores=(
                    "appearance_similarity",
                    "reappearance_coverage",
                ),
                count_metadata=(
                    "n_reference_features",
                    "n_return_features",
                ),
                error=match_error,
            )
        from .evaluator import appearance_analysis_for_item

        identity, appearance = appearance_analysis_for_item(
            item,
            context,
            face_backend=self.face_backend,
            body_backend=self.body_backend,
            appearance_backend=self.appearance_backend,
            reader=self.reader,
        )
        shared = _identity_metadata(
            identity,
            phase_source="seeded_body_tracks",
        )
        shared.update(
            {
                "first_frame_match_status": match_status,
                "first_frame_match_error": match_error,
            }
        )
        return [
            UnitResult(
                unit_type=self.unit_type,
                unit_id=f"target_{result.target_id:04d}",
                scores={
                    "score": _official_revisit_score(
                        result.revisit_scores,
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                    "appearance_similarity": result.score,
                    "reappearance_coverage": result.reappearance_coverage,
                    **_revisit_score_fields(
                        result.revisit_scores,
                        result.revisit_coverages,
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                },
                status=ResultStatus.OK,
                metadata={
                    **shared,
                    "target_id": result.target_id,
                    **_phase_metadata(result.phase),
                    "appearance_frame_scores": {str(index): score for index, score in result.frame_scores.items()},
                    "n_reference_features": result.n_reference_features,
                    "n_return_features": result.n_return_features,
                    "revisits": _revisit_metadata(
                        result.phase,
                        result.revisit_scores,
                        result.revisit_coverages,
                        result.n_revisit_features,
                        count_key="n_return_features",
                        expected_revisits=identity.protocol.expected_revisits,
                    ),
                },
            )
            for result in appearance.target_results
        ]

    def aggregate_item(
        self,
        item: dict,
        units: list[UnitResult],
    ) -> ItemResult:
        return _aggregate_units(
            item,
            units,
            metric_name=self.name,
            prefix="appearance",
            diagnostic_scores=("appearance_similarity", "reappearance_coverage"),
            count_metadata=("n_reference_features", "n_return_features"),
        )
