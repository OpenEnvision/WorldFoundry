# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Shared resource construction and cached Human Entity analyses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from ..features import MaskedDinoV2Features, get_dino_model
from ..revisit_protocol import MAX_SHORT_MISSED_FRAMES, REVISIT_BOUNDARY_BUFFER_FRAMES
from .appearance_features import (
    AppearanceBackend,
    YoloSamDinoAppearanceBackend,
    analyze_appearance,
)
from .body_tracking import (
    BODY_REACQUISITION_FACE_THRESHOLD,
    BodyTrackerBackend,
    YoloSamBodyTracker,
)
from .face_identity import (
    TRACK_ASSOCIATION_THRESHOLD,
    FaceIdentityBackend,
    InsightFaceBackend,
    analyze_identity,
    prepare_face_tracks,
)
from .person_mask_tracker import (
    TransformersSam2MaskPropagator,
    YoloWorldPersonDetector,
)
from .phase_protocol import load_human_entity_protocol
from .schemas import AppearanceAnalysis, IdentityAnalysis

YOLO_PERSON_CONFIDENCE = 0.05
MIN_MASK_AREA_RATIO = 0.0
DINO_BATCH_SIZE = 16


def _face_backend(context: EvaluationContext, weight_metric="human_identity") -> FaceIdentityBackend:
    config = context.metric_config("human_identity")
    backend_name = str(config.get("backend", "insightface")).lower()
    if backend_name != "insightface":
        raise EvaluationError("human_identity.backend is fixed to 'insightface' for comparable scores")
    model_name = str(config.get("insightface_model", "buffalo_l"))
    model_root = config.get("insightface_root")
    if model_root is None:
        # InsightFace accepts an absolute model name. This loads the metric's
        # buffalo_l link directly without creating another models/ directory.
        model_root = context.cache_dir
        model_name = str((context.cache_dir / "metrics" / weight_metric / model_name).resolve())
    key = (
        "human_entity:face_backend:insightface",
        str((Path(model_root).expanduser() / "models" / model_name).resolve()),
        context.device,
    )
    return context.get_resource(
        key,
        lambda: InsightFaceBackend(
            model_name=model_name,
            model_root=model_root,
            device=context.device,
        ),
    )


def _person_detector(context: EvaluationContext, weight_metric="human_appearance") -> YoloWorldPersonDetector:
    config = context.metric_config("human_appearance")
    checkpoint = str(
        Path(config.get("yolo_checkpoint") or context.cache_dir / "metrics" / weight_metric / "yolov8s-worldv2.pt")
        .expanduser()
        .resolve()
    )
    key = (
        "human_entity:person_detector",
        checkpoint,
        YOLO_PERSON_CONFIDENCE,
        context.device,
    )
    return context.get_resource(
        key,
        lambda: YoloWorldPersonDetector(
            checkpoint=checkpoint,
            clip_checkpoint=str(context.cache_dir / "metrics" / weight_metric / "ViT-B-32.pt"),
            confidence=YOLO_PERSON_CONFIDENCE,
            device=context.device,
        ),
    )


def _mask_propagator(
    context: EvaluationContext,
    weight_metric="human_appearance",
) -> TransformersSam2MaskPropagator:
    config = context.metric_config("human_appearance")
    checkpoint = str(
        Path(config.get("sam2_checkpoint") or context.cache_dir / "metrics" / weight_metric / "sam2-hiera-large")
        .expanduser()
        .resolve()
    )
    key = (
        "human_entity:mask_propagator",
        checkpoint,
        MIN_MASK_AREA_RATIO,
        context.device,
    )
    return context.get_resource(
        key,
        lambda: TransformersSam2MaskPropagator(
            checkpoint=checkpoint,
            min_mask_area_ratio=MIN_MASK_AREA_RATIO,
            device=context.device,
        ),
    )


def _dino_extractor(
    context: EvaluationContext,
) -> MaskedDinoV2Features:
    config = context.metric_config("human_appearance")
    checkpoint = str(
        config.get("dino_checkpoint") or context.cache_dir / "metrics" / "human_appearance" / "dinov2-large"
    )
    raw_batch_size = config.get("dino_batch_size", DINO_BATCH_SIZE)
    if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, int) or raw_batch_size < 1:
        raise EvaluationError("human_appearance.dino_batch_size must be positive")
    key = (
        "human_entity:dino_extractor",
        checkpoint,
        raw_batch_size,
        context.device,
    )
    return context.get_resource(
        key,
        lambda: MaskedDinoV2Features(
            model=get_dino_model(context, checkpoint),
            device=context.device,
            batch_size=raw_batch_size,
        ),
    )


def _appearance_backend(context: EvaluationContext) -> AppearanceBackend:
    mask_propagator = _mask_propagator(context)
    feature_extractor = _dino_extractor(context)
    key = (
        "human_entity:appearance_backend",
        id(mask_propagator),
        id(feature_extractor),
    )
    return context.get_resource(
        key,
        lambda: YoloSamDinoAppearanceBackend(
            mask_propagator=mask_propagator,
            feature_extractor=feature_extractor,
        ),
    )


def _body_backend(context: EvaluationContext, weight_metric="human_identity") -> BodyTrackerBackend:
    person_detector = _person_detector(context, weight_metric)
    mask_propagator = _mask_propagator(context, weight_metric)
    key = (
        "human_entity:body_backend",
        id(person_detector),
        id(mask_propagator),
    )
    return context.get_resource(
        key,
        lambda: YoloSamBodyTracker(
            person_detector=person_detector,
            mask_propagator=mask_propagator,
        ),
    )


def _non_negative_int_config(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be a non-negative integer")
    return value


def identity_analysis_for_item(
    item: dict,
    context: EvaluationContext,
    *,
    backend: FaceIdentityBackend | None = None,
    body_backend: BodyTrackerBackend | None = None,
    reader: VideoReader | None = None,
    weight_metric: str = "human_identity",
) -> IdentityAnalysis:
    local_path = item["video"]
    video_reader = reader or VideoReader()
    info = video_reader.probe(local_path)
    protocol = load_human_entity_protocol(
        item,
        total_frames=info.total_frames,
        frame_width=info.width,
        frame_height=info.height,
    )
    selected_backend = backend or _face_backend(context, weight_metric)
    selected_body_backend = body_backend or _body_backend(context, weight_metric)
    identity_config = context.metric_config("human_identity")
    max_revisit_gap_frames = _non_negative_int_config(
        "human_identity.max_revisit_gap_frames",
        identity_config.get("max_revisit_gap_frames", MAX_SHORT_MISSED_FRAMES),
    )
    revisit_boundary_buffer_frames = _non_negative_int_config(
        "human_identity.revisit_boundary_buffer_frames",
        identity_config.get(
            "revisit_boundary_buffer_frames",
            REVISIT_BOUNDARY_BUFFER_FRAMES,
        ),
    )
    track_threshold = TRACK_ASSOCIATION_THRESHOLD
    body_reacquisition_threshold = BODY_REACQUISITION_FACE_THRESHOLD
    cache_key = (
        "human_entity:identity_analysis",
        item["item_id"],
        local_path,
        id(selected_backend),
        id(selected_body_backend),
        track_threshold,
        body_reacquisition_threshold,
        max_revisit_gap_frames,
        revisit_boundary_buffer_frames,
    )

    def create() -> IdentityAnalysis:
        decoded = video_reader.read_indices(
            local_path,
            list(protocol.required_frame_indices),
        )
        decoded_by_index = {record.frame_idx: record.rgb for record in decoded}
        face_tracks = prepare_face_tracks(
            protocol=protocol,
            decoded_frames=decoded_by_index,
            backend=selected_backend,
            track_association_threshold=track_threshold,
        )
        body = selected_body_backend.analyze(
            protocol,
            decoded_by_index,
            max_revisit_gap_frames=max_revisit_gap_frames,
            revisit_boundary_buffer_frames=revisit_boundary_buffer_frames,
            face_tracks=face_tracks,
            face_reacquisition_threshold=body_reacquisition_threshold,
        )
        return analyze_identity(
            protocol=protocol,
            decoded_frames=decoded_by_index,
            backend=selected_backend,
            video_path=local_path,
            face_tracks=face_tracks,
            track_association_threshold=track_threshold,
            target_phases=body.target_phases,
            body_assignments=body.assignments,
            max_revisit_gap_frames=max_revisit_gap_frames,
            revisit_boundary_buffer_frames=revisit_boundary_buffer_frames,
        )

    return context.get_cached(
        cache_key,
        create,
    )


def appearance_analysis_for_item(
    item: dict,
    context: EvaluationContext,
    *,
    face_backend: FaceIdentityBackend | None = None,
    appearance_backend: AppearanceBackend | None = None,
    reader: VideoReader | None = None,
    body_backend: BodyTrackerBackend | None = None,
) -> tuple[IdentityAnalysis, AppearanceAnalysis]:
    identity = identity_analysis_for_item(
        item,
        context,
        backend=face_backend,
        body_backend=body_backend,
        reader=reader,
        weight_metric="human_appearance",
    )
    video_reader = reader or VideoReader()
    local_path = identity.video_path
    decoded = video_reader.read_indices(
        local_path,
        list(identity.protocol.required_frame_indices),
    )
    # Appearance currently re-decodes the full item to keep the shared cache
    # free of hundreds of decoded RGB frames between metric evaluations.
    identity_with_frames = replace(
        identity,
        decoded_frames={record.frame_idx: record.rgb for record in decoded},
    )
    selected_backend = appearance_backend or _appearance_backend(context)
    cache_key = (
        "human_entity:appearance_analysis",
        item["item_id"],
        identity.video_path,
        id(selected_backend),
    )
    appearance = context.get_cached(
        cache_key,
        lambda: analyze_appearance(identity_with_frames, selected_backend),
    )
    return identity, appearance
