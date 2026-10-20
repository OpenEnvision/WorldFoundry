# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Shared Object tracking and appearance analysis, scoped to one evaluation context."""

from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.video import VideoReader

from ..features import MaskedDinoV2Features, get_dino_model
from ..revisit_protocol import MAX_SHORT_MISSED_FRAMES, infer_target_phase
from .appearance_features import analyze_object_appearance
from .mask_alignment import align_mask_to_generated_frame
from .object_redetector import GroundingDinoClipObjectRedetector, object_query_crop
from .object_tracker import TransformersSam2ObjectTracker
from .preparation import load_source, source_path
from .protocol import load_object_entity_protocol
from .schemas import ObjectAppearanceAnalysis, ObjectTrackAnalysis


def _tracker(context, weight_metric="object_appearance"):
    config = context.metric_config("object_appearance")
    checkpoint = str(
        config.get("sam2_checkpoint") or context.cache_dir / "metrics" / weight_metric / "sam2-hiera-large"
    )
    cache_size = config.get("sam2_vision_cache_size", 1)
    return context.get_resource(
        "object:tracker",
        lambda: TransformersSam2ObjectTracker(
            checkpoint=checkpoint, device=context.device, vision_cache_size=cache_size
        ),
    )


def _redetector(context, weight_metric="object_appearance"):
    config = context.metric_config("object_appearance")
    checkpoint = str(
        config.get("grounding_dino_checkpoint") or context.cache_dir / "metrics" / weight_metric / "grounding-dino-base"
    )
    reid = str(config.get("clip_reid_checkpoint") or context.cache_dir / "metrics" / weight_metric / "ViT-B-32.pt")
    # Forward explicit overrides; defaults live in the backend.
    fields = (
        "search_stride",
        "batch_size",
        "box_threshold",
        "text_threshold",
        "nms_threshold",
        "candidate_limit",
        "reid_batch_size",
        "reid_similarity_threshold",
        "reid_high_confidence_threshold",
        "reid_min_similarity_margin",
        "min_confirmations",
        "max_sample_gap",
        "min_box_area_ratio",
        "max_box_area_ratio",
    )
    options = {key: config[f"redetection_{key}"] for key in fields if f"redetection_{key}" in config}
    if "redetection_reid_crop_context" in config:
        options["reid_crop_context_fraction"] = config["redetection_reid_crop_context"]
    return context.get_resource(
        "object:redetector",
        lambda: GroundingDinoClipObjectRedetector(
            grounding_dino_checkpoint=checkpoint, reid_checkpoint=reid, device=context.device, **options
        ),
    )


def _feature_extractor(context):
    config = context.metric_config("object_appearance")
    checkpoint = str(config.get("dino_checkpoint") or context.cache_dir / "metrics/object_appearance/dinov2-large")
    return context.get_resource(
        "object:dino",
        lambda: MaskedDinoV2Features(model=get_dino_model(context, checkpoint), device=context.device, batch_size=1),
    )


def _visibility_segments(observations):
    segments = []
    for index in sorted(i for i, observation in observations.items() if observation.present):
        if segments and segments[-1][1] + 1 == index:
            segments[-1] = (segments[-1][0], index)
        else:
            segments.append((index, index))
    return tuple(segments)


def object_track_analysis_for_item(
    item, context, *, tracker=None, redetector=None, reader=None, weight_metric="object_appearance"
):
    # The orchestrator clears this case cache after both metrics finish. Context
    # configuration/model instances stay fixed throughout the evaluation run.
    def create():
        video_path = str(Path(item["video"]).resolve())
        image_path = source_path(item, "condition_image")
        mask_path = source_path(item, "target_object_mask_first_frame")
        video_reader = reader or VideoReader()
        info = video_reader.probe(video_path)
        protocol = load_object_entity_protocol(
            item, total_frames=info.total_frames, condition_image_path=image_path, target_mask_path=mask_path
        )
        condition_rgb, source_mask = load_source(image_path, mask_path)
        decoded = video_reader.read_indices(video_path, list(protocol.required_frame_indices))
        decoded_frames = {record.frame_idx: record.rgb for record in decoded}
        frames = [decoded_frames[index] for index in protocol.required_frame_indices]
        alignment = align_mask_to_generated_frame(condition_rgb, source_mask, frames[0])
        if not alignment.mask.any():
            raise DataInvalidError("source mask is empty after alignment to the generated first frame")
        query = object_query_crop(frames[0], alignment.mask)
        prompts = (redetector or _redetector(context, weight_metric)).detect(protocol.target_description, query, frames)
        observations = (tracker or _tracker(context, weight_metric)).track(frames, alignment.mask, prompts)
        gap = context.metric_config("object_appearance").get("max_revisit_gap_frames", MAX_SHORT_MISSED_FRAMES)
        phase = infer_target_phase(
            target_id=0,
            total_frames=protocol.total_frames,
            matched_frame_indices=[i for i, observation in observations.items() if observation.present],
            max_revisit_gap_frames=gap,
            expected_revisits=protocol.expected_revisits,
        )
        return ObjectTrackAnalysis(
            protocol=protocol,
            phase=phase,
            observations=observations,
            decoded_frames=decoded_frames,
            video_path=video_path,
            alignment_method=alignment.method,
            frame0_alignment_score=alignment.score,
            redetection_prompts=prompts,
            visibility_segments=_visibility_segments(observations),
            max_revisit_gap_frames=gap,
        )

    return context.get_cached(("object:track", item["item_id"]), create)


def object_appearance_analysis_for_item(
    item, context, *, tracker=None, redetector=None, feature_extractor=None, reader=None
):
    track = object_track_analysis_for_item(item, context, tracker=tracker, redetector=redetector, reader=reader)
    result = analyze_object_appearance(track, feature_extractor or _feature_extractor(context))
    return ObjectAppearanceAnalysis(track=track, result=result)
