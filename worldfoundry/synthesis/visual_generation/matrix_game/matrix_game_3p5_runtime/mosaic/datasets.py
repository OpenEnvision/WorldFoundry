"""Dataset construction for Matrix-Game inference."""

from __future__ import annotations

import json

from ..data import DA3MosaicVideoDataset, SubjectRefMemoryDA3MosaicVideoDataset
from .config import (
    _apply_intrinsics_mode,
    _intrinsics_mode_arg,
    _normalize_mosaic_fuse_mode,
)

_DATASET_TYPES = {
    "da3_video": DA3MosaicVideoDataset,
    "da3_video_subject_ref": SubjectRefMemoryDA3MosaicVideoDataset,
}


def _resolve_dataset_cls_and_extra_kwargs(args):
    args = _apply_intrinsics_mode(args)
    dataset_type = str(getattr(args, "dataset_type", "da3_video"))
    try:
        dataset_cls = _DATASET_TYPES[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Matrix-Game inference supports {sorted(_DATASET_TYPES)}, got {dataset_type!r}") from exc

    kwargs = {
        "vipe_prompt_type": int(getattr(args, "vipe_prompt_type", 1)),
        "vipe_prompt_segment_format": bool(getattr(args, "vipe_prompt_segment_format", False)),
        "align_generating_to_prompt_segments": bool(getattr(args, "align_generating_to_prompt_segments", False)),
        "mosaic_intrinsics_mode": _intrinsics_mode_arg(
            args,
            "mosaic_intrinsics_mode",
            "episode_mean",
        ),
        "mosaic_query_reference_frame": int(getattr(args, "mosaic_query_reference_frame", 4)),
        "mosaic_view_change_prope": bool(
            getattr(args, "mosaic_view_change_prope", False)
            and getattr(args, "enable_mosaic", True)
            and getattr(args, "use_prope", False)
            and not getattr(args, "only_prope", False)
        ),
        "prompt": "",
        "require_depth": True,
        "allow_no_prompt": bool(getattr(args, "allow_no_prompt", False)),
        "loose_prompt_match": bool(getattr(args, "loose_prompt_match", False)),
        "height": int(args.height),
        "width": int(args.width),
        "memory_latent_cache_dir": str(getattr(args, "memory_latent_cache_dir", "") or ""),
        "memory_latent_cache_version": str(getattr(args, "memory_latent_cache_version", "wan2.2_ti2v_5b_vae")),
        "memory_vae_encode_input_frames": int(getattr(args, "memory_vae_encode_input_frames", 1) or 1),
        "dataset_compact_mode": bool(getattr(args, "dataset_compact_mode", False)),
        "mosaic_fuse_mode": _normalize_mosaic_fuse_mode(getattr(args, "mosaic_fuse_mode", "fill_stop_zbuffer")),
        "candidates_per_query_group": int(getattr(args, "candidates_per_query_group", 5)),
        "mosaic_selection_mode": str(getattr(args, "mosaic_selection_mode", "pose_nearest")),
        "mosaic_candidate_nms_mode": _optional_mode(getattr(args, "mosaic_candidate_nms_mode", "pose")),
        "mosaic_candidate_nms_projection_iou_threshold": float(
            getattr(args, "mosaic_candidate_nms_projection_iou_threshold", 0.7)
        ),
        "mosaic_candidate_nms_min_temporal_gap": int(getattr(args, "mosaic_candidate_nms_min_temporal_gap", 0)),
        "mosaic_candidate_nms_pose_distance_threshold": float(
            getattr(args, "mosaic_candidate_nms_pose_distance_threshold", 0.1)
        ),
        "mosaic_candidate_nms_pool_multiplier": float(getattr(args, "mosaic_candidate_nms_pool_multiplier", 2.0)),
        "mosaic_coverage_grid_downsample": int(getattr(args, "mosaic_coverage_grid_downsample", 4)),
        "mosaic_coverage_pool_stride": int(getattr(args, "mosaic_coverage_pool_stride", 2)),
    }
    if dataset_type == "da3_video_subject_ref":
        kwargs.update(
            {
                "subject_ref_dir_name": str(getattr(args, "subject_ref_dir_name", "protagonist_refs")),
                "subject_num_refs_max": int(getattr(args, "subject_num_refs_max", 2)),
                "subject_dissimilar_top_k": int(getattr(args, "subject_dissimilar_top_k", 8)),
                "subject_max_similarity": float(getattr(args, "subject_max_similarity", 0.94)),
                "subject_shuffle_refs": False,
                "subject_ref_canvas_slot_ratio": float(getattr(args, "subject_ref_canvas_slot_ratio", 0.5)),
                "subject_ref_background": str(getattr(args, "subject_ref_background", "imagenet_mean")),
            }
        )
    return dataset_cls, kwargs


def _optional_mode(value):
    return None if value in (None, "", "none", "None") else str(value)


def _resolve_dataset_cache_dir(args):
    if getattr(args, "no_dataset_cache", False):
        return None
    return getattr(args, "dataset_cache_dir", None) or None


def _load_manifest_keys(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Inference manifest must be an object: {path}")
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    for key in ("inference_keys", "sample_keys", "dataset_keys"):
        values = payload.get(key) or split.get(key)
        if isinstance(values, list) and values:
            return list(dict.fromkeys(str(value) for value in values))
    info = payload.get("dataset_info")
    if isinstance(info, dict):
        return list(info)
    raise ValueError(f"Inference manifest contains no sample keys: {path}")


def build_mosaic_inference_dataset(args, dataset_cls=None):
    """Build a Matrix rollout dataset, with an injectable research variant."""
    inferred_cls, extra_kwargs = _resolve_dataset_cls_and_extra_kwargs(args)
    dataset_cls = dataset_cls or inferred_cls
    dataset_index_path = getattr(args, "dataset_index_path", None)
    if not dataset_index_path:
        raise ValueError("Matrix-Game inference requires --dataset_index_path")

    inference_manifest = getattr(args, "inference_manifest_path", None)
    inference_keys = _load_manifest_keys(inference_manifest) if inference_manifest else None
    return dataset_cls(
        base_path=(),
        camera_params_path=(),
        depth_path=(),
        latent_window_size=int(args.latent_window_size),
        inference_assign_yaml=None,
        inference_ratio=1.0,
        inference_paths=inference_keys,
        filter_yaml=getattr(args, "filter_yaml", None),
        include_yaml=getattr(args, "include_yaml", None),
        random_start_latent_prob=0.0,
        seed=int(args.seed),
        rank=int(getattr(args, "rank", 0)),
        max_data_items=getattr(args, "max_data_items", None),
        max_scan_items=getattr(args, "max_scan_items", None),
        dataset_cache_dir=_resolve_dataset_cache_dir(args),
        max_frames_per_scene=getattr(args, "max_frames_per_scene", None),
        force_override_extrinsic=getattr(args, "force_override_extrinsic", None),
        dataset_index_path=dataset_index_path,
        min_frame_count=getattr(args, "min_frame_count", None),
        dataset_path_shuffle_seed=int(getattr(args, "dataset_path_shuffle_seed", 42)),
        min_anchor_frame_idx=int(getattr(args, "min_anchor_frame_idx", 0)),
        **extra_kwargs,
    )


__all__ = ["build_mosaic_inference_dataset"]
