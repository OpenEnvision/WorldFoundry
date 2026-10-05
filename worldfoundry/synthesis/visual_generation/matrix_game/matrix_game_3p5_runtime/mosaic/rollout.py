"""Multi-segment Matrix-Game 3.5 inference rollout.

This module contains the product inference path only.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from worldfoundry.core.io import save_video_h264

from .cleanup import _rank_local_cuda_device, _release_cached_memory, _set_rank_local_cuda_device
from .config import _intrinsics_mode_arg, _normalize_mosaic_fuse_mode
from .history import _compute_mosaic_frame_indices
from .prompting import _require_nonempty_prompt
from .video_io import (
    _decode_latents_to_numpy_frames,
    _encode_frames_per_frame,
    _get_frustum_handler_cls,
    _init_da3_depth_estimator,
    _parse_query_hits,
    _prope_camera_kwargs,
)


class _InferenceResources:
    __slots__ = ("depth_estimator",)

    def __init__(self):
        self.depth_estimator = None


def _release_inference_resources(resources):
    resources.depth_estimator = None
    _release_cached_memory()


def run_mosaic_inference(
    accelerator,
    inference_dataset,
    model,
    log_dir,
    args,
    run_id=0,
):
    if inference_dataset is None:
        return
    resources = _InferenceResources()
    try:
        _run_mosaic_inference_impl(
            accelerator,
            inference_dataset,
            model,
            log_dir,
            args,
            run_id,
            resources=resources,
        )
    finally:
        _release_inference_resources(resources)
    accelerator.wait_for_everyone()


def _run_mosaic_inference_impl(
    accelerator,
    inference_dataset,
    model,
    log_dir,
    args,
    run_id=0,
    *,
    resources,
):
    FrustumHandler = _get_frustum_handler_cls()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    proc_rank = int(getattr(accelerator, "process_index", 0))
    proc_tag = f"proc_{proc_rank:03d}"
    use_mosaic = bool(unwrapped_model.enable_mosaic and not unwrapped_model.only_prope)
    use_mosaic_view_change_prope = bool(use_mosaic and getattr(unwrapped_model, "mosaic_view_change_prope", False))
    max_batches = max(1, int(args.num_inference_batches))
    num_blocks = max(1, int(args.num_inference_blocks))
    sample_offset = int(getattr(args, "inference_sample_offset", 0) or 0)
    if hasattr(inference_dataset, "set_inference_sample_offset"):
        inference_dataset.set_inference_sample_offset(sample_offset)

    _set_rank_local_cuda_device()
    dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        shuffle=False,
        collate_fn=lambda items: items[0],
        num_workers=0,
    )
    dataloader = accelerator.prepare(dataloader)

    output_dir = os.path.join(log_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)
    depth_estimator = None
    if use_mosaic:
        depth_estimator = _init_da3_depth_estimator(
            device=_rank_local_cuda_device() or unwrapped_model.device,
        )
    resources.depth_estimator = depth_estimator

    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        if isinstance(data, dict) and data.get("needs_vae_materialization"):
            data = unwrapped_model._materialize_data(data)

        latent_window_size = int(data["noisy_latent_indices"].shape[-1])
        history_latents = unwrapped_model._ensure_batched_latents(data["clean_latents"]).detach().cpu()
        height_latent, width_latent = history_latents.shape[-2:]
        latent_spatial_stride = 16
        height = height_latent * latent_spatial_stride
        width = width_latent * latent_spatial_stride
        vae_decode_tiled = bool(getattr(unwrapped_model, "vae_decode_tiled", False))
        clean_name = data.get("info", {}).get("clean_name", f"batch_{batch_idx}")

        running_clean_indices = data["clean_latent_indices"].clone()
        running_noisy_indices = data["noisy_latent_indices"].clone()
        mosaic_query_latents = None
        handler = None
        if use_mosaic:
            fixed_intrinsics = bool(
                _intrinsics_mode_arg(args, "mosaic_intrinsics_mode", "episode_mean") == "first_frame"
            )
            initial_intrinsics = data["clean_latent_indices_prope_intrinsic"][0].cpu().numpy()
            initial_extrinsics = data["clean_latent_indices_prope_extrinsic"][:1].cpu().numpy()
            handler = FrustumHandler(
                initial_intrinsics,
                image_size=(height, width),
                grid_size=(height_latent, width_latent),
                depth_inf_thresh=1e9,
                depth_estimator=depth_estimator,
                is_c2w=False,
                use_gpu=True,
                init_extrinsic=initial_extrinsics,
                latent_stride=latent_spatial_stride,
                fixed_intrinsics=fixed_intrinsics,
            )

        last_predicting_frames = None
        last_predicting_extrinsics = None
        last_predicting_intrinsics = None
        section_metadata = {}

        for section_idx in range(num_blocks):
            if section_idx > 0:
                running_clean_indices = running_clean_indices + latent_window_size
                running_noisy_indices = running_noisy_indices + latent_window_size

            current_clean_latents = history_latents[:, :, -1:]
            camera_data = inference_dataset.read_camera_params(
                {
                    "clean_latent_indices_start": data["clean_latent_indices_start"],
                    "clean_latent_indices": running_clean_indices,
                    "noisy_latent_indices": running_noisy_indices,
                    "clean_latents": current_clean_latents,
                    "info": data["info"],
                    "lookup": data.get("lookup", {}),
                    "is_starting": section_idx == 0,
                },
                info=data["info"],
                lookup=data.get("lookup", {}),
            )
            prope_camera_kwargs = _prope_camera_kwargs(camera_data)
            mosaic_latent = None
            mosaic_revgrid = None
            mosaic_view_change = None
            mosaic_frame_indices = None

            if use_mosaic:
                if section_idx == 0:
                    register_frames = _decode_latents_to_numpy_frames(
                        unwrapped_model,
                        current_clean_latents,
                        unwrapped_model.device,
                        tiled=vae_decode_tiled,
                    )
                    registration_kwargs = {"start_index": 0}
                    if args.force_using_input_extrinics:
                        registration_kwargs.update(
                            {
                                "extrinsics": data["clean_latent_indices_prope_extrinsic"][-1:].cpu().numpy(),
                                "intrinsics": data["clean_latent_indices_prope_intrinsic"][-1:].cpu().numpy(),
                                "force_using_input_extrinics": True,
                            }
                        )
                else:
                    register_frames = last_predicting_frames
                    registration_kwargs = {}
                    if args.force_using_input_extrinics:
                        if last_predicting_extrinsics is None or last_predicting_intrinsics is None:
                            raise RuntimeError("Previous section camera parameters are unavailable")
                        registration_kwargs.update(
                            {
                                "extrinsics": last_predicting_extrinsics,
                                "intrinsics": last_predicting_intrinsics,
                                "force_using_input_extrinics": True,
                            }
                        )

                registration_kwargs["cache_frames"] = False
                handler.register_source_sequence(
                    unwrapped_model.device,
                    register_frames,
                    **registration_kwargs,
                )
                new_query_latents = _encode_frames_per_frame(
                    unwrapped_model,
                    register_frames,
                    unwrapped_model.device,
                    tiled=vae_decode_tiled,
                )
                mosaic_query_latents = (
                    new_query_latents
                    if mosaic_query_latents is None
                    else torch.cat([mosaic_query_latents, new_query_latents], dim=2)
                )

                clean_extrinsics = camera_data["clean_latent_indices_prope_extrinsic"]
                noisy_extrinsics = camera_data["noisy_latent_indices_prope_extrinsic"]
                query_extrinsics = torch.cat([clean_extrinsics, noisy_extrinsics], dim=0).cpu().numpy()
                query_extrinsics = handler.align_w2c_trajectory(query_extrinsics[3:])[1:]
                source_latents = mosaic_query_latents[0].clone()
                registered_count = len(handler.extrinsics)
                if source_latents.shape[1] != registered_count:
                    raise RuntimeError(
                        f"Mosaic memory has {source_latents.shape[1]} latent frames but "
                        f"the frustum handler registered {registered_count} RGB frames"
                    )

                candidate_nms_mode = getattr(args, "mosaic_candidate_nms_mode", "none")
                candidate_nms_mode = (
                    None if candidate_nms_mode in (None, "", "none", "None") else str(candidate_nms_mode)
                )
                candidate_budget = int(args.candidates_per_query_group)
                fuse_mode = _normalize_mosaic_fuse_mode(args.mosaic_fuse_mode)
                query_result = handler.query_hits_mode_new(
                    unwrapped_model.device,
                    query_extrinsics,
                    source_latents,
                    candidates_per_query_group=candidate_budget,
                    angle_threshold=None,
                    distance_threshold=None,
                    temporal_threshold=None,
                    fuse_mode=fuse_mode,
                    zbuffer_depth_preference="near",
                    interpolation_mode="nearest",
                    return_revgrid=args.mosaic_use_revgrid_rope,
                    return_candidate_frame_ids=True,
                    return_view_change=use_mosaic_view_change_prope,
                    latent_merge_4frames=False,
                    query_reference_frame=args.mosaic_query_reference_frame,
                    selection_mode=getattr(args, "mosaic_selection_mode", "projection_iou"),
                    candidate_nms_mode=candidate_nms_mode,
                    candidate_nms_projection_iou_threshold=float(args.mosaic_candidate_nms_projection_iou_threshold),
                    candidate_nms_min_temporal_gap=int(args.mosaic_candidate_nms_min_temporal_gap),
                    candidate_nms_pose_distance_threshold=float(args.mosaic_candidate_nms_pose_distance_threshold),
                    candidate_nms_pool_multiplier=float(args.mosaic_candidate_nms_pool_multiplier),
                    coverage_grid_downsample=int(getattr(args, "mosaic_coverage_grid_downsample", 4)),
                    coverage_pool_stride=int(getattr(args, "mosaic_coverage_pool_stride", 2)),
                    source_valid_masks=None,
                )

                queried_latent, mosaic_revgrid, mosaic_view_change = _parse_query_hits(
                    query_result,
                    return_revgrid=bool(args.mosaic_use_revgrid_rope),
                    return_view_change=use_mosaic_view_change_prope,
                )
                mosaic_frame_indices = _compute_mosaic_frame_indices(
                    int(queried_latent.shape[1]),
                    int(args.mosaic_interval),
                    device=queried_latent.device,
                )
                queried_latent = queried_latent.index_select(1, mosaic_frame_indices)
                selected_indices = mosaic_frame_indices.detach().cpu().numpy()
                if mosaic_revgrid is not None:
                    mosaic_revgrid = np.asarray(mosaic_revgrid)[selected_indices]
                if mosaic_view_change is not None:
                    mosaic_view_change = np.asarray(mosaic_view_change)[selected_indices]
                mosaic_latent = queried_latent.unsqueeze(0)

            section_prompt = _require_nonempty_prompt(
                camera_data.get("prompt") or data.get("prompt") or "",
                phase=f"inference section={section_idx}",
                clean_name=clean_name,
            )
            section_output = unwrapped_model.inference_step(
                prompt=section_prompt,
                negative_prompt=args.negative_prompt,
                input_image=None,
                first_frame_latents=current_clean_latents.to(
                    device=unwrapped_model.device,
                    dtype=unwrapped_model.torch_dtype,
                ),
                height=height,
                width=width,
                num_frames=4 * (latent_window_size - 1) + 1,
                seed=int(args.inference_seed) + batch_idx * 1000 + section_idx,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.guidance_scale,
                negative_no_prope=bool(getattr(args, "negative_no_prope", False)),
                negative_no_context=bool(getattr(args, "negative_no_context", False)),
                mosaic_latent=mosaic_latent,
                mosaic_revgrid=mosaic_revgrid,
                mosaic_use_revgrid_rope=args.mosaic_use_revgrid_rope,
                mosaic_view_change=mosaic_view_change,
                mosaic_view_change_prope=use_mosaic_view_change_prope,
                mosaic_frame_indices=mosaic_frame_indices,
                mosaic_drop_holes=bool(args.mosaic_drop_holes),
                tiled=vae_decode_tiled,
                prope_camera_kwargs=prope_camera_kwargs,
                latent_rope_time_indices=None,
                subject_ref_latents=(
                    data.get("subject_ref_latents")
                    if (
                        getattr(unwrapped_model, "subject_ref_memory", False)
                        and torch.is_tensor(data.get("subject_ref_latents"))
                        and int(data["subject_ref_latents"].shape[0]) > 0
                    )
                    else None
                ),
                subject_ref_slot_ratio=float(getattr(unwrapped_model, "subject_ref_canvas_slot_ratio", 0.5)),
                subject_ref_time_gap=int(getattr(unwrapped_model, "subject_ref_time_gap", 1)),
                subject_ref_prope_mode=str(
                    getattr(unwrapped_model, "subject_ref_prope_mode", "identity") or "identity"
                ),
            )
            if not isinstance(section_output, torch.Tensor):
                raise RuntimeError("Matrix inference pipeline must return latent tensors")
            section_latents = section_output.detach().cpu()
            expected_latent_count = 1 + latent_window_size
            if int(section_latents.shape[2]) != expected_latent_count:
                raise RuntimeError(
                    f"Section {section_idx} returned {section_latents.shape[2]} latent frames; "
                    f"expected {expected_latent_count}"
                )

            decoded_section = _decode_latents_to_numpy_frames(
                unwrapped_model,
                section_latents,
                unwrapped_model.device,
                tiled=vae_decode_tiled,
            )
            if decoded_section.shape[0] <= 1:
                raise RuntimeError(f"Section {section_idx} decoded no generated frames")
            last_predicting_frames = decoded_section[1:]

            if args.force_using_input_extrinics:
                noisy_extrinsics = camera_data["noisy_latent_indices_prope_extrinsic"].cpu().numpy()
                noisy_intrinsics = camera_data["noisy_latent_indices_prope_intrinsic"].cpu().numpy()
                predicted_count = int(last_predicting_frames.shape[0])
                if noisy_extrinsics.shape[0] < predicted_count:
                    raise RuntimeError(
                        f"Camera window has {noisy_extrinsics.shape[0]} poses for {predicted_count} generated frames"
                    )
                last_predicting_extrinsics = noisy_extrinsics[:predicted_count]
                last_predicting_intrinsics = noisy_intrinsics[:predicted_count]

            new_history_chunk = section_latents[:, :, 1:]
            history_latents = torch.cat([history_latents, new_history_chunk], dim=2)
            section_metadata[str(section_idx)] = {
                "used_mosaic": mosaic_latent is not None,
                "used_prope": True,
                "prompt": section_prompt,
                "subject_ref_info": data.get("subject_ref_info") or {},
                "subject_ref_latent_info": data.get("subject_ref_latent_info") or {},
            }

        history_frames = _decode_latents_to_numpy_frames(
            unwrapped_model,
            history_latents,
            unwrapped_model.device,
            tiled=vae_decode_tiled,
        )
        history_path = os.path.join(
            output_dir,
            f"run_{run_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}_history.mp4",
        )
        save_video_h264(history_frames, history_path, fps=16, crf=23, preset="medium")
        metadata_path = os.path.join(
            output_dir,
            f"run_{run_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}_sections.json",
        )
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(section_metadata, handle, ensure_ascii=False, indent=2)
        print(f"[inference][rank {proc_rank}] saved Matrix rollout to {history_path}")

        handler = None
        mosaic_query_latents = None
        last_predicting_frames = None
        section_latents = None
        history_latents = None
        history_frames = None
        _release_cached_memory()
