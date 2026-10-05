# Inference methods adapted from AlayaLab/AlayaWorld at ea03cfbb2e4c4e9102ed8ea8562e0b5370ca9b79.
# No training lifecycle is included.
"""ViGeo-conditioned autoregressive inference orchestration.

Only the transitive per-case rollout methods are retained. Model loading,
backbone layers, VAE, text and history encoding remain shared components.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from alaya.config.schema import ValidationModeConfig
from alaya.control.action import build_action_vectors, build_action_vectors_from_pixel_indices
from alaya.memory.da3_depth import DA3DepthEstimator
from alaya.memory.spatial_cache import (
    Sparse3DCache,
    build_retrieved_latent_context,
    forward_warp_indexed_pixel_sources_to_pixel_targets,
    forward_warp_pixel_sources_to_pixel_targets,
    forward_warp_video_to_targets,
    pixel_intrinsics,
    render_colored_pointmaps_to_camera_targets,
)
from alaya.memory.vigeo_geometry import ViGeoGeometryEstimator
from alaya.utils.distributed import init_distributed, rank0_print
from alaya.utils.dtype import resolve_dtype
from alaya.utils.seed import seed_everything
from ltx2.modules.perturbations import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from ltx2.modules.scheduler import LinearQuadraticScheduler, LTX2Scheduler

from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import VideoLatentPatchifier
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import VideoLatentShape


@dataclass
class _RolloutSpatialBank:
    pixels: list[torch.Tensor]
    frame_indices: list[int]
    depths: list[torch.Tensor | None]
    vigeo_pointmaps: list[torch.Tensor] = field(default_factory=list)
    vigeo_valid_masks: list[torch.Tensor] = field(default_factory=list)
    vigeo_predicted_poses: list[torch.Tensor] = field(default_factory=list)
    vigeo_intrinsics: list[torch.Tensor] = field(default_factory=list)
    vigeo_kv_caches: Any = None
    vigeo_scale: float = 1.0
    vigeo_pairwise_scales: tuple[float, ...] = ()
    vigeo_generated_chunks: int = 0
    vigeo_pixel_offset: int = 0
    causal_prefix_pixel: torch.Tensor | None = None
    causal_prefix_frame_index: int | None = None
    subject_anchor_pixel: torch.Tensor | None = None
    subject_anchor_mask: torch.Tensor | None = None
    subject_anchor_paste: bool = True
    subject_exclusion_mask: torch.Tensor | None = None


def _select_depth_by_latent_index(
    depth: torch.Tensor, latent_indices: list[int], temporal_stride: int
) -> dict[int, torch.Tensor]:
    """Select source-frame depth maps from metadata depth."""
    if not torch.is_tensor(depth):
        raise TypeError(f"metadata depth must be a tensor, got {type(depth)!r}")
    d = depth
    if d.dim() == 3:
        d = d.unsqueeze(0)
    if d.dim() == 4:
        d = d.unsqueeze(2)
    if d.dim() != 5:
        raise ValueError(f"metadata depth must be [B,T,H,W] or [B,T,1,H,W], got {tuple(depth.shape)}")
    out: dict[int, torch.Tensor] = {}
    for latent_idx in sorted({int(i) for i in latent_indices}):
        frame_idx = min(max(0, latent_idx * int(temporal_stride)), d.shape[1] - 1)
        out[int(latent_idx)] = d[:, frame_idx].contiguous()
    return out


def _select_depth_by_frame_index(depth: torch.Tensor, frame_indices: list[int]) -> dict[int, torch.Tensor]:
    """Select source-frame depth maps from metadata depth by raw pixel frame."""
    if not torch.is_tensor(depth):
        raise TypeError(f"metadata depth must be a tensor, got {type(depth)!r}")
    d = depth
    if d.dim() == 3:
        d = d.unsqueeze(0)
    if d.dim() == 4:
        d = d.unsqueeze(2)
    if d.dim() != 5:
        raise ValueError(f"metadata depth must be [B,T,H,W] or [B,T,1,H,W], got {tuple(depth.shape)}")
    out: dict[int, torch.Tensor] = {}
    for frame_idx in sorted({int(i) for i in frame_indices}):
        idx = min(max(0, int(frame_idx)), d.shape[1] - 1)
        out[int(frame_idx)] = d[:, idx].contiguous()
    return out


def _as_bool(value: Any) -> bool:
    value = _first_scalar(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.flatten()[0].item())
    return bool(value)


def _first_scalar(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return value


class AlayaV11Rollout:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dist = init_distributed()
        seed_everything(cfg.run.seed + self.dist.rank)
        self.dtype = resolve_dtype(cfg.runtime.dtype)
        self.components = None
        self.history_encoder = None
        self.da3_depth = None
        self.vigeo_geometry = None
        self._text_cache = OrderedDict()
        self._text_cache_hits = self._text_cache_misses = 0
        self._vae_cache_hits = self._vae_cache_misses = 0
        self._perf_text_s = self._perf_vae_s = self._perf_calls = 0

    def _wbench_subject_foreground_layers(
        self, *, source_pixel: torch.Tensor, metadata: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Build a camera-locked subject layer and a dilated bank exclusion mask."""
        if not _as_bool(metadata.get("wbench_lock_subject_foreground", False)):
            return None
        mask_path = str(metadata.get("wbench_subject_mask") or "")
        if not mask_path or not Path(mask_path).is_file():
            rank0_print(
                self.dist,
                "[Validation]",
                f"wbench case={metadata.get('wbench_case_id')} has no subject mask; foreground lock disabled for this sample",
            )
            return None
        try:
            from PIL import Image

            (height, width) = (int(source_pixel.shape[-2]), int(source_pixel.shape[-1]))
            mask_img = Image.open(mask_path).convert("L").resize((width, height), resample=Image.Resampling.NEAREST)
            subject_mask = torch.from_numpy(np.array(mask_img)) > 127
            if int(subject_mask.sum()) < 64:
                raise ValueError(f"subject mask has only {int(subject_mask.sum())} pixels")
            dilation = max(0, int(metadata.get("wbench_subject_mask_dilation_pixels", 12)))
            exclusion_mask = subject_mask
            if dilation > 0:
                kernel = 2 * dilation + 1
                exclusion_mask = (
                    F.max_pool2d(
                        subject_mask.float().view(1, 1, height, width), kernel_size=kernel, stride=1, padding=dilation
                    )[0, 0]
                    > 0.5
                )
            rank0_print(
                self.dist,
                "[Validation]",
                f"wbench case={metadata.get('wbench_case_id')} foreground locked subject={float(subject_mask.float().mean()):.3f} excluded={float(exclusion_mask.float().mean()):.3f}",
            )
            return (
                source_pixel.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                subject_mask.contiguous(),
                exclusion_mask.contiguous(),
            )
        except Exception as exc:
            rank0_print(self.dist, "[Validation]", f"wbench foreground lock failed (fail-open): {exc}")
            return None

    @staticmethod
    def _exclude_subject_from_valid_mask(valid_mask: torch.Tensor, exclusion_mask: torch.Tensor | None) -> torch.Tensor:
        if exclusion_mask is None:
            return valid_mask.bool().contiguous()
        excluded = exclusion_mask.to(device=valid_mask.device, dtype=torch.bool)
        while excluded.dim() < valid_mask.dim():
            excluded = excluded.unsqueeze(0)
        return (valid_mask.bool() & ~excluded.expand_as(valid_mask)).contiguous()

    def _composite_wbench_subject_anchor(
        self, *, bank: _RolloutSpatialBank, warped_pixels: torch.Tensor, coverage_pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bank.subject_anchor_pixel is None or bank.subject_anchor_mask is None:
            return (warped_pixels, coverage_pixels)
        if not getattr(bank, "subject_anchor_paste", True):
            return (warped_pixels, coverage_pixels)
        anchor = bank.subject_anchor_pixel.to(device=warped_pixels.device, dtype=warped_pixels.dtype).unsqueeze(2)
        if int(anchor.shape[0]) != int(warped_pixels.shape[0]):
            if int(anchor.shape[0]) != 1:
                raise ValueError(
                    f"subject anchor batch does not match warped pixels: {anchor.shape[0]} != {warped_pixels.shape[0]}"
                )
            anchor = anchor.expand(int(warped_pixels.shape[0]), -1, -1, -1, -1)
        mask = bank.subject_anchor_mask.to(device=warped_pixels.device, dtype=torch.bool).view(
            1, 1, 1, *bank.subject_anchor_mask.shape[-2:]
        )
        warped_pixels = torch.where(mask, anchor, warped_pixels)
        coverage_pixels = torch.where(
            mask, torch.ones((), device=coverage_pixels.device, dtype=coverage_pixels.dtype), coverage_pixels
        )
        return (warped_pixels.contiguous(), coverage_pixels.contiguous())

    @torch.no_grad()
    def _validate_rollout_sample(
        self,
        *,
        video_pixels: torch.Tensor,
        latent_full: torch.Tensor,
        context: torch.Tensor,
        scheduled_contexts: list[torch.Tensor] | None,
        scheduled_prompt_captions: list[str] | None,
        negative_context: torch.Tensor | None,
        metadata: dict[str, Any],
        mode_cfg: ValidationModeConfig,
        K: int,
        rounds: int,
        N: int,
        gap_steps: int,
        cond_end: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        assert self.components is not None
        (B, _, _, H_lat, W_lat) = latent_full.shape
        sink_count = self.cfg.layout.sink_latent_frames
        explicit_condition = cond_end if N == 0 else 0
        sink_latent = latent_full[:, :, :sink_count].contiguous() if sink_count > 0 else None
        hist_start = sink_count + gap_steps
        hist_end = hist_start + N
        condition_start = hist_end
        target_base_start = condition_start + explicit_condition
        history = latent_full[:, :, hist_start:hist_end].clone().contiguous() if N > 0 else None
        history_action_t_indices = (
            torch.arange(hist_start, hist_end, device=self.dist.device, dtype=torch.float32)
            if bool(getattr(self.cfg.control, "action_history_memory", False)) and N > 0
            else None
        )
        explicit_nearby = (
            latent_full[:, :, condition_start:target_base_start].contiguous() if explicit_condition > 0 else None
        )
        vigeo_prefix_mode = self._uses_vigeo_prefix_last_frame()
        validation_prefix_last_pixel: int | None = None
        if vigeo_prefix_mode:
            prefix_frames = self._vigeo_target_prefix_pixel_frames(history_latent_frames=N)
            validation_prefix_last_pixel = prefix_frames - 1
            sink_latent = latent_full[:, :, :sink_count].contiguous()
            explicit_nearby = latent_full[:, :, condition_start:target_base_start].contiguous()
            motion_start = prefix_frames - self._vigeo_motion_pixel_frames()
            (rollout_decode_anchor, rollout_motion_nearby) = self._encode_vigeo_motion_window(
                self._slice_video_pixel_frames(video_pixels, motion_start, prefix_frames)
            )
            explicit_nearby = rollout_motion_nearby
        else:
            rollout_decode_anchor = None
            rollout_motion_nearby = None
        sink_indices = (
            self._indices_grid(B, sink_count, H_lat, W_lat, t_offset=self._local_sink_t_offset(N))
            if sink_count > 0
            else None
        )
        sigmas = self._validation_sigmas(latent_frames=int(K))
        stg_perturbations = self._validation_stg_perturbations()
        actual_cfg_scale = self._validation_cfg_scale()
        metrics = []
        pred_latents = []
        nearby_condition_latents: list[torch.Tensor | None] = []
        spatial_condition_latents: list[torch.Tensor | None] = []
        spatial_condition_masks: list[torch.Tensor | None] = []
        spatial_condition_prefix_latents: list[torch.Tensor | None] = []
        spatial_bank = self._init_validation_rollout_spatial_bank(
            video_pixels=video_pixels, metadata=metadata, target_start=target_base_start, history_latent_frames=N
        )
        for round_idx in range(rounds):
            round_context = context
            if scheduled_contexts:
                round_context = scheduled_contexts[min(round_idx, len(scheduled_contexts) - 1)]
            current_target_start = target_base_start + round_idx * K
            print(
                f"[Rollout] rank={self.dist.rank} src={mode_cfg.dataset.source} video={metadata.get('video_id', '?')} round={round_idx + 1}/{rounds} target_start={current_target_start}",
                flush=True,
            )
            current_target_rope_t_indices = self._local_target_t_indices(
                K, history_latent_frames=N, condition_latent_frames=cond_end, gap_steps=gap_steps
            )
            current_target_action_t_indices = torch.arange(
                current_target_start, current_target_start + K, device=self.dist.device, dtype=torch.float32
            )
            mem_tokens = None
            mem_indices = None
            round_uses_memory = bool(mode_cfg.use_memory) and round_idx >= int(mode_cfg.memory_start_round)
            if history is not None and round_uses_memory:
                assert self.history_encoder is not None
                (mem_tokens, mem_indices) = self.history_encoder(history)
                mem_indices = mem_indices.clone()
                history_t_offset = self._local_memory_t_offset(N, cond_end)
                mem_indices[:, 0, :, :] += history_t_offset
            if vigeo_prefix_mode:
                if rollout_motion_nearby is None:
                    raise RuntimeError("ViGeo rollout is missing its RGB motion nearby latent")
                nearby_latent = rollout_motion_nearby
            elif cond_end <= 0:
                nearby_latent = None
            elif history is not None:
                nearby_latent = history[:, :, -cond_end:].contiguous()
            elif round_idx == 0:
                nearby_latent = explicit_nearby
            else:
                nearby_latent = pred_latents[-1][:, :, -cond_end:].to(latent_full.dtype).contiguous()
            nearby_indices = (
                self._indices_grid(
                    B, cond_end, H_lat, W_lat, t_offset=self._local_nearby_t_offset(N, cond_end, gap_steps=gap_steps)
                )
                if cond_end > 0
                else None
            )
            _r20_drop = {int(x) for x in metadata.get("wbench_spatial_drop_rounds") or []}
            if int(round_idx) in _r20_drop:
                spatial_context = None
            elif spatial_bank is not None:
                spatial_context = self._build_validation_rollout_bank_spatial_context(
                    bank=spatial_bank,
                    metadata=metadata,
                    target_start=current_target_start,
                    K=K,
                    target_rope_t_indices=current_target_rope_t_indices,
                )
            else:
                spatial_context = self._build_spatial_context(
                    latent_full=latent_full,
                    video_pixels=video_pixels,
                    metadata=metadata,
                    target_start=current_target_start,
                    K=K,
                    cond_end=cond_end,
                    target_rope_t_indices=current_target_rope_t_indices,
                )
            force_spatial_invalid = False
            if force_spatial_invalid:
                spatial_context = self._maybe_force_spatial_all_invalid(spatial_context, force=True)
            spatial_latent = spatial_context["latent"] if spatial_context is not None else None
            spatial_mask_patch = spatial_context.get("mask_patch") if spatial_context is not None else None
            spatial_indices = (
                self._indices_grid_for_t_indices(
                    B, spatial_context.get("rope_t_indices", spatial_context["target_indices"]), H_lat, W_lat
                )
                if spatial_context is not None
                else None
            )
            decoder_prefix = nearby_latent
            if vigeo_prefix_mode:
                if rollout_decode_anchor is None or nearby_latent is None:
                    raise RuntimeError("ViGeo rollout is missing its decoder motion prefix")
                decoder_prefix = torch.cat([rollout_decode_anchor, nearby_latent], dim=2).contiguous()
            nearby_condition_latents.append(
                None if decoder_prefix is None else decoder_prefix.detach().to(dtype=latent_full.dtype).cpu()
            )
            spatial_condition_latents.append(
                None if spatial_latent is None else spatial_latent.detach().to(dtype=latent_full.dtype).cpu()
            )
            spatial_condition_masks.append(None if spatial_mask_patch is None else spatial_mask_patch.detach().cpu())
            spatial_prefix_latent = spatial_context.get("vae_prefix_latent") if spatial_context is not None else None
            spatial_condition_prefix_latents.append(
                None
                if spatial_prefix_latent is None
                else spatial_prefix_latent.detach().to(dtype=latent_full.dtype).cpu()
            )
            if vigeo_prefix_mode:
                stride = int(self.cfg.sample.temporal_stride)
                target_pixel_start = self._validation_vigeo_target_pixel_start(
                    target_start=current_target_start, target_base_start=target_base_start
                )
                previous_pixel = (
                    int(validation_prefix_last_pixel)
                    if round_idx == 0 and validation_prefix_last_pixel is not None
                    else target_pixel_start - 1
                )
                control_kwargs = self._build_explicit_pixel_control_kwargs(
                    metadata=metadata,
                    control_modes=list(mode_cfg.control),
                    target_pixel_start=target_pixel_start + stride - 1,
                    target_latent_frames=K,
                    nearby_pixel=previous_pixel,
                    dtype=self.dtype,
                )
            else:
                control_kwargs = self._build_control_kwargs(
                    metadata=metadata,
                    control_modes=list(mode_cfg.control),
                    target_t_indices=current_target_action_t_indices,
                    condition_t_indices=torch.arange(
                        current_target_start - cond_end,
                        current_target_start,
                        device=self.dist.device,
                        dtype=torch.float32,
                    )
                    if cond_end > 0
                    else None,
                    history_t_indices=history_action_t_indices
                    if mem_tokens is not None and bool(mode_cfg.use_memory)
                    else None,
                    dtype=self.dtype,
                )
            x_t = torch.randn(B, latent_full.shape[1], K, H_lat, W_lat, device=self.dist.device, dtype=self.dtype)
            for sample_step in range(len(sigmas) - 1):
                sigma_now = sigmas[sample_step]
                sigma_next = sigmas[sample_step + 1]

                def _forward_velocity(
                    context_tensor: torch.Tensor,
                    *,
                    control: dict[str, torch.Tensor],
                    perturbations: BatchedPerturbationConfig | None = None,
                ) -> torch.Tensor:
                    return self.components.transformer(
                        x=[x_t.squeeze(0)],
                        t=(sigma_now * 1000.0).view(1).to(device=self.dist.device, dtype=x_t.dtype),
                        context=[context_tensor],
                        seq_len=K * H_lat * W_lat,
                        fps=self.cfg.sample.fps,
                        perturbations=perturbations,
                        history_kv_tokens=mem_tokens,
                        history_indices_grid=mem_indices,
                        gen_t_indices_override=current_target_rope_t_indices,
                        sink_latent=sink_latent,
                        sink_indices_grid=sink_indices,
                        spatial_latent=spatial_latent,
                        spatial_mask_patch=spatial_mask_patch,
                        spatial_indices_grid=spatial_indices,
                        nearby_latent=nearby_latent,
                        nearby_indices_grid=nearby_indices,
                        **control,
                    )

                pos_v = _forward_velocity(round_context, control=control_kwargs)
                pred_v = pos_v
                if actual_cfg_scale > 1.0 and negative_context is not None:
                    neg_v = _forward_velocity(negative_context, control=control_kwargs)
                    pred_v = pred_v + (actual_cfg_scale - 1.0) * (pos_v - neg_v)
                action_cfg_scale = float(mode_cfg.action_cfg_scale)
                if action_cfg_scale > 1.0 and "action_vectors" in control_kwargs:
                    no_action_kwargs: dict[str, torch.Tensor] = {}
                    no_action_pos_v = _forward_velocity(round_context, control=no_action_kwargs)
                    no_action_v = no_action_pos_v
                    if actual_cfg_scale > 1.0 and negative_context is not None:
                        no_action_neg_v = _forward_velocity(negative_context, control=no_action_kwargs)
                        no_action_v = no_action_v + (actual_cfg_scale - 1.0) * (no_action_pos_v - no_action_neg_v)
                    pred_v = no_action_v + action_cfg_scale * (pred_v - no_action_v)
                if stg_perturbations is not None:
                    ptb_v = _forward_velocity(round_context, control=control_kwargs, perturbations=stg_perturbations)
                    pred_v = pred_v + float(self.cfg.validation.stg_scale) * (pos_v - ptb_v)
                if self.cfg.validation.rescale_scale > 0.0 and pred_v is not pos_v:
                    factor = pos_v.float().std() / (pred_v.float().std() + 1e-08)
                    factor = float(self.cfg.validation.rescale_scale) * factor + (
                        1.0 - float(self.cfg.validation.rescale_scale)
                    )
                    pred_v = pred_v * factor
                if sigma_next.item() > 1e-05:
                    dt = (sigma_now - sigma_next).to(dtype=x_t.dtype)
                    x_t = x_t - dt * pred_v
                else:
                    x_t = (x_t.float() - pred_v.float() * sigma_now.float()).to(x_t.dtype)
            pred = x_t
            if (
                bool(getattr(self.cfg.validation, "vigeo_seam_dc_correct", False))
                and vigeo_prefix_mode
                and (nearby_latent is not None)
            ):
                _ref = nearby_latent.to(device=pred.device, dtype=pred.dtype)
                _dc_ref = _ref.mean(dim=(-2, -1), keepdim=True)
                _first = pred[:, :, :1]
                _dc_first = _first.mean(dim=(-2, -1), keepdim=True)
                pred = torch.cat([_first + (_dc_ref - _dc_first), pred[:, :, 1:]], dim=2).contiguous()
            decoded_target_pixels = None
            next_decode_anchor = None
            next_motion_nearby = None
            if vigeo_prefix_mode:
                if rollout_decode_anchor is None or nearby_latent is None:
                    raise RuntimeError("ViGeo rollout cannot decode without its RGB motion prefix")
                (decoded_target_pixels, next_decode_anchor, next_motion_nearby) = (
                    self._decode_and_reencode_vigeo_motion_chunk(
                        anchor_latent=rollout_decode_anchor, motion_latent=nearby_latent, target_latent=pred.detach()
                    )
                )
            pred_latents.append(pred.detach())
            real_start = current_target_start
            real_end = real_start + K
            gt = latent_full[:, :, real_start:real_end]
            metric: dict[str, Any] = {
                "round": round_idx,
                "drift_frames": int(round_idx * K),
                "gt_latent_frames": int(gt.shape[2]),
                "gt_available": bool(gt.shape[2] == K),
                "nearby_from_rgb_motion_window": bool(vigeo_prefix_mode),
                "vigeo_handoff_mode": str(self.cfg.validation.vigeo_handoff_mode),
                "vigeo_pred_decode_mode": str(self.cfg.validation.vigeo_pred_decode_mode),
                "spatial_condition_available": bool(spatial_latent is not None),
                "spatial_forced_invalid": bool(force_spatial_invalid),
            }
            prompt_label = self._validation_prompt_label(mode_cfg=mode_cfg, round_idx=round_idx)
            if prompt_label is not None:
                metric["prompt_label"] = prompt_label
            if scheduled_prompt_captions:
                metric["prompt"] = scheduled_prompt_captions[min(round_idx, len(scheduled_prompt_captions) - 1)]
            skip_spatial_bank_append = False
            metric["spatial_bank_appended"] = bool(spatial_bank is not None and (not skip_spatial_bank_append))
            if gt.shape[2] == K:
                diff = pred.float() - gt.float()
                l2 = diff.pow(2).mean().sqrt().item()
                cos = F.cosine_similarity(pred.float().flatten(), gt.float().flatten(), dim=0).item()
                metric["l2"] = float(l2)
                metric["cos"] = float(cos)
            else:
                metric["l2"] = None
                metric["cos"] = None
            metrics.append(metric)
            if spatial_bank is not None and vigeo_prefix_mode:
                self._record_validation_vigeo_causal_prefix(
                    bank=spatial_bank, decoded_pixels=decoded_target_pixels, target_start=current_target_start
                )
            if spatial_bank is not None and (not skip_spatial_bank_append):
                self._append_validation_rollout_spatial_bank_prediction(
                    bank=spatial_bank,
                    pred_latent=pred.detach(),
                    decoded_pixels=decoded_target_pixels,
                    metadata=metadata,
                    target_start=current_target_start,
                )
            if vigeo_prefix_mode:
                if next_decode_anchor is None or next_motion_nearby is None:
                    raise RuntimeError("ViGeo rollout failed to produce its next RGB motion prefix")
                rollout_decode_anchor = next_decode_anchor
                rollout_motion_nearby = next_motion_nearby
            if history is not None:
                history = torch.cat([history, pred.to(history.dtype)], dim=2)[:, :, -N:].contiguous()
                if history.shape[2] != N:
                    raise RuntimeError(f"validation history length changed: {history.shape[2]} != {N}")
                if history_action_t_indices is not None:
                    history_action_t_indices = torch.cat(
                        [history_action_t_indices, current_target_action_t_indices], dim=0
                    )[-N:].contiguous()
        return (
            metrics,
            pred_latents,
            nearby_condition_latents,
            spatial_condition_latents,
            spatial_condition_masks,
            spatial_condition_prefix_latents,
        )

    def _validation_sigmas(self, *, latent_frames: int | None = None) -> torch.Tensor:
        steps = int(self.cfg.validation.sampling_steps)
        scheduler = self.cfg.validation.scheduler
        if scheduler == "uniform":
            sigmas = torch.linspace(1.0, 0.0, steps + 1)
        elif scheduler == "linear_quadratic":
            sigmas = LinearQuadraticScheduler().execute(steps=steps)
        else:
            if bool(getattr(self.cfg.sigma_shift, "adaptive_sigma_shift", False)):
                m = self._adaptive_sigma_shift_m(latent_frames or steps)
                shift = math.log(max(float(m), 1e-06))
                max_shift = shift
                base_shift = shift
            else:
                max_shift = 2.05
                base_shift = 0.95
            sigmas = LTX2Scheduler().execute(
                steps=steps, latent=None, max_shift=max_shift, base_shift=base_shift, stretch=True, terminal=0.1
            )
        return sigmas.to(device=self.dist.device, dtype=torch.float32)

    def _validation_cfg_scale(self) -> float:
        cfg_scale = float(self.cfg.validation.cfg_scale)
        return 3.0 if cfg_scale <= 1.0 else cfg_scale

    def _validation_stg_perturbations(self) -> BatchedPerturbationConfig | None:
        if self.cfg.validation.stg_scale <= 0.0 or not self.cfg.validation.stg_blocks:
            return None
        return BatchedPerturbationConfig(
            perturbations=[
                PerturbationConfig(
                    perturbations=[
                        Perturbation(
                            type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                            blocks=[int(block) for block in self.cfg.validation.stg_blocks],
                        )
                    ]
                )
            ]
        )

    def _decode_i2v_rollout_chunks_to_video_frames(
        self, *, nearby_latents: list[torch.Tensor | None], chunks: list[torch.Tensor]
    ) -> torch.Tensor:
        if not chunks or len(nearby_latents) != len(chunks):
            raise ValueError("i2v rollout decode requires one nearby latent per target chunk")
        _dump_dir = os.environ.get("ALAYA_DUMP_PRED_LATENTS", "")
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)
            _tag = f"rank{self.dist.rank}_{len(os.listdir(_dump_dir))}"
            torch.save(
                [c.detach().to("cpu", torch.float32) for c in chunks], os.path.join(_dump_dir, f"pred_chunks_{_tag}.pt")
            )
        decoded_chunks: list[torch.Tensor] = []
        stride = int(self.cfg.sample.temporal_stride)
        for index, (nearby, chunk) in enumerate(zip(nearby_latents, chunks)):
            if nearby is None or int(nearby.shape[2]) <= 0:
                raise ValueError(f"i2v rollout chunk {index} requires a non-empty decoder prefix")
            prefix_latents = int(nearby.shape[2])
            context_pixels = 1 + (prefix_latents - 1) * stride
            local_latent = torch.cat([nearby.to(dtype=chunk.dtype, device=chunk.device), chunk], dim=2).contiguous()
            local_frames = self._decode_latent_to_video_frames(local_latent)
            expected_frames = context_pixels + int(chunk.shape[2]) * stride
            if int(local_frames.shape[0]) != expected_frames:
                raise RuntimeError(
                    f"i2v rollout chunk {index} decoded {local_frames.shape[0]} frames, expected {expected_frames}"
                )
            crop = context_pixels - 1 if index == 0 else context_pixels
            decoded_chunks.append(local_frames[crop:])
        return torch.cat(decoded_chunks, dim=0)

    def _decode_latent_to_video_frames(self, latent: torch.Tensor) -> torch.Tensor:
        assert self.components is not None
        decoder = self.components.vae_decoder
        decode_chunk = self.cfg.runtime.vae_decode_chunk_latents
        chunk_latents = max(1, int(decode_chunk if decode_chunk is not None else self.cfg.runtime.vae_chunk_size))
        frames = []
        total_latents = int(latent.shape[2])
        for start in range(0, total_latents, chunk_latents):
            end = min(total_latents, start + chunk_latents)
            chunk = latent[:, :, start:end].to(device=self.dist.device, dtype=self.dtype)
            with torch.no_grad():
                pixel = decoder(chunk)
            pixel = (pixel * 0.5 + 0.5).clamp(0, 1)
            chunk_frames = pixel.squeeze(0).permute(1, 2, 3, 0).contiguous()
            if start > 0 and chunk_frames.shape[0] > 0:
                chunk_frames = chunk_frames[1:]
            frames.append((chunk_frames * 255.0).to(torch.uint8).cpu())
            del chunk, pixel, chunk_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(frames, dim=0)

    def _decode_latent_to_bank_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        assert self.components is not None
        chunk = latent.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            pixel = self.components.vae_decoder(chunk)
        return pixel.detach().to(device=self.dist.device, dtype=self.dtype).contiguous()

    def _encode_vigeo_motion_window(self, video_pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion_pixels = self._vigeo_motion_pixel_frames()
        actual_pixels = self._video_pixel_frame_count(video_pixels)
        if actual_pixels != motion_pixels:
            raise ValueError(f"ViGeo motion window needs {motion_pixels} pixels, got {actual_pixels}")
        latent = self._encode_video(video_pixels, needed_latents=2)
        if int(latent.shape[2]) != 2:
            raise RuntimeError(f"ViGeo motion window VAE produced {latent.shape[2]} latents, expected 2")
        return (latent[:, :, :1].contiguous(), latent[:, :, 1:2].contiguous())

    def _decode_and_reencode_vigeo_motion_chunk(
        self, *, anchor_latent: torch.Tensor, motion_latent: torch.Tensor, target_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(anchor_latent.shape[2]) != 1 or int(motion_latent.shape[2]) != 1:
            raise ValueError("ViGeo motion decode requires one anchor and one motion latent")
        continuation = torch.cat([anchor_latent, motion_latent, target_latent], dim=2).contiguous()
        pixels = self._decode_latent_to_bank_pixels(continuation)
        motion_pixels = self._vigeo_motion_pixel_frames()
        expected_frames = motion_pixels + int(target_latent.shape[2]) * int(self.cfg.sample.temporal_stride)
        if int(pixels.shape[2]) != expected_frames:
            raise RuntimeError(
                f"ViGeo motion continuation decode length mismatch: got {pixels.shape[2]}, expected {expected_frames}"
            )
        target_pixels = pixels[:, :, motion_pixels:].contiguous()
        handoff_mode = str(self.cfg.validation.vigeo_handoff_mode)
        if handoff_mode == "rgb_reencode":
            (next_anchor, next_motion) = self._encode_vigeo_motion_window(pixels[:, :, -motion_pixels:].contiguous())
        elif handoff_mode == "direct_latent":
            next_anchor = continuation[:, :, -2:-1].contiguous()
            next_motion = continuation[:, :, -1:].contiguous()
        else:
            raise ValueError(f"unsupported ViGeo handoff mode: {handoff_mode}")
        return (target_pixels, next_anchor, next_motion)

    def _build_vigeo_validation_latent_full(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        required_latents: int,
        target_base_start: int,
        history_latent_frames: int,
        allow_short: bool,
        allow_empty_target: bool = False,
    ) -> torch.Tensor:
        """Pack sink, optional history, and the GT target continuation."""
        sink_count = int(self.cfg.layout.sink_latent_frames)
        history_latents = max(0, int(history_latent_frames))
        prefix_frames = self._vigeo_target_prefix_pixel_frames(history_latent_frames=history_latents)
        expected_target_base = sink_count + (history_latents if history_latents > 0 else 1)
        if sink_count != 1 or int(target_base_start) != expected_target_base:
            raise ValueError(
                f"ViGeo validation packing requires one sink followed by history or one explicit nearby latent: expected target_base_start={expected_target_base}, got sink={sink_count}, history={history_latents}, target_base_start={target_base_start}"
            )
        if prefix_frames < 1:
            raise ValueError("ViGeo validation requires at least one prefix pixel frame")
        target_latents = max(1, int(required_latents) - int(target_base_start))
        stride = int(self.cfg.sample.temporal_stride)
        target_pixels = target_latents * stride
        available_pixels = self._video_pixel_frame_count(video_pixels)
        requested_pixel_end = prefix_frames + target_pixels
        if available_pixels < requested_pixel_end and bool(getattr(self.cfg.layout, "variable_length", False)):
            video_bcfhw = self._video_pixels_to_bcfhw(video_pixels)
            pad = requested_pixel_end - available_pixels
            last = video_bcfhw[:, :, -1:].expand(-1, -1, pad, -1, -1)
            video_pixels = torch.cat([video_bcfhw, last], dim=2)
            available_pixels = requested_pixel_end
        target_pixel_end = requested_pixel_end
        if available_pixels < requested_pixel_end and (not allow_short):
            raise ValueError(
                f"ViGeo validation target stream is too short: need pixels [0,{requested_pixel_end}), got {available_pixels}"
            )
        if available_pixels < requested_pixel_end:
            available_target_pixels = available_pixels - prefix_frames
            compatible_target_pixels = stride * (available_target_pixels // stride)
            if compatible_target_pixels <= 0:
                if not allow_empty_target:
                    raise ValueError(
                        f"ViGeo validation clip has no complete target latent after its prefix: prefix={prefix_frames}, available={available_pixels}"
                    )
                compatible_target_pixels = 0
            target_pixel_end = prefix_frames + compatible_target_pixels
        metadata["validation_video_available_frames"] = int(available_pixels)
        metadata["validation_video_requested_frames"] = int(requested_pixel_end)
        metadata["validation_video_used_frames"] = int(target_pixel_end)
        video_key = str(metadata.get("video_id", ""))
        sink_pixel_index = sum(video_key.encode("utf-8")) % prefix_frames
        sink_latent = self._encode_video(
            self._slice_video_pixel_frames(video_pixels, sink_pixel_index, sink_pixel_index + 1), needed_latents=1
        )
        history_latent = None
        if history_latents > 0:
            history_latent = self._encode_video(
                self._slice_video_pixel_frames(video_pixels, 0, prefix_frames), needed_latents=history_latents
            )
            if int(history_latent.shape[2]) != history_latents:
                raise RuntimeError(
                    f"ViGeo validation history VAE length mismatch: got {history_latent.shape[2]}, expected {history_latents}"
                )
        used_target_latents = (target_pixel_end - prefix_frames) // stride
        motion_pixels = self._vigeo_motion_pixel_frames()
        continuation_latent = self._encode_video(
            self._slice_video_pixel_frames(video_pixels, prefix_frames - motion_pixels, target_pixel_end),
            needed_latents=used_target_latents + 2,
        )
        if int(continuation_latent.shape[2]) != used_target_latents + 2:
            raise RuntimeError(
                f"ViGeo validation continuation VAE length mismatch: got {continuation_latent.shape[2]}, expected {used_target_latents + 2}"
            )
        nearby_latent = continuation_latent[:, :, 1:2].contiguous()
        target_latent = continuation_latent[:, :, 2:].contiguous()
        prefix_latent = history_latent if history_latent is not None else nearby_latent
        return torch.cat([sink_latent, prefix_latent, target_latent], dim=2).contiguous()

    def _validation_vigeo_target_pixel_start(self, *, target_start: int, target_base_start: int | None = None) -> int:
        if target_base_start is None:
            history_latents = int(self.cfg.layout.history_latent_frames)
            target_base_start = int(self.cfg.layout.sink_latent_frames) + (
                history_latents if history_latents > 0 else 1
            )
        return self._vigeo_target_prefix_pixel_frames() + (int(target_start) - int(target_base_start)) * int(
            self.cfg.sample.temporal_stride
        )

    def _init_validation_rollout_spatial_bank(
        self, *, video_pixels: torch.Tensor, metadata: dict[str, Any], target_start: int, history_latent_frames: int
    ) -> _RolloutSpatialBank | None:
        cfg = self.cfg.spatial_memory
        context_mode = str(getattr(cfg, "context_mode", "retrieval"))
        if not bool(cfg.enabled) or context_mode not in {"target_prefix_pixels", "vigeo_prefix_last_frame"}:
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        stride = int(self.cfg.sample.temporal_stride)
        history_pixels = (
            self._vigeo_prefix_pixel_frames()
            if context_mode == "vigeo_prefix_last_frame"
            else max(1, int(cfg.num_context_frames))
        )
        fixed_single_frame_scale = context_mode == "vigeo_prefix_last_frame" and str(
            metadata.get("source") or ""
        ).lower() in {"arena", "wbench_navi", "custom_i2v"}
        if context_mode == "vigeo_prefix_last_frame":
            target_pixel_start = self._vigeo_target_prefix_pixel_frames(history_latent_frames=history_latent_frames)
            source_indices = (
                [target_pixel_start - 1]
                if fixed_single_frame_scale
                else self._vigeo_scale_context_pixel_indices(history_latent_frames=history_latent_frames)
            )
        else:
            target_pixel_start = int(target_start) * stride
            source_floor = 0
            if not bool(cfg.include_sink):
                source_floor = max(0, int(self.cfg.layout.sink_latent_frames) * stride)
            source_start = max(source_floor, target_pixel_start - history_pixels)
            source_indices = list(range(source_start, target_pixel_start))
        video_frames = self._video_pixel_frame_count(video_pixels)
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        max_frames = min(video_frames, cam_frames)
        source_indices = [idx for idx in source_indices if 0 <= int(idx) < max_frames]
        if not source_indices:
            return None
        if (
            not fixed_single_frame_scale
            and bool(getattr(cfg, "require_full_context", True))
            and (len(source_indices) < history_pixels)
        ):
            return None
        pixels = self._select_video_pixel_frames(video_pixels, source_indices)
        if context_mode == "vigeo_prefix_last_frame":
            geometry = self._get_vigeo_geometry().infer_stream_geometry(
                video_pixels=pixels,
                kv_caches=None,
                reset_cache=True,
                chunk_size=int(cfg.vigeo_stream_chunk_size),
                total_budget=int(cfg.vigeo_cache_budget),
            )
            if fixed_single_frame_scale:
                scale = float(cfg.vigeo_single_frame_scale)
                pairwise_scales: tuple[float, ...] = ()
            else:
                (scale, pairwise_scales) = self._get_vigeo_geometry().estimate_translation_scale(
                    predicted_poses=geometry.predicted_poses,
                    cam_c2w=cam_c2w,
                    frame_indices=source_indices,
                    fallback_scale=float(cfg.vigeo_single_frame_scale),
                )
            stored_pixels = pixels.detach().to(device="cpu", dtype=torch.float16)
            subject_layers = self._wbench_subject_foreground_layers(
                source_pixel=stored_pixels[:, :, -1], metadata=metadata
            )
            if subject_layers is None:
                subject_anchor_pixel = None
                subject_anchor_mask = None
                subject_exclusion_mask = None
            else:
                (subject_anchor_pixel, subject_anchor_mask, subject_exclusion_mask) = subject_layers
            return _RolloutSpatialBank(
                pixels=[stored_pixels[:, :, i].contiguous() for i in range(int(stored_pixels.shape[2]))],
                frame_indices=[int(i) for i in source_indices],
                depths=[None for _ in source_indices],
                vigeo_pointmaps=[pointmap.to(dtype=torch.float16).contiguous() for pointmap in geometry.pointmaps],
                vigeo_valid_masks=[
                    self._exclude_subject_from_valid_mask(mask, subject_exclusion_mask) for mask in geometry.valid_masks
                ],
                vigeo_predicted_poses=[pose.float().contiguous() for pose in geometry.predicted_poses],
                vigeo_intrinsics=[K.float().contiguous() for K in geometry.intrinsics],
                vigeo_kv_caches=geometry.kv_caches,
                vigeo_scale=float(scale),
                vigeo_pairwise_scales=tuple(pairwise_scales),
                vigeo_generated_chunks=0,
                vigeo_pixel_offset=int(target_pixel_start) - int(target_start) * stride,
                subject_anchor_pixel=subject_anchor_pixel,
                subject_anchor_mask=subject_anchor_mask,
                subject_exclusion_mask=subject_exclusion_mask,
            )
        depth_by_local = self._infer_validation_bank_depths(
            pixels=pixels, metadata=metadata, frame_indices=source_indices
        )
        return _RolloutSpatialBank(
            pixels=[pixels[:, :, i].detach().contiguous() for i in range(int(pixels.shape[2]))],
            frame_indices=[int(i) for i in source_indices],
            depths=[depth_by_local.get(i) if depth_by_local is not None else None for i in range(int(pixels.shape[2]))],
        )

    def _append_validation_rollout_spatial_bank_prediction(
        self,
        *,
        bank: _RolloutSpatialBank,
        pred_latent: torch.Tensor,
        decoded_pixels: torch.Tensor | None,
        metadata: dict[str, Any],
        target_start: int,
    ) -> None:
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is None:
            return
        stride = int(self.cfg.sample.temporal_stride)
        if self._uses_vigeo_prefix_last_frame():
            if decoded_pixels is None:
                raise RuntimeError("ViGeo rollout bank append requires the already decoded target pixels")
            pixels = decoded_pixels
        else:
            pixels = self._decode_latent_to_bank_pixels(pred_latent)
        frame_count = int(pixels.shape[2])
        if self._uses_vigeo_prefix_last_frame():
            cfg = self.cfg.spatial_memory
            if not bank.vigeo_intrinsics:
                raise RuntimeError("ViGeo rollout bank has no initialized intrinsic")
            target_pixel_start = int(target_start) * stride + int(bank.vigeo_pixel_offset)
            frame_indices = list(range(target_pixel_start, target_pixel_start + frame_count))
            geometry = self._get_vigeo_geometry().infer_stream_geometry(
                video_pixels=pixels,
                kv_caches=bank.vigeo_kv_caches,
                shared_intrinsic=bank.vigeo_intrinsics[0],
                reset_cache=False,
                chunk_size=int(cfg.vigeo_stream_chunk_size),
                total_budget=int(cfg.vigeo_cache_budget),
            )
            if int(geometry.pointmaps.shape[0]) != frame_count:
                raise RuntimeError(
                    f"ViGeo rollout append length mismatch: geometry={geometry.pointmaps.shape[0]} decoded={frame_count}"
                )
            _dyn_masks = None
            if bool(getattr(cfg, "vigeo_dynamic_mask", False)) and len(bank.pixels) > 0:
                try:
                    with torch.no_grad():
                        (_h, _w) = (int(pixels.shape[-2]), int(pixels.shape[-1]))
                        _src_idx = len(bank.pixels) - 1
                        _cam4 = cam_c2w if cam_c2w.dim() == 4 else cam_c2w.unsqueeze(0)
                        _src_all_c2w = self._validation_vigeo_bank_source_c2w(bank=bank, cam_c2w=_cam4)
                        _sel = torch.tensor([_src_idx], device=_src_all_c2w.device, dtype=torch.long)
                        (_tgt_c2w, _tgt_K) = self._validation_vigeo_target_cameras(
                            bank=bank,
                            metadata=metadata,
                            cam_c2w=_cam4,
                            intrinsic=metadata["intrinsic"],
                            target_pixel_indices=[int(i) for i in frame_indices],
                            height=_h,
                            width=_w,
                        )
                        _wr = render_colored_pointmaps_to_camera_targets(
                            source_pixels=bank.pixels[_src_idx]
                            .unsqueeze(2)
                            .to(device=self.dist.device, dtype=self.dtype),
                            source_pointmaps=bank.vigeo_pointmaps[_src_idx]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(device=self.dist.device, dtype=torch.float32)
                            * float(bank.vigeo_scale),
                            source_valid_masks=bank.vigeo_valid_masks[_src_idx]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(device=self.dist.device, dtype=torch.bool),
                            source_c2w=_src_all_c2w.index_select(1, _sel),
                            target_c2w=_tgt_c2w,
                            target_intrinsic=_tgt_K,
                            height=_h,
                            width=_w,
                            depth_threshold=min(float(cfg.retrieval_depth_threshold), 0.001),
                            fill_value=None,
                            return_coverage=True,
                        )
                        if _wr is not None:
                            (_warped, _cov) = _wr
                            _px = pixels.to(device=_warped.device, dtype=torch.float32)
                            _res = (_warped.float() - _px).abs().mean(dim=1).reshape(frame_count, _h, _w)
                            _covf = _cov.float().reshape(frame_count, _h, _w)
                            _tau = float(getattr(cfg, "vigeo_dynamic_mask_threshold", 0.25))
                            _motion = ((_res > _tau) & (_covf > 0.5)).float()
                            _motion = (
                                F.max_pool2d(_motion.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1) > 0.5
                            )
                            _dyn_masks = _motion.to(device="cpu")
                except Exception as _e:
                    _dyn_masks = None
                    print(f"[vigeo_dynamic_mask] skip (fail-open): {_e}", flush=True)
            stored_pixels = pixels.detach().to(device="cpu", dtype=torch.float16)
            if (
                _as_bool(metadata.get("wbench_subject_anchor_refresh", False))
                and bank.subject_anchor_pixel is not None
                and (bank.subject_anchor_mask is not None)
                and (int(stored_pixels.shape[2]) > 0)
            ):
                try:
                    _a = float(metadata.get("wbench_subject_anchor_refresh_alpha", 1.0) or 1.0)
                    _new = stored_pixels[:, :, -1]
                    if (
                        0.0 < _a < 1.0
                        and bank.subject_anchor_pixel is not None
                        and (bank.subject_anchor_pixel.shape == _new.shape)
                    ):
                        bank.subject_anchor_pixel = (
                            (_a * _new.float() + (1.0 - _a) * bank.subject_anchor_pixel.float())
                            .to(dtype=_new.dtype)
                            .contiguous()
                        )
                    else:
                        bank.subject_anchor_pixel = _new.contiguous()
                except Exception as _e:
                    print(f"[subject_anchor_refresh] skip (fail-open): {_e}", flush=True)
            if (
                _as_bool(metadata.get("wbench_subject_anchor_adaptive", False))
                and bank.subject_anchor_pixel is not None
                and (bank.subject_anchor_mask is not None)
                and (int(stored_pixels.shape[2]) > 0)
            ):
                try:
                    _latest = stored_pixels[:, :, -1].float()
                    _anchor = bank.subject_anchor_pixel.float()
                    _m = bank.subject_anchor_mask.to(dtype=torch.bool)
                    if _anchor.shape == _latest.shape and _m.any():
                        _mae = float((_latest - _anchor).abs()[..., _m].mean().item()) / 2.0
                        _thr = float(metadata.get("wbench_subject_anchor_lost_mae", 0.1) or 0.1)
                        bank.subject_anchor_paste = _mae >= _thr
                except Exception as _e:
                    bank.subject_anchor_paste = True
                    print(f"[subject_anchor_adaptive] fail-open: {_e}", flush=True)
            for local_idx, frame_idx in enumerate(frame_indices):
                bank.pixels.append(stored_pixels[:, :, local_idx].contiguous())
                bank.frame_indices.append(int(frame_idx))
                bank.depths.append(None)
                bank.vigeo_pointmaps.append(geometry.pointmaps[local_idx].to(dtype=torch.float16).contiguous())
                _vm = geometry.valid_masks[local_idx].bool()
                if _dyn_masks is not None:
                    _vm = _vm & ~_dyn_masks[local_idx].to(device=_vm.device).view_as(_vm)
                _vm = self._exclude_subject_from_valid_mask(_vm, bank.subject_exclusion_mask)
                bank.vigeo_valid_masks.append(_vm.contiguous())
                bank.vigeo_predicted_poses.append(geometry.predicted_poses[local_idx].float().contiguous())
                bank.vigeo_intrinsics.append(geometry.intrinsics[local_idx].float().contiguous())
            bank.vigeo_kv_caches = geometry.kv_caches
            bank.vigeo_generated_chunks += 1
            return
        frame_indices = list(range(int(target_start) * stride, int(target_start) * stride + frame_count))
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        keep = [i for (i, frame_idx) in enumerate(frame_indices) if 0 <= int(frame_idx) < cam_frames]
        if not keep:
            return
        pixels = pixels[:, :, keep].contiguous()
        frame_indices = [frame_indices[i] for i in keep]
        depth_by_local = (
            {}
            if self._uses_vigeo_prefix_last_frame()
            else self._infer_validation_bank_depths(pixels=pixels, metadata=metadata, frame_indices=frame_indices)
        )
        for local_idx, frame_idx in enumerate(frame_indices):
            bank.pixels.append(pixels[:, :, local_idx].detach().contiguous())
            bank.frame_indices.append(int(frame_idx))
            bank.depths.append(depth_by_local.get(local_idx) if depth_by_local is not None else None)

    def _record_validation_vigeo_causal_prefix(
        self, *, bank: _RolloutSpatialBank, decoded_pixels: torch.Tensor | None, target_start: int
    ) -> None:
        """Keep the latest RGB only for causal VAE encoding, not geometry retrieval."""
        if decoded_pixels is None or int(decoded_pixels.shape[2]) <= 0:
            raise RuntimeError("ViGeo causal prefix update requires decoded target pixels")
        target_pixel_start = int(target_start) * int(self.cfg.sample.temporal_stride) + int(bank.vigeo_pixel_offset)
        bank.causal_prefix_frame_index = target_pixel_start + int(decoded_pixels.shape[2]) - 1
        bank.causal_prefix_pixel = decoded_pixels[:, :, -1].detach().to(device="cpu", dtype=torch.float16).contiguous()

    def _build_validation_rollout_bank_spatial_context(
        self,
        *,
        bank: _RolloutSpatialBank,
        metadata: dict[str, Any],
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        if not bank.pixels:
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        if cam_c2w.dim() == 3:
            cam_c2w = cam_c2w.unsqueeze(0)
        cam_c2w = cam_c2w.to(device=self.dist.device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        stride = int(self.cfg.sample.temporal_stride)
        target_pixel_start = int(target_start) * stride
        if self._uses_vigeo_prefix_last_frame():
            target_pixel_start += int(bank.vigeo_pixel_offset)
        target_pixel_count = (
            int(K) * stride if self._uses_vigeo_prefix_last_frame() else 1 + max(0, int(K) - 1) * stride
        )
        target_pixel_indices = list(range(target_pixel_start, target_pixel_start + target_pixel_count))
        cam_frames = int(cam_c2w.shape[1])
        if not target_pixel_indices or target_pixel_indices[-1] >= cam_frames:
            return None
        candidate_indices = [
            local_idx
            for (local_idx, frame_idx) in enumerate(bank.frame_indices)
            if 0 <= int(frame_idx) < int(target_pixel_start)
        ]
        if not candidate_indices:
            return None
        if self._uses_vigeo_prefix_last_frame():
            return self._build_validation_vigeo_bank_spatial_context(
                bank=bank,
                candidate_indices=candidate_indices,
                metadata=metadata,
                cam_c2w=cam_c2w,
                intrinsic=intrinsic,
                target_pixel_indices=target_pixel_indices,
                target_start=target_start,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )
        selected = self._select_validation_rollout_bank_sources(
            bank=bank,
            candidate_indices=candidate_indices,
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
        )
        num_context = max(1, int(cfg.num_context_frames))
        if bool(getattr(cfg, "require_full_context", True)) and len(selected) < num_context:
            return None
        if not selected:
            return None
        source_video = torch.stack(bank.pixels, dim=2).to(device=self.dist.device, dtype=self.dtype).contiguous()
        (pixel_height, pixel_width) = (int(source_video.shape[-2]), int(source_video.shape[-1]))
        depth_by_source = {
            int(local_idx): bank.depths[local_idx] for local_idx in selected if bank.depths[local_idx] is not None
        }
        warp_result = forward_warp_indexed_pixel_sources_to_pixel_targets(
            source_pixels=source_video,
            source_pixel_indices=[int(i) for i in selected],
            source_camera_pixel_indices=[int(i) for i in bank.frame_indices],
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            depth_by_source_index=depth_by_source,
            height=pixel_height,
            width=pixel_width,
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 0.001),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        (warped_pixels, coverage_pixels) = warp_result
        spatial_latent = self._encode_spatial_context_video(warped_pixels, expected_latent_frames=int(K))
        mask_patch = self._build_spatial_mask_patch(coverage_pixels=coverage_pixels, spatial_latent=spatial_latent)
        if target_rope_t_indices is None:
            rope_t_indices: list[float] = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(x) for x in target_rope_t_indices.detach().cpu().tolist()]
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "mask_patch": mask_patch,
                "source_indices": [int(bank.frame_indices[i]) for i in selected],
                "target_indices": list(range(int(target_start), int(target_start) + int(K))),
                "source_pixel_indices": [int(bank.frame_indices[i]) for i in selected],
                "target_pixel_indices": target_pixel_indices,
                "rope_t_indices": rope_t_indices,
            },
            metadata=metadata,
        )

    def _build_validation_vigeo_bank_spatial_context(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_pixel_indices: list[int],
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        geometry_count = len(bank.vigeo_pointmaps)
        if geometry_count == 0 or geometry_count != len(bank.pixels):
            return None
        if not (
            len(bank.vigeo_valid_masks) == geometry_count
            and len(bank.vigeo_predicted_poses) == geometry_count
            and (len(bank.vigeo_intrinsics) == geometry_count)
        ):
            raise RuntimeError("ViGeo rollout bank geometry fields have inconsistent lengths")
        (pixel_height, pixel_width) = (int(bank.pixels[0].shape[-2]), int(bank.pixels[0].shape[-1]))
        source_global_c2w = self._validation_vigeo_bank_source_c2w(bank=bank, cam_c2w=cam_c2w)
        (target_c2w, target_intrinsic) = self._validation_vigeo_target_cameras(
            bank=bank,
            metadata=metadata,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            target_pixel_indices=target_pixel_indices,
            height=pixel_height,
            width=pixel_width,
        )
        retrieval_coverages: dict[int, float] = {}
        if int(bank.vigeo_generated_chunks) == 0:
            selected = [candidate_indices[-1]]
        else:
            (selected, retrieval_coverages) = self._select_validation_vigeo_bank_sources(
                bank=bank,
                candidate_indices=candidate_indices,
                source_global_c2w=source_global_c2w,
                target_c2w=target_c2w,
                target_intrinsic=target_intrinsic,
                height=pixel_height,
                width=pixel_width,
            )
            if not selected:
                return None
        selected_tensor = torch.tensor(selected, device=source_global_c2w.device, dtype=torch.long)
        source_c2w = source_global_c2w.index_select(1, selected_tensor)
        source_video = (
            torch.stack([bank.pixels[index] for index in selected], dim=2)
            .to(device=self.dist.device, dtype=self.dtype)
            .contiguous()
        )
        source_pointmaps = (
            torch.stack([bank.vigeo_pointmaps[index] for index in selected], dim=0)
            .unsqueeze(0)
            .to(device=self.dist.device, dtype=torch.float32)
        )
        source_pointmaps = source_pointmaps * float(bank.vigeo_scale)
        source_valid_masks = (
            torch.stack([bank.vigeo_valid_masks[index] for index in selected], dim=0)
            .unsqueeze(0)
            .to(device=self.dist.device, dtype=torch.bool)
        )
        warp_result = render_colored_pointmaps_to_camera_targets(
            source_pixels=source_video,
            source_pointmaps=source_pointmaps,
            source_valid_masks=source_valid_masks,
            source_c2w=source_c2w,
            target_c2w=target_c2w,
            target_intrinsic=target_intrinsic,
            height=pixel_height,
            width=pixel_width,
            depth_threshold=min(float(self.cfg.spatial_memory.retrieval_depth_threshold), 0.001),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        (warped_pixels, coverage_pixels) = warp_result
        (warped_pixels, coverage_pixels) = self._composite_wbench_subject_anchor(
            bank=bank, warped_pixels=warped_pixels, coverage_pixels=coverage_pixels
        )
        previous_pixel_index = int(target_pixel_indices[0]) - 1
        previous_bank_indices = [
            index for (index, frame_index) in enumerate(bank.frame_indices) if int(frame_index) == previous_pixel_index
        ]
        if previous_bank_indices:
            previous_bank_index = previous_bank_indices[-1]
            previous_pixels = bank.pixels[previous_bank_index].unsqueeze(2)
        elif bank.causal_prefix_pixel is not None and bank.causal_prefix_frame_index == previous_pixel_index:
            previous_pixels = bank.causal_prefix_pixel.unsqueeze(2)
        else:
            raise RuntimeError(
                f"ViGeo rollout bank is missing the causal VAE prefix frame {previous_pixel_index}; bank range={bank.frame_indices[:1]}..{bank.frame_indices[-1:]} cached_prefix={bank.causal_prefix_frame_index}"
            )
        (spatial_latent, vae_prefix_latent) = self._encode_spatial_continuation_video(
            previous_pixels=previous_pixels, target_pixels=warped_pixels, expected_target_latent_frames=int(K)
        )
        mask_patch = self._build_spatial_mask_patch(coverage_pixels=coverage_pixels, spatial_latent=spatial_latent)
        if target_rope_t_indices is None:
            rope_t_indices = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(value) for value in target_rope_t_indices.detach().cpu().tolist()]
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "mask_patch": mask_patch,
                "source_indices": [int(bank.frame_indices[index]) for index in selected],
                "target_indices": list(range(int(target_start), int(target_start) + int(K))),
                "scale_context_pixel_indices": self._vigeo_scale_context_pixel_indices(),
                "source_pixel_indices": [int(bank.frame_indices[index]) for index in selected],
                "source_bank_indices": [int(index) for index in selected],
                "vae_prefix_pixel_index": previous_pixel_index,
                "vae_prefix_latent": vae_prefix_latent,
                "target_pixel_indices": target_pixel_indices,
                "rope_t_indices": rope_t_indices,
                "vigeo_scale": float(bank.vigeo_scale),
                "vigeo_source_pose_mode": str(self.cfg.spatial_memory.vigeo_generated_source_pose_mode),
                "vigeo_pairwise_scales": bank.vigeo_pairwise_scales,
                "vigeo_retrieval_coverages": {
                    str(bank.frame_indices[index]): float(retrieval_coverages.get(index, 0.0)) for index in selected
                },
            },
            metadata=metadata,
        )

    def _vigeo_bank_global_c2w(self, bank: _RolloutSpatialBank) -> torch.Tensor:
        poses = self._homogeneous_camera_poses(torch.stack(bank.vigeo_predicted_poses, dim=0))
        anchor_inv = torch.linalg.inv(poses[:1])
        relative = torch.matmul(anchor_inv, poses)
        relative[:, :3, 3] *= float(bank.vigeo_scale)
        return relative.unsqueeze(0)

    def _validation_vigeo_bank_cplan_c2w(self, *, bank: _RolloutSpatialBank, cam_c2w: torch.Tensor) -> torch.Tensor:
        if cam_c2w.dim() == 3:
            cam_c2w = cam_c2w.unsqueeze(0)
        frame_indices = [int(frame_index) for frame_index in bank.frame_indices]
        if not frame_indices:
            raise ValueError("cannot build ViGeo source cameras for an empty bank")
        if min(frame_indices) < 0 or max(frame_indices) >= int(cam_c2w.shape[1]):
            raise ValueError(
                f"ViGeo bank frame indices exceed the planned camera path: range={min(frame_indices)}..{max(frame_indices)}, camera_frames={int(cam_c2w.shape[1])}"
            )
        frame_tensor = torch.tensor(frame_indices, device=cam_c2w.device, dtype=torch.long)
        planned_sources = cam_c2w.index_select(1, frame_tensor)
        return (
            torch.matmul(torch.linalg.inv(cam_c2w[:, :1]), planned_sources)
            .to(device=self.dist.device, dtype=torch.float32)
            .contiguous()
        )

    def _validation_vigeo_bank_source_c2w(self, *, bank: _RolloutSpatialBank, cam_c2w: torch.Tensor) -> torch.Tensor:
        planned = self._validation_vigeo_bank_cplan_c2w(bank=bank, cam_c2w=cam_c2w)
        mode = str(self.cfg.spatial_memory.vigeo_generated_source_pose_mode)
        if mode == "cplan" or int(bank.vigeo_generated_chunks) == 0:
            return planned
        predicted = self._vigeo_bank_global_c2w(bank).to(device=planned.device, dtype=planned.dtype)
        target_prefix_frames = self._vigeo_target_prefix_pixel_frames()
        anchor_frame = target_prefix_frames - 1
        anchor_candidates = [
            index for (index, frame_index) in enumerate(bank.frame_indices) if int(frame_index) == anchor_frame
        ]
        if not anchor_candidates:
            raise RuntimeError(f"ViGeo rollout bank is missing the final scale-prefix frame {anchor_frame}")
        anchor_index = anchor_candidates[-1]
        correction = planned[:, anchor_index] @ torch.linalg.inv(predicted[:, anchor_index])
        aligned_predicted = correction.unsqueeze(1) @ predicted
        generated_indices = [
            index for (index, frame_index) in enumerate(bank.frame_indices) if int(frame_index) >= target_prefix_frames
        ]
        if not generated_indices:
            return planned
        generated_tensor = torch.tensor(generated_indices, device=planned.device, dtype=torch.long)
        result = planned.clone()
        result[:, generated_tensor] = aligned_predicted[:, generated_tensor]
        return result.contiguous()

    def _validation_vigeo_target_cameras(
        self,
        *,
        bank: _RolloutSpatialBank,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_pixel_indices: list[int],
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_tensor = torch.tensor(target_pixel_indices, device=cam_c2w.device, dtype=torch.long)
        metadata_targets = cam_c2w.index_select(1, target_tensor)
        target_c2w = torch.matmul(torch.linalg.inv(cam_c2w[:, :1]), metadata_targets)
        if self._is_uncalibrated_intrinsic_source(metadata):
            prefix_intrinsic_index = min(self._vigeo_prefix_pixel_frames(), len(bank.vigeo_intrinsics)) - 1
            fitted = bank.vigeo_intrinsics[prefix_intrinsic_index].to(device=self.dist.device, dtype=torch.float32)
            target_K = fitted.view(1, 1, 3, 3).expand(int(cam_c2w.shape[0]), len(target_pixel_indices), -1, -1).clone()
        else:
            target_K = torch.stack(
                [
                    self._intrinsic_for_pixel_frame(
                        intrinsic, frame_idx, batch=int(cam_c2w.shape[0]), height=height, width=width
                    )
                    for frame_idx in target_pixel_indices
                ],
                dim=1,
            )
        return (target_c2w.contiguous(), target_K.contiguous())

    def _select_validation_vigeo_bank_sources(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        source_global_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsic: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[list[int], dict[int, float]]:
        num_context = max(1, int(self.cfg.spatial_memory.num_context_frames))
        source_centers = source_global_c2w[0, :, :3, 3].detach().cpu()
        target_centers = target_c2w[0, :, :3, 3].detach().cpu()
        source_forwards = F.normalize(source_global_c2w[0, :, :3, 2].detach().float().cpu(), dim=-1)
        target_forwards = F.normalize(target_c2w[0, :, :3, 2].detach().float().cpu(), dim=-1)
        distances = {
            int(index): float(
                torch.linalg.vector_norm(source_centers[index].view(1, 3) - target_centers, dim=-1).min().item()
            )
            for index in candidate_indices
        }
        angular_distances = {
            int(index): float(
                torch.acos(torch.matmul(target_forwards, source_forwards[index]).clamp(-1.0, 1.0)).min().item()
            )
            for index in candidate_indices
        }
        rotation_weight = max(0.0, float(getattr(self.cfg.spatial_memory, "retrieval_rotation_weight", 0.0)))
        pose_scores = {
            int(index): distances[int(index)] + rotation_weight * angular_distances[int(index)]
            for index in candidate_indices
        }
        nearest = sorted(candidate_indices, key=lambda index: (pose_scores[int(index)], -int(index)))
        prefilter_count = min(len(nearest), max(num_context * 8, num_context))
        nearest = nearest[:prefilter_count]
        coverages = {
            int(index): self._vigeo_source_target_coverage(
                pointmap=bank.vigeo_pointmaps[index],
                valid_mask=bank.vigeo_valid_masks[index],
                source_c2w=source_global_c2w[:, index],
                target_c2w=target_c2w,
                target_intrinsic=target_intrinsic,
                scale=float(bank.vigeo_scale),
                height=height,
                width=width,
            )
            for index in nearest
        }
        visible = [index for index in nearest if coverages[int(index)] >= 0.002]
        if str(getattr(self.cfg.spatial_memory, "retrieval_sort", "nearest")) == "coverage":
            visible = sorted(visible, key=lambda index: -coverages[int(index)])
        selected = visible[:num_context]
        if not selected and nearest:
            selected = [max(nearest, key=lambda index: (coverages[int(index)], -distances[int(index)]))]
        return ([int(index) for index in selected], coverages)

    def _vigeo_source_target_coverage(
        self,
        *,
        pointmap: torch.Tensor,
        valid_mask: torch.Tensor,
        source_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsic: torch.Tensor,
        scale: float,
        height: int,
        width: int,
    ) -> float:
        sample_stride = max(4, int(self.cfg.spatial_memory.downsample))
        points = pointmap[::sample_stride, ::sample_stride].to(device=self.dist.device, dtype=torch.float32) * float(
            scale
        )
        valid = valid_mask[::sample_stride, ::sample_stride].to(device=self.dist.device, dtype=torch.bool)
        valid = valid & torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
        if not bool(valid.any()):
            return 0.0
        points = points[valid]
        ones = torch.ones((int(points.shape[0]), 1), device=points.device, dtype=points.dtype)
        world = torch.matmul(
            source_c2w[0].to(device=points.device, dtype=points.dtype), torch.cat([points, ones], dim=-1).unsqueeze(-1)
        )[:, :3, 0]
        target_count = int(target_c2w.shape[1])
        query_indices = sorted({0, target_count // 2, target_count - 1})
        query_weights = {0: 0.6, target_count // 2: 0.2, target_count - 1: 0.2}
        cell_size = 16
        grid_width = max(1, (int(width) + cell_size - 1) // cell_size)
        grid_height = max(1, (int(height) + cell_size - 1) // cell_size)
        total_cells = float(grid_width * grid_height)
        weighted_coverage = 0.0
        weight_total = 0.0
        world_h = torch.cat([world, torch.ones_like(world[:, :1])], dim=-1).unsqueeze(-1)
        for query_index in query_indices:
            camera = torch.matmul(torch.linalg.inv(target_c2w[0, query_index]).to(world_h), world_h)[:, :3, 0]
            z = camera[:, 2]
            projected = torch.matmul(target_intrinsic[0, query_index].to(camera), camera.unsqueeze(-1))[:, :, 0]
            x = torch.round(projected[:, 0] / torch.clamp(projected[:, 2], min=1e-06)).long()
            y = torch.round(projected[:, 1] / torch.clamp(projected[:, 2], min=1e-06)).long()
            visible = (
                torch.isfinite(camera).all(dim=-1)
                & (z > 0)
                & (x >= 0)
                & (x < int(width))
                & (y >= 0)
                & (y < int(height))
            )
            if bool(visible.any()):
                cells = y[visible] // cell_size * grid_width + x[visible] // cell_size
                coverage = float(torch.unique(cells).numel()) / total_cells
            else:
                coverage = 0.0
            weight = float(query_weights[query_index])
            weighted_coverage += weight * coverage
            weight_total += weight
        return weighted_coverage / max(weight_total, 1e-08)

    @staticmethod
    def _homogeneous_camera_poses(poses: torch.Tensor) -> torch.Tensor:
        poses = poses.detach().float()
        if poses.dim() != 3:
            raise ValueError(f"camera poses must be [F,3/4,4], got {tuple(poses.shape)}")
        if tuple(poses.shape[-2:]) == (4, 4):
            return poses
        if tuple(poses.shape[-2:]) != (3, 4):
            raise ValueError(f"unsupported camera pose shape {tuple(poses.shape)}")
        bottom = torch.zeros((int(poses.shape[0]), 1, 4), device=poses.device, dtype=poses.dtype)
        bottom[:, 0, 3] = 1.0
        return torch.cat([poses, bottom], dim=1)

    def _select_validation_rollout_bank_sources(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        target_pixel_indices: list[int],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> list[int]:
        cfg = self.cfg.spatial_memory
        num_context = max(1, int(cfg.num_context_frames))
        (pixel_height, pixel_width) = (int(bank.pixels[0].shape[-2]), int(bank.pixels[0].shape[-1]))
        selected: list[int] = []
        try:
            cache = Sparse3DCache(downsample=max(1, int(cfg.downsample)))
            for local_idx in candidate_indices:
                frame_idx = int(bank.frame_indices[local_idx])
                depth = bank.depths[local_idx]
                if depth is None:
                    depth = torch.full(
                        (cam_c2w.shape[0], 1, pixel_height, pixel_width),
                        float(cfg.constant_depth),
                        device=self.dist.device,
                        dtype=torch.float32,
                    )
                else:
                    depth = depth.to(device=self.dist.device, dtype=torch.float32)
                cache.add(
                    depth=depth,
                    w2c=torch.linalg.inv(cam_c2w[:, frame_idx]),
                    intrinsic=self._intrinsic_for_pixel_frame(
                        intrinsic, frame_idx, batch=int(cam_c2w.shape[0]), height=pixel_height, width=pixel_width
                    ),
                    latent_index=int(local_idx),
                    frame_id=frame_idx,
                )
            retrieval_views = max(1, int(cfg.retrieval_views))
            if retrieval_views == 1:
                target_view_indices = [target_pixel_indices[-1]]
            else:
                offsets = torch.linspace(0, len(target_pixel_indices) - 1, retrieval_views)
                target_view_indices = [target_pixel_indices[int(round(float(x.item())))] for x in offsets]
            target_w2c = torch.stack([torch.linalg.inv(cam_c2w[:, idx]) for idx in target_view_indices], dim=1)
            target_K = torch.stack(
                [
                    self._intrinsic_for_pixel_frame(
                        intrinsic, idx, batch=int(cam_c2w.shape[0]), height=pixel_height, width=pixel_width
                    )
                    for idx in target_view_indices
                ],
                dim=1,
            )
            retrieved = cache.retrieve(
                target_w2c=target_w2c,
                target_intrinsic=target_K,
                target_hw=(pixel_height, pixel_width),
                num_latents=num_context,
                max_coverage=bool(cfg.retrieval_max_coverage),
                depth_threshold=float(cfg.retrieval_depth_threshold),
            )
            selected = [int(local_idx) for (local_idx, _frame_id) in retrieved]
        except Exception as exc:
            rank0_print(
                self.dist, "[ValidationSpatialBank]", f"coverage source selection failed: {type(exc).__name__}: {exc}"
            )
            selected = []
        if len(selected) < num_context:
            seen = set(selected)
            for local_idx in reversed(candidate_indices):
                if local_idx in seen:
                    continue
                selected.append(int(local_idx))
                seen.add(int(local_idx))
                if len(selected) >= num_context:
                    break
        return selected[:num_context]

    def _infer_validation_bank_depths(
        self, *, pixels: torch.Tensor, metadata: dict[str, Any], frame_indices: list[int]
    ) -> dict[int, torch.Tensor]:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "vigeo":
            return {}
        if backend == "constant":
            return {}
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return {}
            selected = _select_depth_by_frame_index(depth, frame_indices)
            return {
                local_idx: selected[int(frame_idx)]
                for (local_idx, frame_idx) in enumerate(frame_indices)
                if int(frame_idx) in selected
            }
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return {}
        local_frame_indices = list(range(int(pixels.shape[2])))
        cam_subset = self._select_camera_frames(cam_c2w, frame_indices)
        intrinsic_subset = self._select_intrinsic_frames(intrinsic, frame_indices)
        return (
            self._build_spatial_depths_for_pixel_frames(
                video_pixels=pixels,
                metadata={},
                cam_c2w=cam_subset,
                intrinsic=intrinsic_subset,
                frame_indices=local_frame_indices,
            )
            or {}
        )

    def _select_video_pixel_frames(self, video_pixels: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        video = self._video_pixels_to_bcfhw(video_pixels).to(device=self.dist.device, dtype=self.dtype)
        idx = torch.tensor([int(i) for i in frame_indices], device=video.device, dtype=torch.long)
        return video.index_select(2, idx).contiguous()

    def _slice_video_pixel_frames(self, video_pixels: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Return a [B,F,C,H,W] pixel slice accepted by the VAE path."""
        video = self._video_pixels_to_bcfhw(video_pixels)
        start = max(0, int(start))
        end = min(int(end), int(video.shape[2]))
        if end <= start:
            raise ValueError(f"empty video pixel slice [{start}:{end}]")
        return video[:, :, start:end].permute(0, 2, 1, 3, 4).contiguous()

    def _video_pixels_to_bcfhw(self, video_pixels: torch.Tensor) -> torch.Tensor:
        video = video_pixels.detach()
        if video.dim() == 5:
            if video.shape[1] == 3:
                return video.contiguous()
            if video.shape[2] == 3:
                return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.dim() == 4:
            if video.shape[0] == 3:
                return video.unsqueeze(0).contiguous()
            if video.shape[1] == 3:
                return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        raise ValueError(
            f"expected video tensor [B,C,F,H,W], [B,F,C,H,W], [C,F,H,W], or [F,C,H,W], got {tuple(video.shape)}"
        )

    def _pixel_hw_from_video(self, video_pixels: torch.Tensor) -> tuple[int, int]:
        video = self._video_pixels_to_bcfhw(video_pixels)
        return (int(video.shape[-2]), int(video.shape[-1]))

    def _select_camera_frames(self, cam_c2w: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        cam = cam_c2w.to(device=self.dist.device, dtype=torch.float32)
        idx = torch.tensor([int(i) for i in frame_indices], device=cam.device, dtype=torch.long)
        if cam.dim() == 4:
            idx = idx.clamp(0, cam.shape[1] - 1)
            return cam.index_select(1, idx).contiguous()
        if cam.dim() == 3:
            idx = idx.clamp(0, cam.shape[0] - 1)
            return cam.index_select(0, idx).unsqueeze(0).contiguous()
        raise ValueError(f"unexpected cam_c2w shape {tuple(cam_c2w.shape)}")

    def _select_intrinsic_frames(self, intrinsic: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        K = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = torch.tensor([int(i) for i in frame_indices], device=K.device, dtype=torch.long).clamp(
                0, K.shape[1] - 1
            )
            return K.index_select(1, idx).contiguous()
        return K

    def _intrinsic_for_pixel_frame(
        self,
        intrinsic: torch.Tensor,
        frame_idx: int,
        *,
        batch: int,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        K = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = max(0, min(int(frame_idx), int(K.shape[1]) - 1))
            K = K[:, idx]
        elif K.dim() == 2:
            K = K.unsqueeze(0)
        pixel_height = int(height if height is not None else self.cfg.sample.height)
        pixel_width = int(width if width is not None else self.cfg.sample.width)
        K = pixel_intrinsics(K, height=pixel_height, width=pixel_width)
        if K.shape[0] == 1 and int(batch) > 1:
            K = K.expand(int(batch), -1, -1)
        return K.contiguous()

    def _extend_validation_camera_static(self, metadata: dict[str, Any], *, min_frames: int) -> None:
        if not _as_bool(metadata.get("has_camera", False)):
            return
        target_frames = max(0, int(min_frames))
        cam = metadata.get("cam_c2w")
        cam_frames = self._camera_frame_count(cam)
        if cam_frames <= 0 or cam_frames >= target_frames:
            return
        extension_mode = str(getattr(self.cfg.validation, "camera_extension", "static") or "static")
        for key in ("cam_c2w", "cam_c2w_raw"):
            value = metadata.get(key)
            if extension_mode == "forward":
                padded = self._pad_camera_tensor_forward(value, target_frames=target_frames)
            else:
                padded = self._pad_frame_tensor_static(value, target_frames=target_frames)
            if padded is not None:
                metadata[key] = padded
        for key in ("intrinsic", "intrinsic_raw"):
            value = metadata.get(key)
            padded = self._pad_frame_tensor_static(value, target_frames=target_frames, expected_frames=cam_frames)
            if padded is not None:
                metadata[key] = padded
        metadata["validation_camera_extension"] = True
        metadata["validation_camera_extension_mode"] = extension_mode
        metadata["validation_camera_static_extension"] = extension_mode == "static"
        metadata["validation_camera_original_frames"] = int(cam_frames)
        metadata["validation_camera_extended_frames"] = int(target_frames)

    def _camera_frame_count(self, cam_c2w: Any) -> int:
        if not torch.is_tensor(cam_c2w):
            return 0
        if cam_c2w.dim() == 4:
            return int(cam_c2w.shape[1])
        if cam_c2w.dim() == 3:
            return int(cam_c2w.shape[0])
        return 0

    def _pad_frame_tensor_static(
        self, tensor: Any, *, target_frames: int, expected_frames: int | None = None
    ) -> torch.Tensor | None:
        if not torch.is_tensor(tensor):
            return None
        if tensor.dim() == 4:
            frame_dim = 1
        elif tensor.dim() == 3 and (expected_frames is None or int(tensor.shape[0]) == int(expected_frames)):
            frame_dim = 0
        else:
            return None
        frames = int(tensor.shape[frame_dim])
        if frames <= 0 or frames >= int(target_frames):
            return tensor
        pad_count = int(target_frames) - frames
        last = tensor.narrow(frame_dim, frames - 1, 1).expand(
            *[pad_count if dim == frame_dim else int(size) for (dim, size) in enumerate(tensor.shape)]
        )
        return torch.cat([tensor, last.to(device=tensor.device, dtype=tensor.dtype)], dim=frame_dim).contiguous()

    def _pad_camera_tensor_forward(self, tensor: Any, *, target_frames: int) -> torch.Tensor | None:
        if not torch.is_tensor(tensor):
            return None
        if tensor.dim() == 4:
            batched = True
            cam = tensor
        elif tensor.dim() == 3 and tensor.shape[-2:] == (4, 4):
            batched = False
            cam = tensor.unsqueeze(0)
        else:
            return None
        if cam.shape[-2:] != (4, 4):
            return None
        frames = int(cam.shape[1])
        if frames <= 0 or frames >= int(target_frames):
            return tensor
        pad_count = int(target_frames) - frames
        step_per_latent = float(getattr(self.cfg.validation, "camera_forward_step_per_latent", 0.05))
        step_per_frame = step_per_latent / max(1, int(self.cfg.sample.temporal_stride))
        last = cam[:, frames - 1]
        padded = last[:, None].expand(-1, pad_count, -1, -1).clone()
        rel = torch.zeros(int(cam.shape[0]), pad_count, 3, device=cam.device, dtype=cam.dtype)
        rel[:, :, 2] = (torch.arange(1, pad_count + 1, device=cam.device, dtype=cam.dtype) * float(step_per_frame))[
            None, :
        ]
        world_delta = torch.einsum("bij,bpj->bpi", last[:, :3, :3], rel)
        padded[:, :, :3, 3] = last[:, None, :3, 3] + world_delta
        out = torch.cat([cam, padded.to(device=cam.device, dtype=cam.dtype)], dim=1).contiguous()
        return out if batched else out.squeeze(0)

    def _validation_prompt_label(self, *, mode_cfg: ValidationModeConfig, round_idx: int) -> str | None:
        schedule = list(getattr(mode_cfg, "prompt_schedule", []) or [])
        if not schedule:
            return None
        return str(schedule[min(int(round_idx), len(schedule) - 1)]).strip()

    def _video_pixel_frame_count(self, video_pixels: torch.Tensor) -> int:
        if video_pixels.dim() == 5:
            if video_pixels.shape[2] == 3:
                return int(video_pixels.shape[1])
            if video_pixels.shape[1] == 3:
                return int(video_pixels.shape[2])
            raise ValueError(f"expected video tensor [B,F,C,H,W] or [B,C,F,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.dim() != 4:
            raise ValueError(f"expected video tensor [F,C,H,W] or [C,F,H,W], got {tuple(video_pixels.shape)}")
        return int(video_pixels.shape[1] if video_pixels.shape[0] == 3 else video_pixels.shape[0])

    def _encode_video(
        self, video_pixels: torch.Tensor, *, needed_latents: int | None = None, metadata: dict[str, Any] | None = None
    ) -> torch.Tensor:
        video_pixels = video_pixels.to(device=self.dist.device, dtype=self.dtype)
        if video_pixels.dim() == 5:
            if video_pixels.shape[0] != 1:
                raise ValueError("AlayaWorld v1.1 inference supports batch size 1")
            video_pixels = video_pixels[0]
        if video_pixels.dim() != 4:
            raise ValueError(f"expected video tensor [F,C,H,W] or [B,F,C,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.shape[0] != 3:
            video_pixels = video_pixels.permute(1, 0, 2, 3).contiguous()
        needed_latents = needed_latents or self._required_latents_for_max_K()
        needed_pixels = (needed_latents - 1) * self.cfg.sample.temporal_stride + 1
        use_pixels = min(video_pixels.shape[1], needed_pixels)
        use_pixels = 1 + self.cfg.sample.temporal_stride * ((use_pixels - 1) // self.cfg.sample.temporal_stride)
        video_pixels = video_pixels[:, :use_pixels]
        _t0 = time.time()
        latent = self._encode_video_via_latent_cache(video_pixels, metadata)
        if latent is None:
            with torch.no_grad():
                latent = self.components.vae_encoder.encode(
                    video_pixels.unsqueeze(0), chunk_size=self.cfg.runtime.vae_chunk_size, verbose=False
                )
        latent = latent.to(device=self.dist.device, dtype=self.dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.dist.device)
        self._perf_vae_s += time.time() - _t0
        return latent

    def _encode_video_via_latent_cache(
        self, video_pixels: torch.Tensor, metadata: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """Whole-clip latent cache path: the first FRESH_HEAD_LAT latents are encoded fresh and the tail

        is sliced from the cache, which is bit-identical to a fresh full-window encode (a causal VAE
        carries about 16 latents of memory). A miss, a misaligned start or a too-short window returns None.
        """
        cache_dir = getattr(self.cfg.runtime, "vae_latent_cache_dir", None)
        if not cache_dir or metadata is None:
            return None
        from alaya.data.vae_latent_cache import FRESH_HEAD_LAT, aligned_m_start, load_meta, load_slice

        T_pix = int(video_pixels.shape[1])
        n_lat = (T_pix - 1) // self.cfg.sample.temporal_stride + 1
        if n_lat <= FRESH_HEAD_LAT:
            return None
        source = str(metadata.get("source", ""))
        video_id = str(metadata.get("video_id", ""))
        frame_start = int(metadata.get("frame_start", -1))
        if not source or not video_id or frame_start < 0:
            return None
        (H, W) = (int(self.cfg.sample.height), int(self.cfg.sample.width))
        fps = float(self.cfg.sample.fps)
        meta = load_meta(cache_dir, source, video_id, H, W, fps)
        if meta is None:
            self._vae_cache_misses += 1
            return None
        m = aligned_m_start(frame_start, float(meta["ratio"]))
        if m is None:
            self._vae_cache_misses += 1
            return None
        tail = load_slice(cache_dir, source, video_id, H, W, fps, m + FRESH_HEAD_LAT, n_lat - FRESH_HEAD_LAT)
        if tail is None:
            self._vae_cache_misses += 1
            return None
        head_pixels = video_pixels[:, : (FRESH_HEAD_LAT - 1) * self.cfg.sample.temporal_stride + 1]
        with torch.no_grad():
            head = self.components.vae_encoder.encode(
                head_pixels.unsqueeze(0), chunk_size=self.cfg.runtime.vae_chunk_size, verbose=False
            )
        self._vae_cache_hits += 1
        tail = tail.unsqueeze(0).to(device=head.device, dtype=head.dtype)
        return torch.cat([head, tail], dim=2)

    def _encode_caption(self, caption: str, *, sync: bool) -> torch.Tensor:
        if sync and dist.is_initialized():
            captions = [caption]
            dist.broadcast_object_list(captions, src=0)
            caption = captions[0]
        cache_cap = int(getattr(self.cfg.runtime, "text_embed_cache_entries", 0) or 0)
        cache_dir = getattr(self.cfg.runtime, "text_embed_cache_dir", None)
        t0 = time.time()
        if cache_cap > 0:
            cached = self._text_cache.get(caption)
            if cached is not None:
                self._text_cache.move_to_end(caption)
                self._text_cache_hits += 1
                self._perf_text_s += time.time() - t0
                return cached.to(device=self.dist.device, dtype=self.dtype)
        if cache_dir:
            from alaya.data.text_embed_cache import disk_get

            hit = disk_get(cache_dir, caption)
            if hit is not None:
                self._text_cache_hits += 1
                if cache_cap > 0:
                    self._text_cache[caption] = hit
                    while len(self._text_cache) > cache_cap:
                        self._text_cache.popitem(last=False)
                self._perf_text_s += time.time() - t0
                return hit.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            output = self.components.encode_text(self.components.text_encoder, [caption])
        context = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
        context = context.to(device=self.dist.device, dtype=self.dtype)
        self._text_cache_misses += 1
        if cache_dir:
            from alaya.data.text_embed_cache import disk_put

            disk_put(cache_dir, caption, context)
        if cache_cap > 0:
            self._text_cache[caption] = context.detach().to("cpu")
            while len(self._text_cache) > cache_cap:
                self._text_cache.popitem(last=False)
        self._perf_text_s += time.time() - t0
        return context

    def _adaptive_sigma_shift_m(self, latent_frames: int | float) -> float:
        frame_lo = int(getattr(self.cfg.sigma_shift, "adaptive_shift_frame_lo", 8))
        frame_hi = int(getattr(self.cfg.sigma_shift, "adaptive_shift_frame_hi", 121))
        m_lo = float(getattr(self.cfg.sigma_shift, "adaptive_shift_m_lo", 5.0))
        m_hi = float(getattr(self.cfg.sigma_shift, "adaptive_shift_m_hi", 30.0))
        if frame_hi <= frame_lo:
            return m_lo
        t = (float(latent_frames) - float(frame_lo)) / float(frame_hi - frame_lo)
        t = max(0.0, min(1.0, t))
        return m_lo + t * (m_hi - m_lo)

    def _build_control_kwargs(
        self,
        *,
        metadata: dict[str, Any],
        control_modes: list[str],
        target_t_indices: torch.Tensor,
        condition_t_indices: torch.Tensor | None = None,
        history_t_indices: torch.Tensor | None = None,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
        if "action" in control_modes:
            cam_c2w = metadata.get("cam_c2w")
            if cam_c2w is not None and _as_bool(metadata.get("has_camera", False)):
                kwargs["action_vectors"] = build_action_vectors(
                    cam_c2w=cam_c2w,
                    target_latent_indices=target_t_indices,
                    action_scale=self.cfg.control.action_scale,
                    temporal_stride=self.cfg.sample.temporal_stride,
                    device=self.dist.device,
                    dtype=dtype,
                )
                if condition_t_indices is not None and condition_t_indices.numel() > 0:
                    kwargs["action_condition_vectors"] = build_action_vectors(
                        cam_c2w=cam_c2w,
                        target_latent_indices=condition_t_indices,
                        action_scale=self.cfg.control.action_scale,
                        temporal_stride=self.cfg.sample.temporal_stride,
                        device=self.dist.device,
                        dtype=dtype,
                    )
                if (
                    bool(getattr(self.cfg.control, "action_history_memory", False))
                    and history_t_indices is not None
                    and (history_t_indices.numel() > 0)
                ):
                    kwargs["action_history_vectors"] = build_action_vectors(
                        cam_c2w=cam_c2w,
                        target_latent_indices=history_t_indices,
                        action_scale=self.cfg.control.action_scale,
                        temporal_stride=self.cfg.sample.temporal_stride,
                        device=self.dist.device,
                        dtype=dtype,
                    )
        return kwargs

    def _build_explicit_pixel_control_kwargs(
        self,
        *,
        metadata: dict[str, Any],
        control_modes: list[str],
        target_pixel_start: int,
        target_latent_frames: int,
        nearby_pixel: int,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
        if "action" not in control_modes:
            return kwargs
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is None or not _as_bool(metadata.get("has_camera", False)):
            return kwargs
        target_pixels = [
            int(target_pixel_start) + index * int(self.cfg.sample.temporal_stride)
            for index in range(int(target_latent_frames))
        ]
        kwargs["action_vectors"] = build_action_vectors_from_pixel_indices(
            cam_c2w=cam_c2w,
            pixel_indices=target_pixels,
            previous_pixel_index=nearby_pixel,
            action_scale=self.cfg.control.action_scale,
            device=self.dist.device,
            dtype=dtype,
        )
        kwargs["action_condition_vectors"] = build_action_vectors_from_pixel_indices(
            cam_c2w=cam_c2w,
            pixel_indices=[nearby_pixel],
            previous_pixel_index=max(0, nearby_pixel - int(self.cfg.sample.temporal_stride)),
            action_scale=self.cfg.control.action_scale,
            device=self.dist.device,
            dtype=dtype,
        )
        return kwargs

    def _build_spatial_context(
        self,
        *,
        latent_full: torch.Tensor,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        target_start: int,
        K: int,
        cond_end: int,
        target_rope_t_indices: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        if not cfg.enabled:
            return None
        if float(cfg.dropout) > 0.0 and random.random() < float(cfg.dropout):
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        context_mode = str(getattr(cfg, "context_mode", "retrieval"))
        if context_mode == "target_prefix_pixels":
            return self._build_target_prefix_pixel_spatial_context(
                video_pixels=video_pixels,
                metadata=metadata,
                cam_c2w=cam_c2w,
                intrinsic=intrinsic,
                target_start=target_start,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )
        if context_mode == "vigeo_prefix_last_frame":
            return None
        if context_mode != "retrieval":
            raise ValueError(f"unsupported spatial_memory.context_mode={context_mode!r}")
        allowed: list[int] = []
        nearby_allowed: set[int] = set()
        stride = max(1, int(cfg.cache_stride))
        sink_count = int(self.cfg.layout.sink_latent_frames)
        recent_cutoff = int(target_start) - int(cfg.skip_recent_latents)
        if bool(cfg.include_sink) and sink_count > 0 and (not bool(self.cfg.layout.sink_remote)):
            allowed.extend(range(0, sink_count, stride))
        N = int(self.cfg.layout.history_latent_frames)
        hist_end = int(target_start) - int(cond_end)
        hist_start = max(sink_count, hist_end - N)
        if N > 0 and hist_end > hist_start:
            allowed.extend(range(hist_start, hist_end, stride))
        if bool(cfg.include_nearby) and cond_end > 0:
            cond_start = max(sink_count, int(target_start) - int(cond_end))
            nearby_allowed = set(range(cond_start, int(target_start), stride))
            allowed.extend(nearby_allowed)
        allowed = [
            idx
            for idx in sorted(set(allowed))
            if 0 <= idx < int(target_start) and (idx < recent_cutoff or idx in nearby_allowed)
        ]
        if not allowed:
            return None
        (pixel_height, pixel_width) = self._pixel_hw_from_video(video_pixels)
        depth_by_latent_index = self._build_spatial_depths(
            video_pixels=video_pixels, metadata=metadata, cam_c2w=cam_c2w, intrinsic=intrinsic, allowed=allowed
        )
        retrieval_views = max(1, int(cfg.retrieval_views))
        if retrieval_views == 1:
            target_indices = [int(target_start + K - 1)]
        else:
            offsets = torch.linspace(0, max(0, int(K) - 1), retrieval_views, device=self.dist.device)
            target_indices = [int(target_start + int(round(float(x.item())))) for x in offsets]
        context = build_retrieved_latent_context(
            latent_full=latent_full,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            allowed_latent_indices=allowed,
            target_latent_indices=target_indices,
            height=pixel_height,
            width=pixel_width,
            temporal_stride=int(self.cfg.sample.temporal_stride),
            num_context_frames=int(cfg.num_context_frames),
            downsample=int(cfg.downsample),
            constant_depth=float(cfg.constant_depth),
            depth_by_latent_index=depth_by_latent_index,
            retrieval_max_coverage=bool(cfg.retrieval_max_coverage),
            retrieval_depth_threshold=float(cfg.retrieval_depth_threshold),
        )
        if context is None:
            return None
        spatial_latent = context.latent.to(device=self.dist.device, dtype=latent_full.dtype)
        if bool(cfg.use_warped_context):
            warped_pixels = forward_warp_video_to_targets(
                video_pixels=video_pixels,
                source_latent_indices=context.source_latent_indices,
                target_latent_indices=context.target_latent_indices,
                cam_c2w=cam_c2w.to(device=self.dist.device, dtype=torch.float32),
                intrinsic=intrinsic.to(device=self.dist.device, dtype=torch.float32),
                depth_by_latent_index=depth_by_latent_index,
                height=pixel_height,
                width=pixel_width,
                temporal_stride=int(self.cfg.sample.temporal_stride),
                constant_depth=float(cfg.constant_depth),
                depth_threshold=min(float(cfg.retrieval_depth_threshold), 0.001),
            )
            if warped_pixels is not None:
                spatial_latent = self._encode_spatial_context_pixels(warped_pixels)
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "source_indices": context.source_latent_indices,
                "target_indices": context.target_latent_indices,
            },
            metadata=metadata,
        )

    def _build_target_prefix_pixel_spatial_context(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        stride = int(self.cfg.sample.temporal_stride)
        (pixel_height, pixel_width) = self._pixel_hw_from_video(video_pixels)
        target_pixel_start = int(target_start) * stride
        target_pixel_count = 1 + max(0, int(K) - 1) * stride
        target_pixel_indices = list(range(target_pixel_start, target_pixel_start + target_pixel_count))
        video_frames = self._video_pixel_frame_count(video_pixels)
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        max_frames = min(video_frames, cam_frames)
        if not target_pixel_indices or target_pixel_indices[-1] >= max_frames:
            return None
        history_pixels = max(1, int(cfg.num_context_frames))
        source_end = target_pixel_start
        source_floor = 0
        if not bool(cfg.include_sink):
            source_floor = max(0, int(self.cfg.layout.sink_latent_frames) * stride)
        source_start = max(source_floor, source_end - history_pixels)
        source_pixel_indices = list(range(source_start, source_end))
        if not source_pixel_indices:
            return None
        if bool(getattr(cfg, "require_full_context", True)) and len(source_pixel_indices) < history_pixels:
            return None
        depth_by_frame_index = self._build_spatial_depths_for_pixel_frames(
            video_pixels=video_pixels,
            metadata=metadata,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            frame_indices=source_pixel_indices,
        )
        warp_result = forward_warp_pixel_sources_to_pixel_targets(
            video_pixels=video_pixels,
            source_pixel_indices=source_pixel_indices,
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w.to(device=self.dist.device, dtype=torch.float32),
            intrinsic=intrinsic.to(device=self.dist.device, dtype=torch.float32),
            depth_by_frame_index=depth_by_frame_index,
            height=pixel_height,
            width=pixel_width,
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 0.001),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        (warped_pixels, coverage_pixels) = warp_result
        spatial_latent = self._encode_spatial_context_video(warped_pixels, expected_latent_frames=int(K))
        mask_patch = self._build_spatial_mask_patch(coverage_pixels=coverage_pixels, spatial_latent=spatial_latent)
        if target_rope_t_indices is None:
            rope_t_indices: list[float] = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(x) for x in target_rope_t_indices.detach().cpu().tolist()]
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "mask_patch": mask_patch,
                "source_indices": source_pixel_indices,
                "target_indices": list(range(int(target_start), int(target_start) + int(K))),
                "source_pixel_indices": source_pixel_indices,
                "target_pixel_indices": target_pixel_indices,
                "rope_t_indices": rope_t_indices,
            },
            metadata=metadata,
        )

    def _get_vigeo_geometry(self) -> ViGeoGeometryEstimator:
        if self.vigeo_geometry is None:
            cfg = self.cfg.spatial_memory
            device = self.dist.device if str(cfg.vigeo_device) == "auto" else torch.device(str(cfg.vigeo_device))
            self.vigeo_geometry = ViGeoGeometryEstimator(
                repo_path=cfg.vigeo_repo_path,
                checkpoint=str(cfg.vigeo_checkpoint),
                device=device,
                num_tokens=int(cfg.vigeo_num_tokens),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading ViGeo checkpoint={cfg.vigeo_checkpoint} repo={cfg.vigeo_repo_path}",
            )
        return self.vigeo_geometry

    def _encode_spatial_context_video(
        self, warped_pixels: torch.Tensor, *, expected_latent_frames: int
    ) -> torch.Tensor:
        warped_pixels = warped_pixels.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            latent = self.components.vae_encoder.encode(
                warped_pixels, chunk_size=self.cfg.runtime.vae_chunk_size, verbose=False
            )
        latent = latent.to(device=self.dist.device, dtype=self.dtype).contiguous()
        if int(latent.shape[2]) != int(expected_latent_frames):
            raise RuntimeError(
                f"spatial rendered VAE produced {latent.shape[2]} latents, expected {expected_latent_frames}; warped_pixels={tuple(warped_pixels.shape)}"
            )
        return latent

    def _encode_spatial_continuation_video(
        self, *, previous_pixels: torch.Tensor, target_pixels: torch.Tensor, expected_target_latent_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self._video_pixels_to_bcfhw(previous_pixels).to(device=self.dist.device, dtype=self.dtype)
        target = self._video_pixels_to_bcfhw(target_pixels).to(device=self.dist.device, dtype=self.dtype)
        if int(previous.shape[2]) != 1:
            raise ValueError(f"spatial continuation requires one previous pixel frame, got {previous.shape[2]}")
        expected_target_pixels = int(expected_target_latent_frames) * int(self.cfg.sample.temporal_stride)
        if int(target.shape[2]) != expected_target_pixels:
            raise ValueError(
                f"spatial continuation target pixel length mismatch: got {target.shape[2]}, expected {expected_target_pixels}"
            )
        if previous.shape[0] != target.shape[0] or previous.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"spatial continuation prefix/target shape mismatch: previous={tuple(previous.shape)} target={tuple(target.shape)}"
            )
        continuation = torch.cat([previous, target], dim=2)
        latent = self._encode_spatial_context_video(
            continuation, expected_latent_frames=int(expected_target_latent_frames) + 1
        )
        return (latent[:, :, 1:].contiguous(), latent[:, :, :1].contiguous())

    def _maybe_force_spatial_all_invalid(
        self, context: dict[str, Any] | None, *, metadata: dict[str, Any] | None = None, force: bool = False
    ) -> dict[str, Any] | None:
        if context is None:
            return context
        force_invalid = bool(force) or bool(getattr(self.cfg.spatial_memory, "force_all_invalid", False))
        if not force_invalid:
            return context
        latent = context.get("latent")
        if latent is None:
            return context
        forced = dict(context)
        forced["latent"] = torch.zeros_like(latent)
        mask_patch = forced.get("mask_patch")
        if mask_patch is None:
            (B, C, frames, height, width) = latent.shape
            patchifier = VideoLatentPatchifier(patch_size=1)
            (pt, ph, pw) = patchifier.patch_size
            if frames % pt != 0 or height % ph != 0 or width % pw != 0:
                raise ValueError(
                    f"spatial latent shape {(frames, height, width)} is not divisible by patch size {(pt, ph, pw)}"
                )
            tokens = frames // pt * (height // ph) * (width // pw)
            channel_patch = int(C) * int(pt) * int(ph) * int(pw)
            mask_patch = torch.zeros(int(B), int(tokens), int(channel_patch), device=latent.device, dtype=latent.dtype)
        else:
            mask_patch = torch.zeros_like(mask_patch)
        forced["mask_patch"] = mask_patch
        return forced

    def _is_uncalibrated_intrinsic_source(self, metadata: dict[str, Any]) -> bool:
        src = str(metadata.get("source") or "").lower()
        if src == "wbench_navi":
            return bool(getattr(self.cfg.spatial_memory, "wbench_fitted_intrinsic", True))
        if src == "custom_i2v":
            return not bool(metadata.get("has_real_intrinsic"))
        return False

    def _build_spatial_mask_patch(self, *, coverage_pixels: torch.Tensor, spatial_latent: torch.Tensor) -> torch.Tensor:
        (B, C, frames, height, width) = spatial_latent.shape
        if coverage_pixels.dim() != 5 or int(coverage_pixels.shape[0]) != B or int(coverage_pixels.shape[1]) != 1:
            raise ValueError(f"coverage_pixels must be [B,1,T,H,W] with B={B}, got {tuple(coverage_pixels.shape)}")
        patchifier = VideoLatentPatchifier(patch_size=1)
        (pt, ph, pw) = patchifier.patch_size
        if frames % pt != 0 or height % ph != 0 or width % pw != 0:
            raise ValueError(
                f"spatial latent shape {(frames, height, width)} is not divisible by patch size {(pt, ph, pw)}"
            )
        mask_grid = (
            F.adaptive_avg_pool3d(
                coverage_pixels.to(device=self.dist.device, dtype=torch.float32),
                output_size=(frames // pt, height // ph, width // pw),
            )
            .gt_(0.5)
            .to(dtype=torch.float32)
        )
        mask_flat = mask_grid[:, 0].reshape(B, -1, 1)
        channel_patch = int(C) * int(pt) * int(ph) * int(pw)
        return mask_flat.expand(B, -1, channel_patch).to(dtype=spatial_latent.dtype).contiguous()

    def _encode_spatial_context_pixels(self, warped_pixels: torch.Tensor) -> torch.Tensor:
        warped_pixels = warped_pixels.to(device=self.dist.device, dtype=self.dtype)
        latents = []
        with torch.no_grad():
            for frame_idx in range(int(warped_pixels.shape[2])):
                latent = self.components.vae_encoder.encode(
                    warped_pixels[:, :, frame_idx : frame_idx + 1],
                    chunk_size=self.cfg.runtime.vae_chunk_size,
                    verbose=False,
                )
                latents.append(latent[:, :, :1].to(device=self.dist.device, dtype=self.dtype))
        return torch.cat(latents, dim=2).contiguous()

    def _build_spatial_depths(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        allowed: list[int],
    ) -> dict[int, torch.Tensor] | None:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "constant":
            return None
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return None
            return _select_depth_by_latent_index(depth, allowed, int(self.cfg.sample.temporal_stride))
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")
        if self.da3_depth is None:
            device = self.dist.device if str(cfg.da3_device) == "auto" else torch.device(str(cfg.da3_device))
            self.da3_depth = DA3DepthEstimator(
                repo_path=cfg.da3_repo_path,
                model_name=str(cfg.da3_model_name),
                cache_dir=cfg.da3_cache_dir,
                device=device,
                process_res=int(cfg.da3_process_res),
                process_res_method=str(cfg.da3_process_res_method),
                align_to_input_scale=bool(cfg.da3_align_to_input_scale),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading DA3 depth backend model={cfg.da3_model_name} repo={cfg.da3_repo_path}",
            )
        (pixel_height, pixel_width) = self._pixel_hw_from_video(video_pixels)
        return self.da3_depth.infer_latent_depths(
            video_pixels=video_pixels,
            latent_indices=allowed,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            height=pixel_height,
            width=pixel_width,
            temporal_stride=int(self.cfg.sample.temporal_stride),
        )

    def _build_spatial_depths_for_pixel_frames(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        frame_indices: list[int],
    ) -> dict[int, torch.Tensor] | None:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "constant":
            return None
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return None
            return _select_depth_by_frame_index(depth, frame_indices)
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")
        if self.da3_depth is None:
            device = self.dist.device if str(cfg.da3_device) == "auto" else torch.device(str(cfg.da3_device))
            self.da3_depth = DA3DepthEstimator(
                repo_path=cfg.da3_repo_path,
                model_name=str(cfg.da3_model_name),
                cache_dir=cfg.da3_cache_dir,
                device=device,
                process_res=int(cfg.da3_process_res),
                process_res_method=str(cfg.da3_process_res_method),
                align_to_input_scale=bool(cfg.da3_align_to_input_scale),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading DA3 depth backend model={cfg.da3_model_name} repo={cfg.da3_repo_path}",
            )
        (pixel_height, pixel_width) = self._pixel_hw_from_video(video_pixels)
        return self.da3_depth.infer_frame_depths(
            video_pixels=video_pixels,
            frame_indices=frame_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            height=pixel_height,
            width=pixel_width,
        )

    def _required_latents_for_max_K(self) -> int:
        max_gap = int(self.cfg.layout.max_gap_sec * self.cfg.sample.fps / self.cfg.sample.temporal_stride)
        max_k = max(self._active_output_latent_frames())
        if self._uses_fixed_no_history_condition_window():
            return self.cfg.layout.sink_latent_frames + max_gap + max_k + 1
        return (
            self.cfg.layout.sink_latent_frames
            + max_gap
            + self.cfg.layout.history_latent_frames
            + self._max_explicit_condition_latents()
            + max_k
        )

    def _active_output_latent_frames(self) -> list[int]:
        pairs = [
            (int(k), float(p))
            for (k, p) in zip(self.cfg.layout.output.latent_frames, self.cfg.layout.output.probs)
            if float(p) > 0.0
        ]
        if not pairs:
            raise ValueError("layout.output.probs must contain at least one positive value")
        return [k for (k, _) in pairs]

    def _max_explicit_condition_latents(self) -> int:
        if self.cfg.layout.history_latent_frames > 0:
            return 0
        if self.cfg.layout.condition.i2v_prob <= 0 and self.cfg.layout.condition.v2v_prob <= 0:
            return 0
        max_k = max(self.cfg.layout.output.latent_frames)
        max_cond = 1 if self.cfg.layout.condition.i2v_prob > 0 else 0
        if self.cfg.layout.condition.v2v_prob > 0:
            max_cond = max(max_cond, max(1, int(max_k * self.cfg.layout.condition.v2v_ratio_max)))
        return max_cond

    def _uses_fixed_no_history_condition_window(self) -> bool:
        return self.cfg.layout.history_latent_frames == 0 and (
            self.cfg.layout.condition.i2v_prob > 0 or self.cfg.layout.condition.v2v_prob > 0
        )

    def _uses_vigeo_prefix_last_frame(self) -> bool:
        return (
            bool(self.cfg.spatial_memory.enabled)
            and str(self.cfg.spatial_memory.context_mode) == "vigeo_prefix_last_frame"
            and (str(self.cfg.spatial_memory.depth_backend) == "vigeo")
        )

    def _vigeo_prefix_pixel_frames(self) -> int:
        configured = self.cfg.spatial_memory.vigeo_prefix_frames
        return int(self.cfg.spatial_memory.num_context_frames if configured is None else configured)

    def _vigeo_target_prefix_pixel_frames(self, *, history_latent_frames: int | None = None) -> int:
        history_latents = (
            int(self.cfg.layout.history_latent_frames) if history_latent_frames is None else int(history_latent_frames)
        )
        if history_latents <= 0:
            return self._vigeo_prefix_pixel_frames()
        return 1 + (history_latents - 1) * int(self.cfg.sample.temporal_stride)

    def _vigeo_scale_context_pixel_indices(self, *, history_latent_frames: int | None = None) -> list[int]:
        context_frames = self._vigeo_prefix_pixel_frames()
        target_prefix_frames = self._vigeo_target_prefix_pixel_frames(history_latent_frames=history_latent_frames)
        start = target_prefix_frames - context_frames
        if start < 0:
            raise ValueError(
                f"ViGeo scale context does not fit before the target: context_frames={context_frames}, target_prefix_frames={target_prefix_frames}"
            )
        return list(range(start, target_prefix_frames))

    def _vigeo_motion_pixel_frames(self) -> int:
        return int(self.cfg.sample.temporal_stride) + 1

    def _local_sink_t_offset(self, history_latent_frames: int) -> int:
        return 0

    def _local_memory_t_offset(self, history_latent_frames: int, condition_latent_frames: int) -> int:
        return 1

    def _local_nearby_t_offset(
        self, history_latent_frames: int, condition_latent_frames: int, gap_steps: int = 0
    ) -> int:
        history_latents = max(0, int(history_latent_frames))
        condition_latents = max(0, int(condition_latent_frames))
        if history_latents > 0:
            return 1 + max(0, history_latents - condition_latents)
        return 1

    def _local_target_t_indices(
        self, frames: int, *, history_latent_frames: int, condition_latent_frames: int, gap_steps: int = 0
    ) -> torch.Tensor:
        history_latents = max(0, int(history_latent_frames))
        condition_latents = max(0, int(condition_latent_frames))
        if history_latents > 0:
            start = 1 + history_latents
        else:
            start = 1 + condition_latents
        return torch.arange(start, start + int(frames), device=self.dist.device, dtype=torch.float32)

    def _indices_grid(self, batch: int, frames: int, height: int, width: int, *, t_offset: int) -> torch.Tensor:
        patchifier = VideoLatentPatchifier(patch_size=1)
        shape = VideoLatentShape(batch=batch, channels=1, frames=frames, height=height, width=width)
        coords = patchifier.get_patch_grid_bounds(shape, device=self.dist.device).clone().to(torch.float32)
        coords[:, 0, :, :] += t_offset
        return coords

    def _indices_grid_for_t_indices(
        self, batch: int, t_indices: list[int | float] | torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        frames = int(t_indices.numel()) if torch.is_tensor(t_indices) else len(t_indices)
        coords = self._indices_grid(batch, frames, height, width, t_offset=0)
        if torch.is_tensor(t_indices):
            t = t_indices.to(device=self.dist.device, dtype=coords.dtype)
        else:
            t = torch.tensor(t_indices, device=self.dist.device, dtype=coords.dtype)
        per_token = t.view(frames, 1, 1).expand(frames, height, width).reshape(-1)
        bounds = torch.stack([per_token, per_token + 1], dim=-1)
        coords[:, 0, :, :] = bounds.unsqueeze(0)
        return coords
