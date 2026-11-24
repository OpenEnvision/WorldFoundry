"""Observer adapter wrapping LiveWorld pipeline calls.

This module keeps LiveWorld-specific logic isolated from the event-centric flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pipeline_unified_backbone import (
    UnifiedBackbonePipeline,
)


@dataclass
class ObserverIterationInput:
    """Inputs for a single observer iteration."""

    # Preceding frames used for P1/P9 conditioning (required).
    preceding_frames: np.ndarray
    prompt: str
    num_frames: int
    infer_steps: int
    target_scene_proj: torch.Tensor
    guidance_scale: float | None = None
    cpu_offload: bool = False
    preceding_scene_proj: torch.Tensor | None = None
    reference_frames: np.ndarray | None = None
    instance_reference_frames: np.ndarray | None = None  # [R_inst, H, W, 3]
    target_fg_proj: torch.Tensor | None = None
    preceding_fg_proj: torch.Tensor | None = None
    seed: int | None = None
    sp_context_scale: float = 1.0


@dataclass
class ObserverIterationOutput:
    """Outputs from a single observer iteration."""

    frames: torch.Tensor
    # Optional static point cloud update (points, colors).
    static_update: tuple[np.ndarray, np.ndarray] | None


class ObserverAdapter:
    """Thin wrapper around UnifiedBackbonePipeline for event-centric usage."""

    def __init__(self, pipeline: UnifiedBackbonePipeline) -> None:
        self.pipeline = pipeline

    def _resolve_few_step_schedule(self):
        """Return denoising step list if current observer config is few-step."""
        schedule = getattr(self.pipeline.config, "denoising_step_list", None)
        if schedule is None:
            return None

        try:
            schedule_list = list(schedule)
        except TypeError:
            return None

        return schedule_list if len(schedule_list) > 0 else None

    def run_iteration(self, inputs: ObserverIterationInput) -> ObserverIterationOutput:
        """Run one iteration of LiveWorld generation with provided projections."""
        if inputs.preceding_frames is None or len(inputs.preceding_frames) == 0:
            raise ValueError("preceding_frames is required for observer conditioning")
        # UnifiedBackbonePipeline API requires a first_frame parameter; for a P-only observer
        # we pass the first preceding frame as a deterministic anchor.
        anchor_frame = Image.fromarray(inputs.preceding_frames[0])
        denoising_step_list = self._resolve_few_step_schedule()

        # The LiveWorld pipeline handles the diffusion step; point cloud updates are
        # intentionally left to the caller to avoid hidden side effects.
        if denoising_step_list is not None:
            # Few-step CFG policy:
            # - guidance_scale is None  -> disable CFG branch
            # - guidance_scale is value -> enable CFG branch
            use_cfg_few_step = inputs.guidance_scale is not None
            generated = self.pipeline.run_single_iteration_few_step(
                first_frame=anchor_frame,
                target_scene_proj=inputs.target_scene_proj,
                prompt=inputs.prompt,
                num_frames=inputs.num_frames,
                infer_steps=inputs.infer_steps,
                guidance_scale=inputs.guidance_scale,
                use_cfg=use_cfg_few_step,
                cpu_offload=inputs.cpu_offload,
                preceding_frames=inputs.preceding_frames,
                preceding_scene_proj=inputs.preceding_scene_proj,
                reference_frames=inputs.reference_frames,
                instance_reference_frames=inputs.instance_reference_frames,
                target_fg_proj=inputs.target_fg_proj,
                preceding_fg_proj=inputs.preceding_fg_proj,
                seed=inputs.seed,
                sp_context_scale=inputs.sp_context_scale,
                denoising_step_list=denoising_step_list,
            )
        else:
            generated = self.pipeline.run_single_iteration(
                first_frame=anchor_frame,
                target_scene_proj=inputs.target_scene_proj,
                prompt=inputs.prompt,
                num_frames=inputs.num_frames,
                infer_steps=inputs.infer_steps,
                guidance_scale=inputs.guidance_scale,
                cpu_offload=inputs.cpu_offload,
                preceding_frames=inputs.preceding_frames,
                preceding_scene_proj=inputs.preceding_scene_proj,
                reference_frames=inputs.reference_frames,
                instance_reference_frames=inputs.instance_reference_frames,
                target_fg_proj=inputs.target_fg_proj,
                preceding_fg_proj=inputs.preceding_fg_proj,
                seed=inputs.seed,
                sp_context_scale=inputs.sp_context_scale,
            )

        # NOTE: Static point cloud update is not performed here. The caller must
        # invoke the 3D updater explicitly to avoid implicit fallback logic.
        return ObserverIterationOutput(frames=generated, static_update=None)
