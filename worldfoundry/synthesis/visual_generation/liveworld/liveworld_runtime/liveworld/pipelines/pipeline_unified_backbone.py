"""
LiveWorld Pipeline: Multi-iteration video generation with scene projection conditioning.

This pipeline handles:
1. Single iteration denoising with [T, P, R] frame order
2. State Adapter conditioning with scene_proj + fg_proj (32 channels total)
3. CFG and cpu_offload for memory optimization
4. Point cloud utilities: voxel downsampling, depth to pointcloud, projection
5. Reference frame selection based on 3D IoU
6. Multi-iteration video generation

Usage:
    from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pipeline_unified_backbone import UnifiedBackbonePipeline
    pipeline = UnifiedBackbonePipeline(config, device, generator, text_encoder, vae)
    video = pipeline.run_single_iteration(...)
    # or for multi-iteration:
    video, iterations = pipeline.run_multi_iteration(...)
"""

from __future__ import annotations

import json
import os
import time as _time
import traceback
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc_compat import (
    FlowUniPCMultistepScheduler as FlowUniPCMultistepScheduler,
)
from worldfoundry.base_models.perception_core.general_perception.qwen3_vl_entity import (
    Qwen3VLEntityExtractor,
)
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.pointcloud_updater import (
    ReconstructionResult,
    create_pointcloud_updater,
)
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.utils import (
    FlowMatchScheduler,
    set_seed,
)

from ..geometry_utils import (
    BackboneInferenceOptions,
    _compute_projection_density_max_pixels,
    _get_visible_points_and_coverage_gpu,
    _limit_points_by_density,
    _merge_pointcloud_incremental,
    _safe_frame_index,
    _voxel_downsample,
    compute_3d_iou,
    compute_3d_iou_batched,
    # Moved from this file to break circular dep with pointcloud_updater
    compute_iteration_plan,
    generate_blue_noise_tile,
    get_visible_points_for_frame,
    render_projection,
    save_point_cloud_ply,
    scale_intrinsics_batch,
)
from ..geometry_utils import (
    scale_intrinsics as scale_intrinsics_from_size,
)
from ..utils import save_video_h264

# isort: split
from worldfoundry.base_models.perception_core.segment.sam3.video_segmenter import (
    Sam3VideoSegmenter,
)

# =============================================================================
# Iteration Planning Utilities
# =============================================================================


# =============================================================================
# Pipeline Config and State
# =============================================================================


@dataclass
class BackboneIterationConfig:
    """Configuration for a single iteration of LiveWorld generation."""

    num_frames: int = 33  # Number of frames to generate (4N+1)
    infer_steps: int = 50  # Denoising steps
    guidance_scale: float = 5.0
    use_cfg: bool = True
    cpu_offload: bool = False


@dataclass
class BackboneIterationState:
    """State maintained across multiple iterations."""

    all_generated_frames: list[np.ndarray] = field(default_factory=list)
    current_first_frame: Image.Image | None = None


@dataclass
class BackboneOutputPaths:
    """Output paths for iterative inference artifacts."""

    output_dir: str
    video_dir: str
    pointcloud_dir: str


@dataclass
class IterationState:
    """State shared across iterative inference steps."""

    all_generated_frames: list[np.ndarray] = field(default_factory=list)
    current_first_frame: Image.Image | None = None
    all_scene_proj_frames: list[np.ndarray] = field(default_factory=list)
    all_used_scene_proj_frames: list[np.ndarray] = field(default_factory=list)
    points_world: np.ndarray | None = None
    colors: np.ndarray | None = None
    poses_c2w: np.ndarray | None = None

    intrinsics: np.ndarray | None = None
    intrinsics_size: tuple[int, int] | None = None
    accumulated_anchor_frames: np.ndarray | None = None
    accumulated_anchor_poses: np.ndarray | None = None
    accumulated_anchor_intrinsics: np.ndarray | None = None
    accumulated_anchor_depths: np.ndarray | None = None
    accumulated_anchor_global_indices: list[int] = field(default_factory=list)
    frame_visible_points: dict[int, np.ndarray] = field(default_factory=dict)
    target_scene_proj: torch.Tensor | None = None
    projection_density_max_pixels: int | None = None
    projection_density_blue_noise: np.ndarray | None = None
    projection_density_rng: np.random.Generator | None = None
    pointcloud_updater: Any | None = None  # PointCloudUpdater instance
    initial_dynamic_mask: np.ndarray | None = None  # First-frame dynamic mask from SAM3 (dynamic + sky)
    initial_fg_only_mask: np.ndarray | None = None  # First-frame foreground mask (dynamic only, no sky)


@dataclass
class IterationResult:
    """Result of a single iteration."""

    generated_video: torch.Tensor
    frames_to_store: list[np.ndarray]
    scene_proj_frames: list[np.ndarray] | None = None
    updated_points_world: np.ndarray | None = None
    updated_colors: np.ndarray | None = None
    iter_idx: int = 0
    output_start: int = 0
    output_end: int = 0
    model_frames: int = 0


@dataclass
class BackboneInferenceResult:
    """Final result of iterative inference."""

    state: IterationState
    iteration_plan: list[tuple[int, int, int]]
    generated_video: torch.Tensor
    output_size: tuple[int, int]
    start_frame: int
    num_frames: int
    first_frame: Image.Image | None = None
    initial_points_world: np.ndarray | None = None
    initial_colors: np.ndarray | None = None


# =============================================================================
# Main Pipeline Class
# =============================================================================


class UnifiedBackbonePipeline:
    """
    LiveWorld video generation pipeline with State Adapter.

    Supports:
    - Single iteration denoising with [T, P, R] frame order
    - Scene projection + foreground projection (32 channels: scene 16 + fg 16)
    - Preceding frames (P) and reference frames (R)
    - I2V mode with CLIP conditioning
    - T2V mode without CLIP
    - CFG and CPU offload
    """

    def __init__(
        self,
        config,
        device: torch.device,
        generator,
        vae,
        text_encoder,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize LiveWorld pipeline.

        Args:
            config: Model config (OmegaConf)
            device: torch device
            generator: State Adapter generator model
            vae: VAE wrapper
            dtype: Model dtype
        """
        self.config = config
        self.device = device
        self.generator = generator
        self.vae = vae
        self.text_encoder = text_encoder
        self.dtype = dtype

        # Config values
        self.vae_stride = getattr(config, "vae_stride", (4, 8, 8))
        self.timestep_shift = getattr(config, "timestep_shift", 3.0)
        self.num_train_timesteps = getattr(config, "num_train_timestep", 1000)
        self.negative_prompt = getattr(config, "negative_prompt", "")

        # State Adapter config: always use 32 channels (scene 16 + fg 16)
        self.use_fg_proj = getattr(config, "use_fg_proj", False)

        # Get latent shape
        self.h, self.w = config.image_or_video_shape[-2:]
        self.h_pixel = self.h * self.vae_stride[1]
        self.w_pixel = self.w * self.vae_stride[2]
        self.latent_channels = config.image_or_video_shape[2]

    def _pixel_to_latent_frames(self, pixel_frames: int) -> int:
        """Convert pixel frame count to latent frame count."""
        return (pixel_frames - 1) // self.vae_stride[0] + 1

    def _encode_frames_to_latent(self, frames: np.ndarray) -> torch.Tensor:
        """
        Encode pixel frames to latent space.

        Args:
            frames: [N, H, W, 3] uint8 numpy array

        Returns:
            latent: [1, N_latent, C, h, w] tensor
        """
        # Convert to tensor: [N, H, W, 3] -> [N, 3, H, W] -> float [-1, 1]
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        frames_tensor = frames_tensor.to(device=self.device, dtype=self.dtype)
        # VAE expects [B, C, F, H, W]
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, N, H, W]
        # Ensure VAE is on the correct device (may be on CPU after offload)
        vae_device = next(self.vae.model.parameters()).device
        if vae_device != self.device:
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)
        latent = self.vae.encode_to_latent(frames_tensor)  # [1, N_latent, C, h, w]
        return latent

    def _prepare_sp_context(
        self,
        target_scene_proj: torch.Tensor,
        target_fg_proj: torch.Tensor | None,
        preceding_scene_proj: torch.Tensor | None,
        preceding_fg_proj: torch.Tensor | None,
        num_t: int,
        num_p: int,
    ) -> list[torch.Tensor]:
        """
        Prepare State Adapter context by concatenating scene and fg projections.

        Training code flow (task_fm_wan_liveworld.py):
        1. target_scene_proj [B, T, C, H, W] + target_fg_proj [B, T, C, H, W]
           -> concat along C (dim=2) -> [B, T, 2C, H, W]
        2. Same for preceding: [B, P, 2C, H, W]
        3. Concat target and preceding along frame dim -> [B, T+P, 2C, H, W]
        4. Permute to [B, 2C, T+P, H, W] and return as list

        Final sp_context shape is [32, T+P, h, w] where 32 = scene(16) + fg(16).

        Args:
            target_scene_proj: [C, T, h, w] where C=16
            target_fg_proj: [C, T, h, w] or None
            preceding_scene_proj: [C, P, h, w] or None
            preceding_fg_proj: [C, P, h, w] or None
            num_t: Number of target latent frames
            num_p: Number of preceding latent frames

        Returns:
            sp_context: List containing the concatenated projection tensor [32, T+P, h, w]
        """

        def _adjust_frames(proj: torch.Tensor, target_frames: int) -> torch.Tensor:
            """Adjust projection to target number of frames."""
            if proj.shape[1] > target_frames:
                return proj[:, :target_frames, :, :]
            elif proj.shape[1] < target_frames:
                pad = target_frames - proj.shape[1]
                return torch.cat([proj, proj[:, -1:, :, :].repeat(1, pad, 1, 1)], dim=1)
            return proj

        # Adjust target scene projection to num_t frames
        target_scene = _adjust_frames(target_scene_proj, num_t)

        # Prepare target fg (use zeros if not provided)
        if self.use_fg_proj and target_fg_proj is not None:
            target_fg = _adjust_frames(target_fg_proj, num_t)
        else:
            target_fg = torch.zeros_like(target_scene)

        # Concat target scene + fg along channel: [C, T, h, w] -> [2C, T, h, w]
        target_combined = torch.cat([target_scene, target_fg], dim=0)

        # Handle preceding
        if num_p > 0:
            if preceding_scene_proj is not None:
                prec_scene = _adjust_frames(preceding_scene_proj, num_p)
            else:
                # No preceding_scene_proj provided, use zeros
                prec_scene = torch.zeros(
                    target_scene.shape[0],
                    num_p,
                    target_scene.shape[2],
                    target_scene.shape[3],
                    device=target_scene.device,
                    dtype=target_scene.dtype,
                )

            # Prepare preceding fg (use zeros if not provided)
            if self.use_fg_proj and preceding_fg_proj is not None:
                prec_fg = _adjust_frames(preceding_fg_proj, num_p)
            else:
                prec_fg = torch.zeros_like(prec_scene)

            # Concat preceding scene + fg along channel: [C, P, h, w] -> [2C, P, h, w]
            prec_combined = torch.cat([prec_scene, prec_fg], dim=0)

            # Concat target and preceding along frame dim: [2C, T+P, h, w]
            sp_proj = torch.cat([target_combined, prec_combined], dim=1)
        else:
            sp_proj = target_combined

        return [sp_proj]

    def _resolve_few_step_timesteps(
        self,
        denoising_step_list: list[float] | None = None,
    ) -> torch.Tensor:
        """
        Resolve few-step timesteps for CausVid-style bidirectional inference.

        If denoising_step_list is not provided, it falls back to config.denoising_step_list.
        """
        if denoising_step_list is None:
            denoising_step_list = getattr(self.config, "denoising_step_list", None)
        if denoising_step_list is None:
            raise ValueError("Few-step inference requires denoising_step_list.")

        if isinstance(denoising_step_list, torch.Tensor):
            timesteps = denoising_step_list.detach().clone().float().to(self.device)
        else:
            timesteps = torch.tensor(list(denoising_step_list), dtype=torch.float32, device=self.device)

        if timesteps.numel() < 1:
            raise ValueError("denoising_step_list must contain at least one timestep.")

        return timesteps

    @torch.inference_mode()
    def run_single_iteration(
        self,
        first_frame: Image.Image,
        target_scene_proj: torch.Tensor,
        prompt: str,
        num_frames: int,
        infer_steps: int = 50,
        guidance_scale: float | None = None,
        use_cfg: bool = True,
        cpu_offload: bool = False,
        preceding_frames: np.ndarray | None = None,
        preceding_scene_proj: torch.Tensor | None = None,
        reference_frames: np.ndarray | None = None,
        target_fg_proj: torch.Tensor | None = None,
        preceding_fg_proj: torch.Tensor | None = None,
        seed: int | None = None,
        sp_context_scale: float = 1.0,
        preceding_noise_timestep: int = 0,
        instance_reference_frames: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Run a single iteration of video generation.

        Frame order: [T, P, R_scene, R_inst] where:
        - T = target frames (noisy, to be denoised)
        - P = preceding frames (clean, from previous iteration)
        - R_scene = scene reference frames (clean, retrieved by 3D IoU)
        - R_inst = instance reference frames (clean, from event foreground)

        Args:
            first_frame: First frame PIL image (I2V condition)
            target_scene_proj: Target scene projection [C, T, h, w] (16 channels)
            prompt: Text prompt
            num_frames: Number of frames to generate (4N+1 format)
            infer_steps: Number of denoising steps
            guidance_scale: CFG scale (default: config value)
            use_cfg: Whether to use CFG
            cpu_offload: Whether to offload models to CPU
            preceding_frames: Preceding frames [P, H, W, 3] uint8 numpy
            preceding_scene_proj: Preceding scene projection [C, P, h, w] (16 channels)
            reference_frames: Reference frames [R_scene, H, W, 3] uint8 numpy (scene references)
            target_fg_proj: Target foreground projection [C, T, h, w] (16 channels)
            preceding_fg_proj: Preceding foreground projection [C, P, h, w] (16 channels)
            seed: Random seed for noise generation
            sp_context_scale: Scale for State Adapter conditioning (default: 1.0)
            instance_reference_frames: Instance reference frames [R_inst, H, W, 3] uint8 numpy (foreground entity references)

        Returns:
            generated_video: [T_pixel, H, W, 3] uint8 tensor
        """
        if guidance_scale is None:
            guidance_scale = getattr(self.config, "guidance_scale", 5.0)

        # Calculate latent frames
        num_latent_frames = self._pixel_to_latent_frames(num_frames)
        num_t = num_latent_frames

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # ========== CPU offload: move VAE/CLIP/TextEncoder to GPU for encoding ==========
        if cpu_offload:
            _t0 = _time.time()
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)
            self.text_encoder.to(self.device)
            torch.cuda.empty_cache()
            print(f"    [offload] encoders → GPU  {_time.time() - _t0:.1f}s")

        # ========== Prepare preceding latents (P) ==========
        _t0 = _time.time()
        num_p = 0
        preceding_latent = None
        if preceding_frames is not None and len(preceding_frames) > 0:
            preceding_latent = self._encode_frames_to_latent(preceding_frames)
            num_p = preceding_latent.shape[1]

            # Add noise to preceding frames (P9 only, not P1)
            if preceding_noise_timestep > 0 and num_p > 1:
                _noise_sched = FlowMatchScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    num_inference_steps=self.num_train_timesteps,
                    shift=1.0,
                )
                noise_p = torch.randn_like(preceding_latent)
                t_p = torch.full(
                    [num_p],
                    preceding_noise_timestep,
                    device=self.device,
                    dtype=torch.float32,
                )
                preceding_latent = _noise_sched.add_noise(
                    preceding_latent.squeeze(0),
                    noise_p.squeeze(0),
                    t_p,
                ).unsqueeze(0)
                print(f"  Added noise to preceding frames: timestep={preceding_noise_timestep}")
        print(f"    [encode] preceding P={num_p}  {_time.time() - _t0:.1f}s")

        # ========== Prepare reference latents (R_scene + R_inst) ==========
        _t0 = _time.time()
        num_r_scene = 0
        reference_scene_latent = None
        if reference_frames is not None and len(reference_frames) > 0:
            ref_latents = []
            for i in range(len(reference_frames)):
                single = self._encode_frames_to_latent(reference_frames[i : i + 1])
                ref_latents.append(single)
            reference_scene_latent = torch.cat(ref_latents, dim=1)
            num_r_scene = reference_scene_latent.shape[1]

        num_r_inst = 0
        reference_inst_latent = None
        if instance_reference_frames is not None and len(instance_reference_frames) > 0:
            inst_latents = []
            for i in range(len(instance_reference_frames)):
                single = self._encode_frames_to_latent(instance_reference_frames[i : i + 1])
                inst_latents.append(single)
            reference_inst_latent = torch.cat(inst_latents, dim=1)
            num_r_inst = reference_inst_latent.shape[1]

        num_r = num_r_scene + num_r_inst
        reference_latent = None
        if num_r > 0:
            parts = []
            if reference_scene_latent is not None:
                parts.append(reference_scene_latent)
            if reference_inst_latent is not None:
                parts.append(reference_inst_latent)
            reference_latent = torch.cat(parts, dim=1)
        print(f"    [encode] refs R_scene={num_r_scene} R_inst={num_r_inst}  {_time.time() - _t0:.1f}s")

        # ========== Prepare text conditioning ==========
        _t0 = _time.time()
        text_conditional_dict = self.text_encoder(text_prompts=[prompt])
        context = text_conditional_dict["prompt_embeds"]
        context = [c.to(self.device) for c in context]

        uncon_context = None
        if use_cfg:
            unconditional_dict = self.text_encoder(text_prompts=[self.negative_prompt])
            uncon_context = unconditional_dict["prompt_embeds"]
            uncon_context = [c.to(self.device) for c in uncon_context]
        print(f"    [encode] text  {_time.time() - _t0:.1f}s")

        # ========== CPU offload: move encoders to CPU, generator to GPU ==========
        if cpu_offload:
            _t0 = _time.time()
            self.text_encoder.to("cpu")
            self.vae.model.to("cpu")
            self.vae.mean = self.vae.mean.to("cpu")
            self.vae.std = self.vae.std.to("cpu")
            torch.cuda.empty_cache()
            self.generator.to(self.device)
            print(f"    [offload] generator → GPU  {_time.time() - _t0:.1f}s")

        # ========== Prepare State Adapter context ==========
        sp_context = self._prepare_sp_context(
            target_scene_proj=target_scene_proj,
            target_fg_proj=target_fg_proj,
            preceding_scene_proj=preceding_scene_proj,
            preceding_fg_proj=preceding_fg_proj,
            num_t=num_t,
            num_p=num_p,
        )

        # ========== Initialize noise ==========
        noise = torch.randn(
            1, num_latent_frames, self.latent_channels, self.h, self.w, device=self.device, dtype=self.dtype
        )

        # ========== Setup scheduler ==========
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(infer_steps, device=self.device, shift=self.timestep_shift)

        latents = noise

        # ========== Denoising loop ==========
        denoising_pbar = tqdm(
            enumerate(sample_scheduler.timesteps),
            total=len(sample_scheduler.timesteps),
            desc=f"Denoising (T={num_t}, P={num_p}, R_scene={num_r_scene}, R_inst={num_r_inst})",
        )

        for step_idx, t in denoising_pbar:
            # Build per-frame timestep [B, T+P+R]
            timestep_t = t * torch.ones([1, num_t], device=self.device, dtype=torch.float32)
            timestep_parts = [timestep_t]
            if num_p > 0:
                timestep_parts.append(torch.zeros([1, num_p], device=self.device, dtype=torch.float32))
            if num_r > 0:
                timestep_parts.append(torch.zeros([1, num_r], device=self.device, dtype=torch.float32))
            timestep = torch.cat(timestep_parts, dim=1)

            # Build input latents [B, T+P+R, C, h, w]
            latent_parts = [latents]
            if preceding_latent is not None and num_p > 0:
                latent_parts.append(preceding_latent)
            if reference_latent is not None and num_r > 0:
                latent_parts.append(reference_latent)
            combined_latents = torch.cat(latent_parts, dim=1)

            generator_kwargs = {
                "noisy_image_or_video": combined_latents,
                "timestep": timestep,
                "context": context,
                "clip_fea": None,
                "y": None,
                "sp_context": sp_context,
                "sp_context_scale": float(sp_context_scale),
                "num_t": num_t,
                "num_p": num_p,
                "num_r": num_r,
                "num_r_scene": num_r_scene if num_r_inst > 0 else None,
                "num_r_inst": num_r_inst if num_r_inst > 0 else None,
            }

            flow_pred_cond, _ = self.generator(**generator_kwargs)

            if use_cfg and uncon_context is not None:
                generator_kwargs["context"] = uncon_context
                flow_pred_uncond, _ = self.generator(**generator_kwargs)
                flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_cond - flow_pred_uncond)
            else:
                flow_pred = flow_pred_cond

            # Extract only T frames flow prediction
            flow_pred_t = flow_pred[:, :num_t]

            latents = sample_scheduler.step(flow_pred_t, t, latents, return_dict=False)[0]

            denoising_pbar.set_postfix(
                {"timestep": f"{t:.3f}", "step": f"{step_idx + 1}/{len(sample_scheduler.timesteps)}"}
            )

        # ========== CPU offload: move generator to CPU, VAE to GPU for decoding ==========
        if cpu_offload:
            _t0 = _time.time()
            self.generator.to("cpu")
            torch.cuda.empty_cache()
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)
            print(f"    [offload] VAE → GPU  {_time.time() - _t0:.1f}s")

        # ========== Decode latents ==========
        _t0 = _time.time()
        videos = self.vae.decode_to_pixel(latents, use_cache=False)
        videos = (videos + 1.0) / 2.0
        videos = videos.clamp(0, 1)

        generated_video = videos[0]  # [T, 3, H, W]
        generated_video = (generated_video * 255).byte().cpu()
        generated_video = rearrange(generated_video, "t c h w -> t h w c")
        print(f"    [decode] VAE decode  {_time.time() - _t0:.1f}s")

        if cpu_offload:
            _t0 = _time.time()
            self.vae.model.to("cpu")
            self.vae.mean = self.vae.mean.to("cpu")
            self.vae.std = self.vae.std.to("cpu")
            torch.cuda.empty_cache()
            print(f"    [offload] VAE → CPU  {_time.time() - _t0:.1f}s")

        return generated_video

    @torch.inference_mode()
    def run_single_iteration_few_step(
        self,
        first_frame: Image.Image,
        target_scene_proj: torch.Tensor,
        prompt: str,
        num_frames: int,
        infer_steps: int = 50,  # kept for signature compatibility
        guidance_scale: float | None = None,
        use_cfg: bool = True,
        cpu_offload: bool = False,
        preceding_frames: np.ndarray | None = None,
        preceding_scene_proj: torch.Tensor | None = None,
        reference_frames: np.ndarray | None = None,
        target_fg_proj: torch.Tensor | None = None,
        preceding_fg_proj: torch.Tensor | None = None,
        seed: int | None = None,
        sp_context_scale: float = 1.0,
        preceding_noise_timestep: int = 0,
        instance_reference_frames: np.ndarray | None = None,
        denoising_step_list: list[float] | None = None,
    ) -> torch.Tensor:
        """
        Run a single iteration using CausVid-style few-step bidirectional inference.

        Denoising loop:
        1. Predict x0 at current timestep.
        2. If not final step, add fresh Gaussian noise to x0 at next timestep.
        3. Repeat until the final timestep.

        Note:
        - CFG branch is controlled by *use_cfg*.
        - For event-centric usage, caller sets:
          guidance_scale is None -> use_cfg=False
          guidance_scale has value -> use_cfg=True
        """
        if use_cfg and guidance_scale is None:
            guidance_scale = getattr(self.config, "guidance_scale", 5.0)

        # Calculate latent frames
        num_latent_frames = self._pixel_to_latent_frames(num_frames)
        num_t = num_latent_frames

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # ========== CPU offload: move VAE/CLIP/TextEncoder to GPU for encoding ==========
        if cpu_offload:
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)
            self.text_encoder.to(self.device)
            torch.cuda.empty_cache()

        # ========== Prepare preceding latents (P) ==========
        num_p = 0
        preceding_latent = None
        if preceding_frames is not None and len(preceding_frames) > 0:
            preceding_latent = self._encode_frames_to_latent(preceding_frames)
            num_p = preceding_latent.shape[1]

            # Add noise to preceding frames (P9 only, not P1)
            if preceding_noise_timestep > 0 and num_p > 1:
                _noise_sched = FlowMatchScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    num_inference_steps=self.num_train_timesteps,
                    shift=1.0,
                )
                noise_p = torch.randn_like(preceding_latent)
                t_p = torch.full(
                    [num_p],
                    preceding_noise_timestep,
                    device=self.device,
                    dtype=torch.float32,
                )
                preceding_latent = _noise_sched.add_noise(
                    preceding_latent.squeeze(0),
                    noise_p.squeeze(0),
                    t_p,
                ).unsqueeze(0)
                print(f"  Added noise to preceding frames: timestep={preceding_noise_timestep}")

        # ========== Prepare reference latents (R_scene + R_inst) ==========
        num_r_scene = 0
        reference_scene_latent = None
        if reference_frames is not None and len(reference_frames) > 0:
            ref_latents = []
            for i in range(len(reference_frames)):
                single = self._encode_frames_to_latent(reference_frames[i : i + 1])  # [1, 1, C, h, w]
                ref_latents.append(single)
            reference_scene_latent = torch.cat(ref_latents, dim=1)  # [1, R_scene, C, h, w]
            num_r_scene = reference_scene_latent.shape[1]

        num_r_inst = 0
        reference_inst_latent = None
        if instance_reference_frames is not None and len(instance_reference_frames) > 0:
            inst_latents = []
            for i in range(len(instance_reference_frames)):
                single = self._encode_frames_to_latent(instance_reference_frames[i : i + 1])
                inst_latents.append(single)
            reference_inst_latent = torch.cat(inst_latents, dim=1)  # [1, R_inst, C, h, w]
            num_r_inst = reference_inst_latent.shape[1]

        num_r = num_r_scene + num_r_inst
        reference_latent = None
        if num_r > 0:
            parts = []
            if reference_scene_latent is not None:
                parts.append(reference_scene_latent)
            if reference_inst_latent is not None:
                parts.append(reference_inst_latent)
            reference_latent = torch.cat(parts, dim=1)  # [1, R, C, h, w]

        # ========== Prepare text conditioning ==========
        text_conditional_dict = self.text_encoder(text_prompts=[prompt])
        context = text_conditional_dict["prompt_embeds"]
        context = [c.to(self.device) for c in context]

        uncon_context = None
        if use_cfg:
            unconditional_dict = self.text_encoder(text_prompts=[self.negative_prompt])
            uncon_context = unconditional_dict["prompt_embeds"]
            uncon_context = [c.to(self.device) for c in uncon_context]

        # ========== CPU offload: move encoders to CPU, generator to GPU ==========
        if cpu_offload:
            self.text_encoder.to("cpu")
            self.vae.model.to("cpu")
            self.vae.mean = self.vae.mean.to("cpu")
            self.vae.std = self.vae.std.to("cpu")
            torch.cuda.empty_cache()
            self.generator.to(self.device)

        # ========== Prepare State Adapter context ==========
        sp_context = self._prepare_sp_context(
            target_scene_proj=target_scene_proj,
            target_fg_proj=target_fg_proj,
            preceding_scene_proj=preceding_scene_proj,
            preceding_fg_proj=preceding_fg_proj,
            num_t=num_t,
            num_p=num_p,
        )

        # ========== Few-step denoising schedule ==========
        few_step_timesteps = self._resolve_few_step_timesteps(
            denoising_step_list=denoising_step_list,
        )
        # Initial noisy target
        latents = torch.randn(
            1, num_latent_frames, self.latent_channels, self.h, self.w, device=self.device, dtype=self.dtype
        )

        # Base scheduler for add_noise between explicit few steps.
        # This matches CausVid's bidirectional few-step inference behavior.
        base_scheduler = self.generator.get_scheduler()

        denoising_pbar = tqdm(
            enumerate(few_step_timesteps),
            total=len(few_step_timesteps),
            desc=f"Denoising (T={num_t}, P={num_p}, R_scene={num_r_scene}, R_inst={num_r_inst})",
        )

        for step_idx, current_timestep in denoising_pbar:
            timestep_t = current_timestep * torch.ones([1, num_t], device=self.device, dtype=torch.float32)
            timestep_parts = [timestep_t]
            if num_p > 0:
                timestep_parts.append(torch.zeros([1, num_p], device=self.device, dtype=torch.float32))
            if num_r > 0:
                timestep_parts.append(torch.zeros([1, num_r], device=self.device, dtype=torch.float32))
            timestep = torch.cat(timestep_parts, dim=1)

            latent_parts = [latents]
            if preceding_latent is not None and num_p > 0:
                latent_parts.append(preceding_latent)
            if reference_latent is not None and num_r > 0:
                latent_parts.append(reference_latent)
            combined_latents = torch.cat(latent_parts, dim=1)

            generator_kwargs = {
                "noisy_image_or_video": combined_latents,
                "timestep": timestep,
                "context": context,
                "clip_fea": None,
                "y": None,
                "sp_context": sp_context,
                "sp_context_scale": float(sp_context_scale),
                "num_t": num_t,
                "num_p": num_p,
                "num_r": num_r,
                "num_r_scene": num_r_scene if num_r_inst > 0 else None,
                "num_r_inst": num_r_inst if num_r_inst > 0 else None,
            }

            _, pred_x0_cond = self.generator(**generator_kwargs)
            pred_x0_cond_t = pred_x0_cond[:, :num_t]

            if use_cfg and uncon_context is not None:
                generator_kwargs["context"] = uncon_context
                _, pred_x0_uncond = self.generator(**generator_kwargs)
                pred_x0_uncond_t = pred_x0_uncond[:, :num_t]
                pred_x0 = pred_x0_uncond_t + guidance_scale * (pred_x0_cond_t - pred_x0_uncond_t)
            else:
                pred_x0 = pred_x0_cond_t

            if step_idx < len(few_step_timesteps) - 1:
                next_timestep = few_step_timesteps[step_idx + 1]
                next_timestep_t = next_timestep * torch.ones([1, num_t], device=self.device, dtype=torch.float32)
                latents = base_scheduler.add_noise(
                    pred_x0.flatten(0, 1),
                    torch.randn_like(pred_x0.flatten(0, 1)),
                    next_timestep_t.flatten(0, 1),
                ).unflatten(0, pred_x0.shape[:2])
            else:
                latents = pred_x0

            denoising_pbar.set_postfix(
                {"timestep": f"{float(current_timestep):.3f}", "step": f"{step_idx + 1}/{len(few_step_timesteps)}"}
            )

        # ========== CPU offload: move generator to CPU, VAE to GPU for decoding ==========
        if cpu_offload:
            self.generator.to("cpu")
            torch.cuda.empty_cache()
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)

        # ========== Decode latents ==========
        videos = self.vae.decode_to_pixel(latents, use_cache=False)
        videos = (videos + 1.0) / 2.0
        videos = videos.clamp(0, 1)

        generated_video = videos[0]  # [T, 3, H, W]
        generated_video = (generated_video * 255).byte().cpu()
        generated_video = rearrange(generated_video, "t c h w -> t h w c")

        if cpu_offload:
            self.vae.model.to("cpu")
            self.vae.mean = self.vae.mean.to("cpu")
            self.vae.std = self.vae.std.to("cpu")
            torch.cuda.empty_cache()

        return generated_video

    def run_iterative_inference(
        self,
        first_frame: Image.Image,
        prompt: str,
        num_frames: int,
        first_frame_image: str,
        geometry_poses_c2w: np.ndarray,
        frames_per_iter: int | None = None,
        options: BackboneInferenceOptions | None = None,
        target_scene_proj: torch.Tensor | None = None,
        output_paths: BackboneOutputPaths | None = None,
        iter_inputs: dict[int, dict] | None = None,
    ) -> BackboneInferenceResult:
        """
        Run full iterative inference with optional point cloud updates.

        All data must be loaded before calling this function. This function only
        performs inference, not data loading.

        Args:
            first_frame: First frame image (required, pre-loaded)
            prompt: Text prompt (required, pre-loaded)
            num_frames: Total number of frames to generate
            first_frame_image: Path to the first-frame image file
            geometry_poses_c2w: Camera extrinsics from geometry.npz (required, pre-loaded)
            frames_per_iter: Frames per iteration
            options: Inference options
            target_scene_proj: Pre-computed scene projection (optional)
            output_paths: Output paths for saving intermediate results
        """
        if options is None:
            options = BackboneInferenceOptions()

        return run_iterative_inference(
            pipeline=self,
            first_frame=first_frame,
            prompt=prompt,
            num_frames=num_frames,
            frames_per_iter=frames_per_iter,
            options=options,
            first_frame_image=first_frame_image,
            geometry_poses_c2w=geometry_poses_c2w,
            target_scene_proj=target_scene_proj,
            output_paths=output_paths,
            iter_inputs=iter_inputs,
        )

    def save_iterative_outputs(
        self,
        result: BackboneInferenceResult,
        output_paths: BackboneOutputPaths,
        options: BackboneInferenceOptions,
        output_name: str | None = None,
        input_dir: str | None = None,
    ) -> None:
        """Save videos, point clouds, and metadata for iterative inference."""
        save_iterative_outputs(
            pipeline=self,
            result=result,
            output_paths=output_paths,
            options=options,
            output_name=output_name,
            input_dir=input_dir,
        )


# =============================================================================
# Iterative Inference Utilities
# =============================================================================


def load_frame_from_image(image_path: str, target_size: tuple[int, int]) -> Image.Image:
    """Load the first frame from an image file.

    Args:
        image_path: Path to image file (png, jpg, etc.)
        target_size: (width, height) to resize to.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def retrieve_reference_frames(
    target_frame_indices: list[int],
    all_generated_frames: list[np.ndarray],
    frame_visible_points: dict[int, np.ndarray],
    points_world: np.ndarray,
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    max_reference_frames: int = 7,
    iou_threshold: float = 0.1,
    voxel_size_iou: float = 0.1,
    iou_device: str | torch.device | None = "auto",
    max_iou_frames: int = 50,
    max_iou_points: int = 50000,
) -> tuple[np.ndarray | None, list[int], list[str]]:
    """Retrieve reference frames following LiveWorld paper Algorithm 1.

    For each target frame, find the candidate frame with maximal 3D IoU
    (spatial overlap). Add it as a reference only when IoU > iou_threshold,
    and keep at most ``max_reference_frames`` unique references.

    Returns:
        Tuple of (reference_frames, ref_indices, diagnostics):
        - reference_frames: [R, H, W, 3] or None
        - ref_indices: list of selected frame indices
        - diagnostics: list of human-readable diagnostic lines
    """
    diag: list[str] = []
    diag.append("=== retrieve_reference_frames ===")
    diag.append(f"target_frame_indices: {list(target_frame_indices)}")
    diag.append(f"max_reference_frames: {max_reference_frames}, iou_threshold: {iou_threshold}")
    diag.append(f"voxel_size_iou: {voxel_size_iou}")
    diag.append(f"max_iou_points: {max_iou_points}")
    diag.append(f"num_generated_frames: {len(all_generated_frames)}")
    diag.append(f"num_world_points: {len(points_world) if points_world is not None else 0}")
    diag.append(f"image_size: {image_size}")
    diag.append("")

    if max_reference_frames <= 0 or len(all_generated_frames) == 0 or points_world is None:
        diag.append(
            f"EARLY EXIT: max_refs={max_reference_frames}, n_gen={len(all_generated_frames)}, pts={'None' if points_world is None else len(points_world)}"
        )
        return None, [], diag

    if intrinsics.ndim == 2:
        intrinsics = np.tile(intrinsics[None], (len(poses_c2w), 1, 1))

    target_list = [int(idx) for idx in target_frame_indices if 0 <= int(idx) < len(poses_c2w)]
    if not target_list:
        diag.append("EARLY EXIT: no valid target indices within poses_c2w range")
        return None, [], diag

    # Resolve GPU device for visibility + IoU.
    if iou_device is None:
        _vis_device = None
    elif str(iou_device) == "auto":
        _vis_device = "cuda" if torch.cuda.is_available() else None
    else:
        _vis_device = str(iou_device)

    # Compute scene maps for target frames from the current world point cloud.
    _t_vis = _time.time()
    height, width = image_size
    total_pixels = max(1, height * width)
    min_coverage = 0.20
    target_scene_maps: list[np.ndarray] = []
    target_scene_map_indices: list[int] = []
    diag.append(f"--- Target frame coverage (min_coverage={min_coverage:.2f}, device={_vis_device}) ---")

    if _vis_device is not None and str(_vis_device).startswith("cuda"):
        # GPU path: batch visibility + coverage computation.
        vis_results = _get_visible_points_and_coverage_gpu(
            points_world,
            poses_c2w,
            intrinsics,
            target_list,
            image_size,
            device=_vis_device,
        )
        for frame_idx, visible_pts, coverage, n_unique in vis_results:
            if len(visible_pts) == 0:
                diag.append(f"  target frame {frame_idx}: 0 visible points -> SKIP")
                continue
            if coverage < min_coverage:
                diag.append(
                    f"  target frame {frame_idx}: {len(visible_pts)} visible pts, coverage={coverage:.4f} ({n_unique}/{total_pixels} pixels) -> SKIP (< {min_coverage})"
                )
                continue
            diag.append(
                f"  target frame {frame_idx}: {len(visible_pts)} visible pts, coverage={coverage:.4f} ({n_unique}/{total_pixels} pixels) -> PASS"
            )
            target_scene_maps.append(visible_pts)
            target_scene_map_indices.append(frame_idx)
    else:
        # CPU fallback.
        for frame_idx in target_list:
            intr_idx = _safe_frame_index(frame_idx, len(intrinsics))
            visible_pts, _ = get_visible_points_for_frame(
                points_world, None, poses_c2w[frame_idx], intrinsics[intr_idx], image_size
            )
            if len(visible_pts) == 0:
                diag.append(f"  target frame {frame_idx}: 0 visible points -> SKIP")
                continue
            pose_w2c = np.linalg.inv(poses_c2w[frame_idx])
            pts_cam = (pose_w2c[:3, :3] @ visible_pts.T).T + pose_w2c[:3, 3]
            K = intrinsics[intr_idx]
            pts_proj = (K @ pts_cam.T).T
            px = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-8)
            px_int = np.floor(px).astype(np.int64)
            pixel_ids = px_int[:, 1] * width + px_int[:, 0]
            n_unique = len(np.unique(pixel_ids))
            coverage = n_unique / total_pixels
            if coverage < min_coverage:
                diag.append(
                    f"  target frame {frame_idx}: {len(visible_pts)} visible pts, coverage={coverage:.4f} ({n_unique}/{total_pixels} pixels) -> SKIP (< {min_coverage})"
                )
                continue
            diag.append(
                f"  target frame {frame_idx}: {len(visible_pts)} visible pts, coverage={coverage:.4f} ({n_unique}/{total_pixels} pixels) -> PASS"
            )
            target_scene_maps.append(visible_pts)
            target_scene_map_indices.append(frame_idx)

    _dt_vis = _time.time() - _t_vis
    diag.append(f"  => {len(target_scene_maps)}/{len(target_list)} targets passed coverage filter  ({_dt_vis:.1f}s)")
    diag.append("")

    if not target_scene_maps:
        diag.append("EARLY EXIT: no target frames passed coverage filter")
        return None, [], diag

    # Prepare candidate scene maps from historical generated frames.
    hist_items = sorted(frame_visible_points.items(), key=lambda x: x[0])
    hist_items = [
        (idx, pts)
        for idx, pts in hist_items
        if pts is not None and len(pts) > 0 and 0 <= idx < len(all_generated_frames)
    ]
    diag.append("--- Historical candidates ---")
    diag.append(f"  {len(hist_items)} candidates from frame_visible_points (frames: {[idx for idx, _ in hist_items]})")
    for idx, pts in hist_items:
        diag.append(f"    hist frame {idx}: {len(pts)} visible pts")
    diag.append("")

    if not hist_items:
        diag.append("EARLY EXIT: no historical candidates with visible points")
        return None, [], diag

    # For each target frame, find its best candidate by 3D IoU.
    _t_iou = _time.time()
    resolved_device = _vis_device
    use_gpu = resolved_device is not None and str(resolved_device).startswith("cuda")

    selected_set: set[int] = set()
    selected_indices: list[int] = []

    diag.append(f"--- IoU matching (use_gpu={use_gpu}, device={resolved_device}) ---")
    for ti, target_pts in enumerate(target_scene_maps):
        target_fidx = target_scene_map_indices[ti]
        if len(selected_indices) >= max_reference_frames:
            diag.append(
                f"  target frame {target_fidx}: SKIP (already reached max_reference_frames={max_reference_frames})"
            )
            break
        if use_gpu:
            scores = compute_3d_iou_batched(
                target_pts,
                hist_items,
                voxel_size_iou,
                device=resolved_device,
                max_points=max_iou_points,
            )
        else:
            scores = []
            target_ds = _voxel_downsample(target_pts, voxel_size_iou)
            for hist_idx, hist_points in hist_items:
                hist_ds = _voxel_downsample(hist_points, voxel_size_iou)
                iou = compute_3d_iou(target_ds, hist_ds, voxel_size_iou, device=None)
                scores.append((hist_idx, iou))

        # Filter to above threshold, then sort oldest-first.
        passing = [(idx, iou) for idx, iou in scores if iou > iou_threshold]
        passing.sort(key=lambda x: x[0])  # oldest first

        # For diagnostics: show all scores sorted by IoU descending.
        scores.sort(key=lambda x: x[1], reverse=True)
        diag.append(f"  target frame {target_fidx} ({len(target_pts)} pts):")
        diag.append(f"    {len(passing)}/{len(scores)} candidates above threshold {iou_threshold}")
        top_n = min(10, len(scores))
        for rank, (hist_idx, iou) in enumerate(scores[:top_n]):
            marker = ""
            if hist_idx in selected_set:
                marker = " [already selected]"
            elif iou <= iou_threshold:
                marker = f" [below threshold {iou_threshold}]"
            diag.append(f"    #{rank}: hist frame {hist_idx}, IoU={iou:.6f}{marker}")
        if len(scores) > top_n:
            diag.append(f"    ... ({len(scores) - top_n} more candidates omitted)")

        best_hist_idx = None
        for hist_idx, iou in passing:
            if hist_idx not in selected_set:
                best_hist_idx = hist_idx
                best_iou = iou
                break
        if best_hist_idx is not None:
            selected_set.add(best_hist_idx)
            selected_indices.append(best_hist_idx)
            diag.append(f"    => SELECTED hist frame {best_hist_idx} (IoU={best_iou:.6f}) [oldest-first]")
        else:
            diag.append("    => NO match (all below threshold or already selected)")

    _dt_iou = _time.time() - _t_iou
    diag.append(f"  IoU matching took {_dt_iou:.1f}s")
    diag.append(f"  [timing] visible_pts={_dt_vis:.1f}s  iou_match={_dt_iou:.1f}s  total={_dt_vis + _dt_iou:.1f}s")
    diag.append("")

    if not selected_indices:
        diag.append("RESULT: no reference frames selected")
        return None, [], diag

    reference_frames: list[np.ndarray] = []
    final_indices: list[int] = []
    for idx in selected_indices:
        if 0 <= idx < len(all_generated_frames):
            reference_frames.append(all_generated_frames[idx])
            final_indices.append(idx)

    if not reference_frames:
        diag.append("RESULT: no reference frames (indices out of range)")
        return None, [], diag

    diag.append(f"RESULT: {len(final_indices)} reference frames selected: {final_indices}")
    return np.stack(reference_frames, axis=0), final_indices, diag


# =============================================================================
# Scene Projection Generation
# =============================================================================


def generate_scene_projection_from_pointcloud(
    points_world: np.ndarray,
    colors: np.ndarray,
    target_frames: list[int],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    output_size: tuple[int, int],
    intrinsics_size: tuple[int, int],
    vae,
    device,
    dtype,
    density_max_pixels: int | None = None,
    density_rng: np.random.Generator | None = None,
    density_blue_noise: np.ndarray | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Generate scene projection latent from a point cloud."""

    def _normalize_intrinsics_matrix(intrinsics_in: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Ensure intrinsics is a 3x3 matrix."""
        if intrinsics_in.ndim == 2 and intrinsics_in.shape == (3, 3):
            return intrinsics_in.astype(np.float32)
        flat = intrinsics_in.reshape(-1)
        if flat.size >= 4:
            fx, fy, cx, cy = flat[:4]
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        H, W = size_hw
        focal = max(H, W) * 1.2
        return np.array(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    height, width = output_size
    proc_h, proc_w = intrinsics_size

    device_str = str(device) if not isinstance(device, str) else device
    use_density = density_max_pixels is not None and density_max_pixels > 0

    projections_list = []
    for frame_idx in target_frames:
        pose_idx = _safe_frame_index(frame_idx, len(poses_c2w))
        if intrinsics.ndim == 2:
            K_raw = intrinsics
        else:
            intr_idx = _safe_frame_index(frame_idx, len(intrinsics))
            K_raw = intrinsics[intr_idx]
        K_raw = _normalize_intrinsics_matrix(K_raw, (proc_h, proc_w))
        K_scaled = scale_intrinsics_from_size(K_raw, (proc_h, proc_w), (height, width))

        if use_density:
            pts, cols = _limit_points_by_density(
                points_world,
                colors,
                poses_c2w[pose_idx],
                K_scaled,
                (height, width),
                density_max_pixels,
                rng=density_rng,
                blue_noise=density_blue_noise,
            )
        else:
            pts, cols = points_world, colors

        proj = render_projection(
            points_world=pts,
            K=K_scaled,
            c2w=poses_c2w[pose_idx],
            image_size=(height, width),
            channels=["rgb"],
            colors=cols,
            fill_holes_kernel=0,
            device=device_str,
        )
        projections_list.append(proj)

    projections = np.stack(projections_list, axis=0)
    projections = projections.transpose(0, 3, 1, 2)

    proj_tensor = torch.from_numpy(projections).float() / 127.5 - 1.0

    with torch.no_grad():
        # Ensure VAE is on the target device for encoding
        vae_device = next(vae.model.parameters()).device
        if vae_device != device:
            vae.model.to(device)
            vae.mean = vae.mean.to(device)
            vae.std = vae.std.to(device)

        proj_tensor = proj_tensor.to(device=device, dtype=dtype)
        proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        scene_proj_latent = vae.encode_to_latent(proj_tensor)
        scene_proj_latent = scene_proj_latent.squeeze(0).permute(1, 0, 2, 3)

    return scene_proj_latent


def generate_scene_projection_with_handler(
    first_frame_image: str,
    target_frames: list[int],
    output_size: tuple[int, int],
    handler,
    geometry_poses_c2w: np.ndarray,
    vae,
    device,
    dtype,
    options,
) -> tuple[Any, ...]:
    """Generate initial scene projection using the 3D handler (Stream3R or MapAnything).

    Workflow:
    1. Qwen + SAM3 segments dynamic objects in first frame
    2. Handler reconstructs first-frame point cloud (replaces depth estimation)
    3. Handler renders scene projections for target frames

    Args:
        first_frame_image: Path to the first-frame image file.

    Returns:
        (scene_proj_latent, points_world, colors, first_frame,
         intrinsics, alignment_transform, intrinsics_size)
    """
    if not os.path.exists(first_frame_image):
        raise FileNotFoundError(f"First frame image not found: {first_frame_image}")

    first_frame = cv2.cvtColor(cv2.imread(first_frame_image), cv2.COLOR_BGR2RGB)
    print(f"  Loaded first frame from {first_frame_image}")

    print("\n  [Step 1/3] Detecting dynamic objects with Qwen + SAM3...")
    if isinstance(device, int):
        device_str = f"cuda:{device}"
    else:
        device_str = str(device)
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"

    qwen_extractor = Qwen3VLEntityExtractor(
        model_path=options.qwen_model_path,
        device=device_str,
    )
    dynamic_prompts, _ = qwen_extractor.extract(first_frame_image)
    if options.cpu_offload:
        del qwen_extractor
        torch.cuda.empty_cache()

    fg_prompts_only = list(dynamic_prompts) if dynamic_prompts else []
    all_prompts_init = fg_prompts_only + ["sky"]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.

    sam3_segmenter = Sam3VideoSegmenter(
        checkpoint_path=options.sam3_model_path,
    )
    first_frame_pil = Image.fromarray(first_frame)
    per_category = sam3_segmenter.segment_per_category(
        video_path=[first_frame_pil],
        prompts=all_prompts_init,
        frame_index=0,
        expected_frames=1,
    )
    if options.cpu_offload:
        del sam3_segmenter
        torch.cuda.empty_cache()

    # Build combined dynamic_mask (fg objects + sky) for point cloud exclusion
    # and fg_only_mask (fg objects only, no sky) for foreground projection
    dynamic_mask = None
    fg_only_mask = None
    if per_category:
        h_mask, w_mask = first_frame.shape[:2]
        combined = np.zeros((h_mask, w_mask), dtype=bool)
        fg_combined = np.zeros((h_mask, w_mask), dtype=bool)
        for prompt, masks in per_category.items():
            if masks.size == 0:
                continue
            frame_mask = masks[0]  # first frame only
            combined |= frame_mask
            if prompt.lower() != "sky":
                fg_combined |= frame_mask
        dynamic_mask = combined if combined.any() else None
        fg_only_mask = fg_combined if fg_combined.any() else None
        print(f"  Dynamic mask: {int(combined.sum())} pixels ({len(per_category)} categories)")
        if fg_only_mask is not None:
            print(f"  FG-only mask: {int(fg_combined.sum())} pixels (excl. sky)")

    print("\n  [Step 2/3] Reconstructing first-frame point cloud with handler...")
    recon: ReconstructionResult = handler.reconstruct_first_frame(
        frame=first_frame,
        geometry_poses_c2w=geometry_poses_c2w,
        dynamic_mask=dynamic_mask,
        options=options,
    )
    if options.cpu_offload and hasattr(handler, "offload_to_cpu"):
        handler.offload_to_cpu()

    print(f"\n  [Step 3/3] Rendering scene projections for {len(target_frames)} target frames...")
    if options.cpu_offload:
        vae.model.to(device)
        vae.mean = vae.mean.to(device)
        vae.std = vae.std.to(device)
    scene_proj_latent = handler.render_scene_projection(
        target_frame_indices=target_frames,
        output_size=output_size,
        vae=vae,
        device=device,
        dtype=dtype,
    )

    return (
        scene_proj_latent,
        recon.points_world,
        recon.colors,
        first_frame,
        recon.intrinsics,
        recon.alignment_transform,
        recon.intrinsics_size,
        dynamic_mask,
        fg_only_mask,
    )


# =============================================================================
# Iteration Processing
# =============================================================================


def process_iteration(
    pipeline: UnifiedBackbonePipeline,
    iter_idx: int,
    iteration_plan: list[tuple[int, int, int]],
    state: IterationState,
    prompt: str,
    iter_input: dict | None,
    options: BackboneInferenceOptions,
    t0: int,
    output_size: tuple[int, int],
    output_paths: BackboneOutputPaths | None = None,
) -> IterationResult:
    """Process a single iteration of multi-round inference."""
    height, width = output_size
    device = pipeline.device

    output_start, output_end, model_frames = iteration_plan[iter_idx]
    new_frames_count = output_end - output_start

    print("\n" + "=" * 60)
    print(f"ITERATION {iter_idx + 1}/{len(iteration_plan)}")
    print(f"  Output frames: {output_start} to {output_end - 1} ({new_frames_count} new frames)")
    print(f"  Model generates: {model_frames} frames")
    print("=" * 60)

    iter_prompt = prompt
    iter_scene_proj = None
    iter_fg_proj = None
    if iter_input:
        override_prompt = iter_input.get("prompt")
        if override_prompt:
            iter_prompt = override_prompt
        iter_scene_proj = iter_input.get("scene_proj")
        iter_fg_proj = iter_input.get("fg_proj")
        if iter_scene_proj is not None:
            iter_scene_proj = iter_scene_proj.to(device=device, dtype=pipeline.dtype)
            state.target_scene_proj = iter_scene_proj
        if iter_fg_proj is not None:
            iter_fg_proj = iter_fg_proj.to(device=device, dtype=pipeline.dtype)

    # T2V mode: NO overlap
    # Generated frames are 1-indexed in video (frame 0 is input first frame)
    # output_start/output_end are 0-indexed generated frame indices
    # clip frame = t0 + 1 + generated_frame_index
    target_frame_indices = list(range(t0 + 1 + output_start, t0 + 1 + output_end))

    if options.limit_projection_density and iter_idx == 0 and state.projection_density_max_pixels is None:  # noqa: SIM102 - Statement form kept for readability.
        if state.points_world is not None and state.intrinsics is not None:
            intr_size = state.intrinsics_size
            state.projection_density_max_pixels = _compute_projection_density_max_pixels(
                points_world=state.points_world,
                poses_c2w=state.poses_c2w,
                intrinsics=state.intrinsics,
                target_frames=target_frame_indices,
                output_size=output_size,
                intrinsics_size=intr_size,
            )
            max_pixels = int(state.projection_density_max_pixels or 0)
            print(f"  Projection density cap computed (max_pixels={max_pixels})")

    if options.limit_projection_density and state.projection_density_blue_noise is None:
        tile_size = max(8, int(options.projection_density_noise_size))
        rng = state.projection_density_rng if state.projection_density_rng is not None else np.random.default_rng()
        state.projection_density_blue_noise = generate_blue_noise_tile(tile_size, rng)
        print(f"  Blue noise tile initialized (size={tile_size})")

    vae_for_proj_on_gpu = False
    if iter_scene_proj is None and iter_idx > 0 and state.points_world is not None:
        print("\n[Re-rendering scene projection for this iteration...]")
        print(
            f"  Pose indices: {target_frame_indices[0]} to {target_frame_indices[-1]} "
            f"({len(target_frame_indices)} poses)"
        )
        intr_size = state.intrinsics_size
        if options.cpu_offload:
            pipeline.vae.model.to(device)
            pipeline.vae.mean = pipeline.vae.mean.to(device)
            pipeline.vae.std = pipeline.vae.std.to(device)
            vae_for_proj_on_gpu = True
        state.target_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=state.points_world,
            colors=state.colors,
            target_frames=target_frame_indices,
            poses_c2w=state.poses_c2w,
            intrinsics=state.intrinsics,
            output_size=output_size,
            intrinsics_size=intr_size,
            vae=pipeline.vae,
            device=device,
            dtype=pipeline.dtype,
            density_max_pixels=state.projection_density_max_pixels if options.limit_projection_density else None,
            density_rng=state.projection_density_rng if options.limit_projection_density else None,
            density_blue_noise=state.projection_density_blue_noise if options.limit_projection_density else None,
        )
        print(f"  Scene projection shape: {state.target_scene_proj.shape}")

    preceding_frames = None
    preceding_scene_proj = None
    preceding_fg_proj = None

    if iter_idx == 0:
        max_preceding = max(0, int(options.max_preceding_frames_first_iter))
    else:
        max_preceding = max(0, int(options.max_preceding_frames_other_iter))
    if max_preceding > 0:
        if iter_idx == 0:
            # First iteration: first frame becomes preceding frame (P1 mode)
            # The preceding_scene_proj is the world point cloud projected back to the
            # first frame's camera pose. Since the point cloud was reconstructed from
            # the first frame, this is essentially the original image's background.
            print("\n[Preparing preceding frames (P1 mode)...]")
            first_frame_np = np.array(state.current_first_frame)
            if max_preceding >= 1:
                preceding_frames = first_frame_np[np.newaxis, ...]
                print("  Using 1 preceding frame (first_frame from clip)")

            # For P1 mode, use the first frame's pose (t0) to project back
            if preceding_frames is not None and state.points_world is not None:
                print("  Generating preceding scene projection (P1: project to first frame pose)...")
                # Use the start frame pose (t0), which is where the point cloud was reconstructed from
                preceding_pose_indices = [t0]
                intr_size = state.intrinsics_size
                preceding_scene_proj = generate_scene_projection_from_pointcloud(
                    points_world=state.points_world,
                    colors=state.colors,
                    target_frames=preceding_pose_indices,
                    poses_c2w=state.poses_c2w,
                    intrinsics=state.intrinsics,
                    output_size=output_size,
                    intrinsics_size=intr_size,
                    vae=pipeline.vae,
                    device=device,
                    dtype=pipeline.dtype,
                    density_max_pixels=state.projection_density_max_pixels
                    if options.limit_projection_density
                    else None,
                    density_rng=state.projection_density_rng if options.limit_projection_density else None,
                    density_blue_noise=state.projection_density_blue_noise
                    if options.limit_projection_density
                    else None,
                )
                print(f"  Preceding scene projection shape: {preceding_scene_proj.shape}")

            # Generate preceding fg projection: first frame with background removed
            if pipeline.use_fg_proj and state.initial_fg_only_mask is not None:
                print("  Generating preceding fg projection (P1: foreground from first frame)...")
                fg_img = first_frame_np.copy()
                fg_mask = state.initial_fg_only_mask
                # Resize mask to match output_size if needed
                if fg_mask.shape[:2] != (output_size[0], output_size[1]):
                    fg_mask = cv2.resize(
                        fg_mask.astype(np.uint8),
                        (output_size[1], output_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                # Resize fg_img to output_size if needed
                if fg_img.shape[:2] != (output_size[0], output_size[1]):
                    fg_img = cv2.resize(fg_img, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
                # Zero out background (keep only foreground)
                fg_img[~fg_mask] = 0
                # Encode to latent: [1, H, W, 3] -> [1, 3, H, W] -> normalize -> [1, 3, 1, H, W]
                fg_tensor = torch.from_numpy(fg_img).float() / 127.5 - 1.0  # [-1, 1]
                fg_tensor = fg_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
                fg_tensor = fg_tensor.to(device=device, dtype=pipeline.dtype)
                with torch.no_grad():
                    vae_device = next(pipeline.vae.model.parameters()).device
                    if vae_device != device:
                        pipeline.vae.model.to(device)
                        pipeline.vae.mean = pipeline.vae.mean.to(device)
                        pipeline.vae.std = pipeline.vae.std.to(device)
                    preceding_fg_proj = pipeline.vae.encode_to_latent(fg_tensor)  # [1, 16, 1, h, w]
                    preceding_fg_proj = preceding_fg_proj.squeeze(0).permute(1, 0, 2, 3)  # [16, 1, h, w]
                print(f"  Preceding fg projection shape: {preceding_fg_proj.shape}")
        elif len(state.all_generated_frames) > 0:
            # P9 mode: use last N generated frames as preceding
            print("\n[Preparing preceding frames (P9 mode)...]")
            num_use = min(max_preceding, len(state.all_generated_frames))
            preceding_start_idx = len(state.all_generated_frames) - num_use
            preceding_frames = np.stack(state.all_generated_frames[preceding_start_idx:], axis=0)
            print(f"  Using {len(preceding_frames)} preceding frames from previous iteration")

            if preceding_frames is not None and state.points_world is not None:
                print("  Generating preceding scene projection...")
                preceding_pose_start = target_frame_indices[0] - len(preceding_frames)
                # Clamp to valid pose range (in case preceding frames extend before t0)
                preceding_pose_start = max(0, preceding_pose_start)
                preceding_pose_indices = list(range(preceding_pose_start, target_frame_indices[0]))
                if preceding_pose_indices:
                    intr_size = state.intrinsics_size
                    preceding_scene_proj = generate_scene_projection_from_pointcloud(
                        points_world=state.points_world,
                        colors=state.colors,
                        target_frames=preceding_pose_indices,
                        poses_c2w=state.poses_c2w,
                        intrinsics=state.intrinsics,
                        output_size=output_size,
                        intrinsics_size=intr_size,
                        vae=pipeline.vae,
                        device=device,
                        dtype=pipeline.dtype,
                        density_max_pixels=state.projection_density_max_pixels
                        if options.limit_projection_density
                        else None,
                        density_rng=state.projection_density_rng if options.limit_projection_density else None,
                        density_blue_noise=state.projection_density_blue_noise
                        if options.limit_projection_density
                        else None,
                    )
                    print(f"  Preceding scene projection shape: {preceding_scene_proj.shape}")

    if vae_for_proj_on_gpu:
        pipeline.vae.model.to("cpu")
        pipeline.vae.mean = pipeline.vae.mean.to("cpu")
        pipeline.vae.std = pipeline.vae.std.to("cpu")
        torch.cuda.empty_cache()

    reference_frames = None
    ref_indices: list[int] = []
    max_refs = max(0, int(options.max_reference_frames))
    if max_refs > 0 and iter_idx > 0 and state.points_world is not None and len(state.frame_visible_points) > 0:
        print("\n[Retrieving reference frames based on 3D IoU...]")
        reference_frames, ref_indices, _diag = retrieve_reference_frames(
            target_frame_indices=target_frame_indices,
            all_generated_frames=state.all_generated_frames,
            frame_visible_points=state.frame_visible_points,
            points_world=state.points_world,
            poses_c2w=state.poses_c2w,
            intrinsics=state.intrinsics,
            image_size=output_size,
            max_reference_frames=max_refs,
            iou_threshold=0.05,
            voxel_size_iou=options.voxel_size_iou or 0.002,
            iou_device=options.iou_device,
            max_iou_frames=options.max_iou_frames,
            max_iou_points=options.max_iou_points,
        )
        if reference_frames is not None and output_paths is not None:
            os.makedirs(output_paths.video_dir, exist_ok=True)
            ref_video_path = os.path.join(output_paths.video_dir, f"reference_iter_{iter_idx + 1:02d}.mp4")
            ref_video = reference_frames.astype(np.uint8)
            save_video_h264(ref_video_path, ref_video, fps=options.fps)

            ref_meta_path = os.path.join(output_paths.video_dir, f"reference_iter_{iter_idx + 1:02d}.json")
            with open(ref_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iter_idx": iter_idx,
                        "reference_indices": ref_indices,
                        "target_frame_indices": target_frame_indices,
                    },
                    f,
                    indent=2,
                )

    # Decode scene projection before generation so we can inspect the input early
    scene_proj_decoded = None
    if state.target_scene_proj is not None:
        if options.cpu_offload:
            pipeline.vae.model.to(device)
            pipeline.vae.mean = pipeline.vae.mean.to(device)
            pipeline.vae.std = pipeline.vae.std.to(device)
        with torch.no_grad():
            _proj_decode = state.target_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
            _proj_pixel = pipeline.vae.decode_to_pixel(_proj_decode, use_cache=False)
            _proj_pixel = ((_proj_pixel + 1.0) / 2.0).clamp(0, 1)
            scene_proj_decoded = (_proj_pixel[0] * 255).byte().cpu()
            scene_proj_decoded = rearrange(scene_proj_decoded, "t c h w -> t h w c")
        if options.cpu_offload:
            pipeline.vae.model.to("cpu")
            pipeline.vae.mean = pipeline.vae.mean.to("cpu")
            pipeline.vae.std = pipeline.vae.std.to("cpu")
            torch.cuda.empty_cache()
        if output_paths is not None:
            os.makedirs(output_paths.video_dir, exist_ok=True)
            _proj_path = os.path.join(
                output_paths.video_dir,
                f"iteration_{iter_idx + 1:02d}_scene_projection_input.mp4",
            )
            save_video_h264(_proj_path, scene_proj_decoded, fps=options.fps)
            print(f"  Saved scene projection input: {_proj_path} ({scene_proj_decoded.shape[0]} frames)")

    guidance_scale = getattr(pipeline.config, "guidance_scale", 5.0)

    if options.use_few_step or options.denoising_step_list is not None:
        generated_video = pipeline.run_single_iteration_few_step(
            first_frame=state.current_first_frame,
            target_scene_proj=state.target_scene_proj,
            prompt=iter_prompt,
            num_frames=model_frames,
            infer_steps=options.infer_steps,
            guidance_scale=guidance_scale,
            use_cfg=False,
            cpu_offload=options.cpu_offload,
            preceding_frames=preceding_frames,
            preceding_scene_proj=preceding_scene_proj,
            reference_frames=reference_frames,
            target_fg_proj=iter_fg_proj,
            preceding_fg_proj=preceding_fg_proj,
            seed=options.seed + iter_idx,
            sp_context_scale=options.sp_context_scale,
            preceding_noise_timestep=options.preceding_noise_timestep,
            denoising_step_list=options.denoising_step_list,
        )
    else:
        generated_video = pipeline.run_single_iteration(
            first_frame=state.current_first_frame,
            target_scene_proj=state.target_scene_proj,
            prompt=iter_prompt,
            num_frames=model_frames,
            infer_steps=options.infer_steps,
            guidance_scale=guidance_scale,
            use_cfg=not options.no_cfg,
            cpu_offload=options.cpu_offload,
            preceding_frames=preceding_frames,
            preceding_scene_proj=preceding_scene_proj,
            reference_frames=reference_frames,
            target_fg_proj=iter_fg_proj,
            preceding_fg_proj=preceding_fg_proj,
            seed=options.seed + iter_idx,
            sp_context_scale=options.sp_context_scale,
            preceding_noise_timestep=options.preceding_noise_timestep,
        )
    print(f"  Generated {generated_video.shape[0]} frames from model")

    # T2V mode: NO overlap, store all generated frames each iteration
    frames_to_store = [f.numpy() for f in generated_video]
    if iter_idx == 0:
        # First iteration: prepend input first frame
        first_frame_np = np.array(state.current_first_frame)
        frames_to_store = [first_frame_np] + frames_to_store  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        print(f"  Storing {len(frames_to_store)} frames (first_frame + {len(generated_video)} generated)")
    else:
        print(f"  Storing all {len(frames_to_store)} generated frames (no overlap)")

    _update_point_cloud_with_iteration(  # update after every round video generation
        iter_idx=iter_idx,
        generated_video=generated_video,
        target_frame_indices=target_frame_indices,
        state=state,
        options=options,
        output_size=output_size,
        pointcloud_dir=output_paths.pointcloud_dir if output_paths else None,
        device=device,
    )

    if output_paths is not None:
        if iter_idx == 0:
            first_frame_tensor = torch.from_numpy(np.array(state.current_first_frame)).unsqueeze(0)
            video_frames_to_save = torch.cat([first_frame_tensor, generated_video], dim=0)
        else:
            video_frames_to_save = generated_video

        # Save per-iteration generated video
        iter_gen_path = os.path.join(
            output_paths.video_dir,
            f"iteration_{iter_idx + 1:02d}_generated_frames{output_start}-{output_end - 1}.mp4",
        )
        save_video_h264(iter_gen_path, video_frames_to_save, fps=options.fps)
        print(f"  Saved generated video: {iter_gen_path} ({video_frames_to_save.shape[0]} frames)")

        # Save concat (generated + projection side-by-side)
        if scene_proj_decoded is not None:
            gen_frames = video_frames_to_save
            proj_frames = scene_proj_decoded

            # Align frame counts (iter0 has 1 extra first frame)
            if gen_frames.shape[0] == proj_frames.shape[0] + 1:
                pad = torch.zeros((1, proj_frames.shape[1], proj_frames.shape[2], 3), dtype=proj_frames.dtype)
                proj_frames = torch.cat([pad, proj_frames], dim=0)
            elif gen_frames.shape[0] != proj_frames.shape[0]:
                min_t = min(gen_frames.shape[0], proj_frames.shape[0])
                gen_frames = gen_frames[:min_t]
                proj_frames = proj_frames[:min_t]

            concat_frames = torch.cat([gen_frames, proj_frames], dim=2)
            iter_concat_path = os.path.join(
                output_paths.video_dir,
                f"iteration_{iter_idx + 1:02d}_concat_frames{output_start}-{output_end - 1}.mp4",
            )
            save_video_h264(iter_concat_path, concat_frames, fps=options.fps)
            print(f"  Saved concatenated video: {iter_concat_path} ({concat_frames.shape[0]} frames)")

    # Collect for final full video (must be outside output_paths guard
    # so validation with output_paths=None still populates the list)
    if scene_proj_decoded is not None:
        state.all_used_scene_proj_frames.extend([f.numpy() for f in scene_proj_decoded])

    scene_proj_frames = _render_iteration_scene_projection(
        iter_idx=iter_idx,
        target_frame_indices=target_frame_indices,
        state=state,
        pipeline=pipeline,
        output_size=output_size,
        options=options,
    )

    # if only we later iters...
    if iter_idx < len(iteration_plan) - 1:
        _update_frame_visibility(
            iter_idx=iter_idx,
            target_frame_indices=target_frame_indices,
            output_start=output_start,
            state=state,
            output_size=output_size,
        )

        last_frame_np = generated_video[-1].numpy()
        state.current_first_frame = Image.fromarray(last_frame_np)

    return IterationResult(
        generated_video=generated_video,
        frames_to_store=frames_to_store,
        scene_proj_frames=scene_proj_frames,
        updated_points_world=state.points_world,
        updated_colors=state.colors,
        iter_idx=iter_idx,
        output_start=output_start,
        output_end=output_end,
        model_frames=model_frames,
    )


def _update_point_cloud_with_iteration(
    iter_idx: int,
    generated_video: torch.Tensor,
    target_frame_indices: list[int],
    state: IterationState,
    options: BackboneInferenceOptions,
    output_size: tuple[int, int],
    pointcloud_dir: str | None,
    device: torch.device,
) -> None:
    """Update point cloud with generated frames."""
    height, width = output_size

    print("\n[Updating point cloud with generated frames...]")

    this_iter_frames = generated_video.detach().cpu().numpy()

    update_frame_indices = target_frame_indices
    if (
        target_frame_indices
        and state.accumulated_anchor_global_indices
        and target_frame_indices[0] in state.accumulated_anchor_global_indices
    ):
        print(f"  Skipping overlap frame at global idx {target_frame_indices[0]}")
        update_frame_indices = target_frame_indices[1:]
        this_iter_frames = this_iter_frames[1:]

    if len(update_frame_indices) == 0:
        print("  No new frames to update point cloud (all are overlaps)")
        return

    try:
        new_points, new_colors = state.pointcloud_updater.update(
            iter_idx=iter_idx,
            frames=this_iter_frames,
            frame_indices=update_frame_indices,
            state_points=state.points_world,
            state_colors=state.colors,
            options=options,
            debug_dir=pointcloud_dir,
        )
        if options.cpu_offload and hasattr(state.pointcloud_updater, "offload_to_cpu"):
            state.pointcloud_updater.offload_to_cpu()
        if new_points is not None and len(new_points) > 0:
            # Sync authoritative state from handler
            handler = state.pointcloud_updater
            if handler.points_world is not None:
                state.points_world = handler.points_world
                state.colors = handler.colors
            else:
                # Fallback: merge manually
                merge_voxel = (
                    options.stream3r_merge_voxel_size
                    if options.pointcloud_backend == "stream3r"
                    else (options.voxel_size or 0.001)
                )
                state.points_world, state.colors = _merge_pointcloud_incremental(
                    state.points_world,
                    state.colors,
                    new_points,
                    new_colors,
                    merge_voxel,
                )
            print(f"  [PointCloud] Update: {len(state.points_world)} points")

            if state.points_world is not None and state.colors is not None and pointcloud_dir:
                ply_path = os.path.join(pointcloud_dir, f"iteration_{iter_idx + 1:02d}_pointcloud.ply")
                save_point_cloud_ply(ply_path, state.points_world, state.colors)

        for idx in update_frame_indices:
            if idx not in state.accumulated_anchor_global_indices:
                state.accumulated_anchor_global_indices.append(idx)
    except Exception as exc:
        print(f"  Warning: {options.pointcloud_backend} update failed: {exc}")
        traceback.print_exc()
        print("  Continuing with existing point cloud...")


def _update_frame_visibility(
    iter_idx: int,
    target_frame_indices: list[int],
    output_start: int,
    state: IterationState,
    output_size: tuple[int, int],
) -> None:
    """Update visibility data for reference frame retrieval."""
    if state.points_world is None or state.poses_c2w is None:
        return

    print("\n[Computing visible points for reference frame retrieval...]")
    if state.intrinsics.ndim == 3:
        intr = state.intrinsics
    else:
        intr = np.tile(state.intrinsics[None], (len(state.poses_c2w), 1, 1))

    frames_to_process = []
    for local_idx, frame_idx in enumerate(target_frame_indices):
        if frame_idx < len(state.poses_c2w):
            if iter_idx == 0:
                global_frame_idx = local_idx
            else:
                if local_idx == 0:
                    continue
                global_frame_idx = output_start + local_idx - 1
            frames_to_process.append((local_idx, frame_idx, global_frame_idx))

    for _, frame_idx, global_frame_idx in tqdm(frames_to_process, desc="  Computing frame visibility", leave=False):
        intr_idx = _safe_frame_index(frame_idx, len(intr))
        visible_pts, _ = get_visible_points_for_frame(
            state.points_world,
            state.colors,
            state.poses_c2w[frame_idx],
            intr[intr_idx],
            output_size,
        )
        if len(visible_pts) > 0:
            state.frame_visible_points[global_frame_idx] = visible_pts

    print(
        f"  Updated visible points for {len(frames_to_process)} frames, total tracked: {len(state.frame_visible_points)}"
    )


def _render_iteration_scene_projection(
    iter_idx: int,
    target_frame_indices: list[int],
    state: IterationState,
    pipeline: UnifiedBackbonePipeline,
    output_size: tuple[int, int],
    options: BackboneInferenceOptions,
) -> list[np.ndarray] | None:
    """Render scene projections for visualization."""
    if state.points_world is None or state.colors is None:
        return None

    print("\n[Rendering scene projection for visualization...]")

    proj_frame_indices = target_frame_indices if iter_idx == 0 else target_frame_indices[1:]

    if len(proj_frame_indices) == 0:
        return None

    if options.cpu_offload:
        pipeline.vae.model.to(pipeline.device)
        pipeline.vae.mean = pipeline.vae.mean.to(pipeline.device)
        pipeline.vae.std = pipeline.vae.std.to(pipeline.device)

    handler = state.pointcloud_updater
    iter_scene_proj = handler.render_scene_projection(
        target_frame_indices=proj_frame_indices,
        output_size=output_size,
        vae=pipeline.vae,
        device=pipeline.device,
        dtype=pipeline.dtype,
        density_max_pixels=state.projection_density_max_pixels if options.limit_projection_density else None,
        density_rng=state.projection_density_rng if options.limit_projection_density else None,
        density_blue_noise=state.projection_density_blue_noise if options.limit_projection_density else None,
    )

    iter_scene_proj_decode = iter_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
    iter_scene_proj_video = pipeline.vae.decode_to_pixel(iter_scene_proj_decode, use_cache=False)
    iter_scene_proj_video = (iter_scene_proj_video + 1.0) / 2.0
    iter_scene_proj_video = iter_scene_proj_video.clamp(0, 1)
    iter_scene_proj_frames = iter_scene_proj_video[0]
    iter_scene_proj_np = (iter_scene_proj_frames * 255).byte().cpu()
    iter_scene_proj_np = rearrange(iter_scene_proj_np, "t c h w -> t h w c").numpy()

    if options.cpu_offload:
        pipeline.vae.model.to("cpu")
        pipeline.vae.mean = pipeline.vae.mean.to("cpu")
        pipeline.vae.std = pipeline.vae.std.to("cpu")
        torch.cuda.empty_cache()

    print(f"  Rendered {len(proj_frame_indices)} scene projection frames")

    return list(iter_scene_proj_np)


# =============================================================================
# High-level Iterative Inference
# =============================================================================


def run_iterative_inference(
    pipeline: UnifiedBackbonePipeline,
    first_frame: Image.Image,
    prompt: str,
    num_frames: int,
    frames_per_iter: int | None,
    options: BackboneInferenceOptions,
    first_frame_image: str,
    geometry_poses_c2w: np.ndarray,
    target_scene_proj: torch.Tensor | None = None,
    output_paths: BackboneOutputPaths | None = None,
    iter_inputs: dict[int, dict] | None = None,
) -> BackboneInferenceResult:
    """Run the full iterative LiveWorld inference loop.

    All data must be loaded before calling this function. This function only
    performs inference, not data loading.

    Args:
        pipeline: UnifiedBackbonePipeline instance
        first_frame: First frame image (required, pre-loaded)
        prompt: Text prompt (required, pre-loaded)
        num_frames: Total number of frames to generate
        frames_per_iter: Frames per iteration
        options: Inference options
        first_frame_image: Path to the first-frame image file
        geometry_poses_c2w: Camera extrinsics from geometry.npz (required, pre-loaded)
        target_scene_proj: Pre-computed scene projection (optional, if None will be generated)
        output_paths: Output paths for saving intermediate results
    """
    iteration_plan = compute_iteration_plan(num_frames, frames_per_iter)

    height = pipeline.h_pixel
    width = pipeline.w_pixel
    output_size = (height, width)

    # Propagate target resolution to options so updaters use it
    options.target_hw = (height, width)

    points_world = None
    colors = None
    poses_c2w = None
    intrinsics = None
    intrinsics_size = None
    anchor_frame0 = None
    initial_points_world = None
    initial_colors = None
    initial_dynamic_mask = None
    initial_fg_only_mask = None

    # Create 3D handler early so it handles everything
    handler = create_pointcloud_updater(
        backend=options.pointcloud_backend,
        device=pipeline.device,
    )
    print(f"[PointCloud] Using backend: {options.pointcloud_backend}")

    # Configure dynamic object detection for point cloud updates
    if hasattr(handler, "set_dynamic_models"):
        handler.set_dynamic_models(
            qwen_model_path=options.qwen_model_path,
            sam3_model_path=options.sam3_model_path,
        )

    if first_frame_image:
        print("Generating scene projection with handler + Qwen + SAM3...")

        first_iter_frames = iteration_plan[0][2]
        target_frame_indices = list(range(first_iter_frames))

        print(f"  Using extrinsics from geometry.npz (shape: {geometry_poses_c2w.shape})")

        result = generate_scene_projection_with_handler(
            first_frame_image=first_frame_image,
            target_frames=target_frame_indices,
            output_size=output_size,
            handler=handler,
            geometry_poses_c2w=geometry_poses_c2w,
            vae=pipeline.vae,
            device=pipeline.device,
            dtype=pipeline.dtype,
            options=options,
        )

        (
            target_scene_proj,
            points_world,
            colors,
            anchor_frame0,
            intrinsics,
            alignment_transform,
            intrinsics_size,
            initial_dynamic_mask,
            initial_fg_only_mask,
        ) = result

        # Transform geometry poses into Stream3R world coordinate system
        if alignment_transform is not None:
            poses_c2w = np.array(
                [
                    alignment_transform @ p.astype(np.float32)
                    if p.shape == (4, 4)
                    else alignment_transform @ np.vstack([p.astype(np.float32), [0, 0, 0, 1]])
                    for p in geometry_poses_c2w
                ],
                dtype=np.float32,
            )
            print(f"  Aligned geometry poses to Stream3R coords: {poses_c2w.shape}")
        else:
            poses_c2w = geometry_poses_c2w.copy()
            print(f"  Using full geometry poses (no alignment): {poses_c2w.shape}")

        if points_world is not None:
            initial_points_world = points_world.copy()
        if colors is not None:
            initial_colors = colors.copy()

        if options.cpu_offload:
            pipeline.vae.model.to("cpu")
            pipeline.vae.mean = pipeline.vae.mean.to("cpu")
            pipeline.vae.std = pipeline.vae.std.to("cpu")
            torch.cuda.empty_cache()

    elif target_scene_proj is not None:
        target_scene_proj = target_scene_proj.to(device=pipeline.device, dtype=pipeline.dtype)
        if target_scene_proj.dim() == 4 and target_scene_proj.shape[1] == 16:
            target_scene_proj = target_scene_proj.permute(1, 0, 2, 3)
    else:
        num_latent_frames = pipeline._pixel_to_latent_frames(num_frames)
        target_scene_proj = torch.zeros(
            16, num_latent_frames, pipeline.h, pipeline.w, device=pipeline.device, dtype=pipeline.dtype
        )

    if output_paths is not None:
        os.makedirs(output_paths.output_dir, exist_ok=True)
        os.makedirs(output_paths.pointcloud_dir, exist_ok=True)
        os.makedirs(output_paths.video_dir, exist_ok=True)

        # Save initial point cloud immediately after generation
        if initial_points_world is not None and initial_colors is not None:
            initial_ply_path = os.path.join(output_paths.pointcloud_dir, "iteration_00_initial_pointcloud.ply")
            save_point_cloud_ply(initial_ply_path, initial_points_world, initial_colors)
            print(f"\nInitial point cloud saved: {initial_ply_path}")
            print(f"  Points: {len(initial_points_world)}")

    state = IterationState(
        all_generated_frames=[],
        current_first_frame=first_frame,
        all_scene_proj_frames=[],
        all_used_scene_proj_frames=[],
        points_world=points_world,
        colors=colors,
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        intrinsics_size=intrinsics_size,
        target_scene_proj=target_scene_proj,
        initial_dynamic_mask=initial_dynamic_mask,
        initial_fg_only_mask=initial_fg_only_mask,
        projection_density_rng=np.random.default_rng(options.seed),
    )

    # Attach handler to state (already created above)
    state.pointcloud_updater = handler

    if anchor_frame0 is not None:
        init_frame = cv2.resize(anchor_frame0, (width, height), interpolation=cv2.INTER_LINEAR)
        init_intrinsics = intrinsics[0:1].copy() if intrinsics.ndim == 3 else intrinsics[None].copy()

        state.accumulated_anchor_frames = init_frame[None]
        state.accumulated_anchor_poses = poses_c2w[0:1]
        state.accumulated_anchor_intrinsics = scale_intrinsics_batch(
            init_intrinsics,
            (intrinsics_size[0], intrinsics_size[1]) if intrinsics_size else (height, width),
            (height, width),
        )
        state.accumulated_anchor_global_indices = [0]

        if intrinsics_size is not None:
            print(
                f"Initialized anchor with first frame (intrinsics {intrinsics_size[0]}x{intrinsics_size[1]} -> "
                f"{height}x{width})"
            )

    iter_inputs = iter_inputs or {}
    for iter_idx in range(len(iteration_plan)):  # iterative video generation process
        # Reset all random states for deterministic per-iteration behavior
        set_seed(options.seed + iter_idx)
        state.projection_density_rng = np.random.default_rng(options.seed + iter_idx)

        iter_input = iter_inputs.get(iter_idx)
        result = process_iteration(
            pipeline=pipeline,
            iter_idx=iter_idx,
            iteration_plan=iteration_plan,
            state=state,
            prompt=prompt,
            iter_input=iter_input,
            options=options,
            t0=0,
            output_size=output_size,
            output_paths=output_paths,
        )

        state.all_generated_frames.extend(result.frames_to_store)
        if result.scene_proj_frames is not None:
            state.all_scene_proj_frames.extend(result.scene_proj_frames)

    final_video = torch.from_numpy(np.stack(state.all_generated_frames, axis=0))

    return BackboneInferenceResult(
        state=state,
        iteration_plan=iteration_plan,
        generated_video=final_video,
        output_size=output_size,
        start_frame=0,
        num_frames=num_frames,
        first_frame=first_frame,
        initial_points_world=initial_points_world,
        initial_colors=initial_colors,
    )


def save_iterative_outputs(
    pipeline: UnifiedBackbonePipeline,
    result: BackboneInferenceResult,
    output_paths: BackboneOutputPaths,
    options: BackboneInferenceOptions,
    output_name: str | None = None,
    input_dir: str | None = None,
) -> None:
    """Save outputs for iterative inference."""

    state = result.state

    generated_video_final = result.generated_video
    num_gen_frames = generated_video_final.shape[0]

    if output_name is None:
        output_name = os.path.basename(output_paths.output_dir.rstrip("/"))

    if state.all_scene_proj_frames:
        scene_proj_video_save = torch.from_numpy(np.stack(state.all_scene_proj_frames, axis=0))
        if scene_proj_video_save.shape[0] > num_gen_frames:
            scene_proj_video_save = scene_proj_video_save[:num_gen_frames]
        elif scene_proj_video_save.shape[0] < num_gen_frames:
            pad = num_gen_frames - scene_proj_video_save.shape[0]
            scene_proj_video_save = torch.cat(
                [scene_proj_video_save, scene_proj_video_save[-1:].repeat(pad, 1, 1, 1)], dim=0
            )
    else:
        if result.initial_points_world is not None and state.poses_c2w is not None:
            intr_size = state.intrinsics_size
            all_target_indices = list(range(result.start_frame, result.start_frame + result.num_frames))

            if options.cpu_offload:
                pipeline.vae.model.to(pipeline.device)
                pipeline.vae.mean = pipeline.vae.mean.to(pipeline.device)
                pipeline.vae.std = pipeline.vae.std.to(pipeline.device)

            full_scene_proj = generate_scene_projection_from_pointcloud(
                points_world=result.initial_points_world,
                colors=result.initial_colors,
                target_frames=all_target_indices,
                poses_c2w=state.poses_c2w,
                intrinsics=state.intrinsics,
                output_size=result.output_size,
                intrinsics_size=intr_size,
                vae=pipeline.vae,
                device=pipeline.device,
                dtype=pipeline.dtype,
            )
            scene_proj_for_decode = full_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
            scene_proj_video = pipeline.vae.decode_to_pixel(scene_proj_for_decode, use_cache=False)
            scene_proj_video = (scene_proj_video + 1.0) / 2.0
            scene_proj_video = scene_proj_video.clamp(0, 1)
            scene_proj_video_tensor = scene_proj_video[0]
            scene_proj_video_save = (scene_proj_video_tensor * 255).byte().cpu()
            scene_proj_video_save = rearrange(scene_proj_video_save, "t c h w -> t h w c")

            if options.cpu_offload:
                pipeline.vae.model.to("cpu")
                pipeline.vae.mean = pipeline.vae.mean.to("cpu")
                pipeline.vae.std = pipeline.vae.std.to("cpu")
                torch.cuda.empty_cache()
        else:
            scene_proj_video_save = torch.zeros_like(generated_video_final)

    if result.initial_points_world is not None and state.poses_c2w is not None:
        intr_size = state.intrinsics_size
        all_target_indices = list(range(result.start_frame, result.start_frame + result.num_frames))

        if options.cpu_offload:
            pipeline.vae.model.to(pipeline.device)
            pipeline.vae.mean = pipeline.vae.mean.to(pipeline.device)
            pipeline.vae.std = pipeline.vae.std.to(pipeline.device)

        initial_scene_proj = generate_scene_projection_from_pointcloud(
            points_world=result.initial_points_world,
            colors=result.initial_colors,
            target_frames=all_target_indices,
            poses_c2w=state.poses_c2w,
            intrinsics=state.intrinsics,
            output_size=result.output_size,
            intrinsics_size=intr_size,
            vae=pipeline.vae,
            device=pipeline.device,
            dtype=pipeline.dtype,
        )
        initial_proj_for_decode = initial_scene_proj.permute(1, 0, 2, 3).unsqueeze(0)
        initial_proj_video = pipeline.vae.decode_to_pixel(initial_proj_for_decode, use_cache=False)
        initial_proj_video = (initial_proj_video + 1.0) / 2.0
        initial_proj_video = initial_proj_video.clamp(0, 1)
        initial_proj_tensor = initial_proj_video[0]
        initial_proj_video_save = (initial_proj_tensor * 255).byte().cpu()
        initial_proj_video_save = rearrange(initial_proj_video_save, "t c h w -> t h w c")

        if initial_proj_video_save.shape[0] > num_gen_frames:
            initial_proj_video_save = initial_proj_video_save[:num_gen_frames]
        elif initial_proj_video_save.shape[0] < num_gen_frames:
            pad = num_gen_frames - initial_proj_video_save.shape[0]
            initial_proj_video_save = torch.cat(
                [initial_proj_video_save, initial_proj_video_save[-1:].repeat(pad, 1, 1, 1)], dim=0
            )

        if options.cpu_offload:
            pipeline.vae.model.to("cpu")
            pipeline.vae.mean = pipeline.vae.mean.to("cpu")
            pipeline.vae.std = pipeline.vae.std.to("cpu")
            torch.cuda.empty_cache()
    else:
        initial_proj_video_save = scene_proj_video_save

    if state.all_used_scene_proj_frames:
        used_scene_proj_video = torch.from_numpy(np.stack(state.all_used_scene_proj_frames, axis=0))
        if used_scene_proj_video.shape[0] > num_gen_frames:
            used_scene_proj_video = used_scene_proj_video[:num_gen_frames]
        elif used_scene_proj_video.shape[0] < num_gen_frames:
            pad = num_gen_frames - used_scene_proj_video.shape[0]
            used_scene_proj_video = torch.cat(
                [used_scene_proj_video, used_scene_proj_video[-1:].repeat(pad, 1, 1, 1)], dim=0
            )
    else:
        used_scene_proj_video = scene_proj_video_save

    comparison_video = torch.cat([generated_video_final, scene_proj_video_save], dim=2)

    output_path = os.path.join(output_paths.video_dir, "generated_full.mp4")
    save_video_h264(output_path, generated_video_final, fps=options.fps)

    comparison_path = os.path.join(output_paths.video_dir, "comparison_full.mp4")
    save_video_h264(comparison_path, comparison_video, fps=options.fps)

    scene_proj_path = os.path.join(output_paths.video_dir, "scene_projection_full.mp4")
    save_video_h264(scene_proj_path, scene_proj_video_save, fps=options.fps)

    used_scene_proj_path = os.path.join(output_paths.video_dir, "scene_projection_used_full.mp4")
    save_video_h264(used_scene_proj_path, used_scene_proj_video, fps=options.fps)

    initial_proj_path = os.path.join(output_paths.video_dir, "initial_projection_full.mp4")
    save_video_h264(initial_proj_path, initial_proj_video_save, fps=options.fps)

    if result.first_frame is not None:
        first_frame_output = os.path.join(output_paths.output_dir, "first_frame.png")
        result.first_frame.save(first_frame_output)

    if result.initial_points_world is not None and result.initial_colors is not None:
        initial_ply_path = os.path.join(output_paths.pointcloud_dir, "iteration_00_initial_pointcloud.ply")
        save_point_cloud_ply(initial_ply_path, result.initial_points_world, result.initial_colors)

    if state.points_world is not None and state.colors is not None:
        final_ply_path = os.path.join(output_paths.pointcloud_dir, "final_pointcloud.ply")
        save_point_cloud_ply(final_ply_path, state.points_world, state.colors)

    metadata = {
        "output_name": output_name,
        "num_frames": result.num_frames,
        "iterations": len(result.iteration_plan),
        "iteration_plan": [(start, end, frames) for start, end, frames in result.iteration_plan],
        "start_frame": result.start_frame,
        "fps": options.fps,
        "infer_steps": options.infer_steps,
        "use_few_step": options.use_few_step,
        "denoising_step_list": options.denoising_step_list,
        "seed": options.seed,
        "voxel_size": options.voxel_size,
        "resolution": f"{generated_video_final.shape[2]}x{generated_video_final.shape[1]}",
        "input_dir": input_dir,
    }
    metadata_path = os.path.join(output_paths.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
