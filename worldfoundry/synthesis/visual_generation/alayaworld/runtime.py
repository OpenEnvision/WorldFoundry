"""Inference-only AlayaWorld rollout built on WorldFoundry's canonical LTX-2.3 stack."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
    load_ltx_video_decoder_checkpoint,
    load_ltx_video_encoder_checkpoint,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import (
    LTXVideoEmbeddingProcessorModule,
    load_ltx_embedding_processor_module,
    load_ltx_gemma_module,
)
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.tokenizer import LTXVGemmaTokenizer
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.representations.ltx.patchifiers import (
    VideoLatentPatchifier,
    get_pixel_coords,
)
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import (
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.core.geometry.trajectory import (
    camera_poses_to_adaln_actions,
    camera_trajectory_view_matrices,
    parse_camera_trajectory,
    select_adaln_actions,
)
from worldfoundry.runtime.compile_cache import configure_persistent_compile_cache
from worldfoundry.synthesis.visual_generation.alayaworld.history_encoder import (
    AlayaHistoryEncoder,
    load_alaya_history_encoder,
)
from worldfoundry.synthesis.visual_generation.alayaworld.ltx_compiling import (
    CompilationConfig,
    compile_transformer,
)
from worldfoundry.synthesis.visual_generation.alayaworld.model import (
    AlayaConditioning,
    AlayaPrefix,
    AlayaWorldModel,
    load_alaya_world_model,
)

from .spatial import AlayaSpatialMemory, SpatialBank

_LTX_SCALE_FACTORS = SpatioTemporalScaleFactors(time=8, height=32, width=32)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dtype_from_name(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {value!r}; choose bf16, fp16, or fp32") from exc


def _fit_image(path: str | Path, *, height: int, width: int) -> torch.Tensor:
    """Load, cover-resize and center-crop an image to ``[3,H,W]`` in [-1, 1]."""

    image = Image.open(Path(path).expanduser()).convert("RGB")
    source_width, source_height = image.size
    scale = max(float(height) / source_height, float(width) / source_width)
    resized = image.resize(
        (max(width, round(source_width * scale)), max(height, round(source_height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    resized = resized.crop((left, top, left + width, top + height))
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _pad_or_trim_frames(value: torch.Tensor, frames: int, *, dim: int) -> torch.Tensor:
    if value.shape[dim] >= frames:
        return value.narrow(dim, 0, frames).contiguous()
    tail = value.select(dim, value.shape[dim] - 1).unsqueeze(dim)
    expand_shape = list(tail.shape)
    expand_shape[dim] = frames - value.shape[dim]
    return torch.cat((value, tail.expand(*expand_shape)), dim=dim).contiguous()


class AlayaWorldRuntime:
    """Autoregressive AlayaWorld I2V runtime.

    The class owns only Alaya's rollout policy.  Model construction, transformer
    blocks, RoPE, attention dispatch, Gemma, the text connector and both VAE
    halves are the existing in-tree LTX-2.3 implementations.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        da3_path: str | None = None,
        *,
        height: int = 544,
        width: int = 960,
        fps: float = 24.0,
        rounds: int = 5,
        num_frames: int | None = None,
        seed: int = 42,
        dtype: str | torch.dtype = "bf16",
        device: str | torch.device | None = None,
        camera_path: str | None = None,
        camera_trajectory: str = "w*296",
        camera_translation_step: float = 0.08,
        camera_rotation_step_degrees: float = 0.375,
        intrinsic: Sequence[float] = (0.4482, 0.7966, 0.5, 0.5),
        action_scale: str = "0.1337,0.2769,2.021,0.00442,0.01026,0.00448",
        action_freq_scale: float = 1000.0,
        history_latent_frames: int = 16,
        condition_latent_frames: int = 1,
        output_latent_frames: int = 4,
        sampling_steps: int = 4,
        action_history_memory: bool = False,
        spatial_enabled: bool = True,
        depth_backend: str = "da3",
        spatial_num_context_frames: int = 10,
        spatial_retrieval_views: int = 1,
        spatial_downsample: int = 4,
        spatial_maximum_coverage: bool = True,
        spatial_retrieval_depth_threshold: float = 0.1,
        spatial_constant_depth: float = 1.0,
        spatial_include_sink: bool = False,
        spatial_require_full_context: bool = True,
        da3_process_res: int = 504,
        da3_process_res_method: str = "upper_bound_resize",
        da3_align_to_input_scale: bool = True,
        history_compress_t: int = 1,
        history_compress_h: int = 2,
        history_compress_w: int = 2,
        history_lr_compress_t: int = 1,
        history_lr_compress_h: int = 2,
        history_lr_compress_w: int = 2,
        history_gate_init: float = 0.5,
        history_self_attention: bool = True,
        history_lr_branch: bool = True,
        vae_spatial_tile_size: int = 768,
        vae_spatial_overlap: int = 64,
        vae_temporal_tile_size: int = 80,
        vae_temporal_overlap: int = 24,
        vae_decode_chunk_latents: int = 8,
        vae_decode_overlap_latents: int = 6,
        context_parallel: bool = True,
        decode_rank0_only: bool = True,
        compile_mode: str | None = "reduce-overhead",
        compile_backend: str = "inductor",
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = False,
        **_: Any,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.gemma_root = str(Path(gemma_root).expanduser())
        self.da3_path = str(Path(da3_path).expanduser()) if da3_path else None
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.requested_frames = int(num_frames) if num_frames is not None else None
        requested_rounds = int(rounds)
        self.seed = int(seed)
        self.dtype = _dtype_from_name(dtype)
        self.camera_path = str(Path(camera_path).expanduser()) if camera_path else None
        self.camera_trajectory = str(camera_trajectory)
        self.camera_translation_step = float(camera_translation_step)
        self.camera_rotation_step_degrees = float(camera_rotation_step_degrees)
        self.intrinsic = tuple(float(value) for value in intrinsic)
        self.action_scale = str(action_scale)
        self.action_freq_scale = float(action_freq_scale)
        self.history_frames = int(history_latent_frames)
        self.condition_frames = int(condition_latent_frames)
        self.output_frames = int(output_latent_frames)
        if self.output_frames <= 0:
            raise ValueError("output_latent_frames must be positive")
        chunk_pixel_frames = self.output_frames * 8
        self.rounds = (
            max(1, (self.requested_frames + chunk_pixel_frames - 1) // chunk_pixel_frames)
            if self.requested_frames is not None
            else requested_rounds
        )
        self.sampling_steps = int(sampling_steps)
        self.action_history_memory = bool(action_history_memory)
        self.spatial_enabled = bool(spatial_enabled)
        self.depth_backend = str(depth_backend).lower()
        self.spatial_options = {
            "num_context_frames": int(spatial_num_context_frames),
            "retrieval_views": int(spatial_retrieval_views),
            "downsample": int(spatial_downsample),
            "maximum_coverage": bool(spatial_maximum_coverage),
            "retrieval_depth_threshold": float(spatial_retrieval_depth_threshold),
            "constant_depth": float(spatial_constant_depth),
            "include_sink": bool(spatial_include_sink),
            "require_full_context": bool(spatial_require_full_context),
            "da3_process_res": int(da3_process_res),
            "da3_process_res_method": str(da3_process_res_method),
            "da3_align_to_input_scale": bool(da3_align_to_input_scale),
        }
        self.history_options = {
            "compress_t": int(history_compress_t),
            "compress_h": int(history_compress_h),
            "compress_w": int(history_compress_w),
            "lr_compress_t": int(history_lr_compress_t),
            "lr_compress_h": int(history_lr_compress_h),
            "lr_compress_w": int(history_lr_compress_w),
            "gate_init": float(history_gate_init),
            "use_self_attn": bool(history_self_attention),
            "use_lr_branch": bool(history_lr_branch),
        }
        self.vae_tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(
                tile_size_in_pixels=int(vae_spatial_tile_size),
                tile_overlap_in_pixels=int(vae_spatial_overlap),
            ),
            temporal_config=TemporalTilingConfig(
                tile_size_in_frames=int(vae_temporal_tile_size),
                tile_overlap_in_frames=int(vae_temporal_overlap),
            ),
        )
        self.decode_chunk = max(1, int(vae_decode_chunk_latents))
        self.decode_overlap = max(1, int(vae_decode_overlap_latents))
        self.context_parallel = bool(context_parallel)
        self.decode_rank0_only = bool(decode_rank0_only)
        normalized_compile_mode = "none" if compile_mode is None else str(compile_mode).strip().lower()
        if normalized_compile_mode not in {"none", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("compile_mode must be one of none, default, reduce-overhead, or max-autotune")
        self.compile_mode = normalized_compile_mode
        self.compile_backend = str(compile_backend)
        self.compile_fullgraph = bool(compile_fullgraph)
        self.compile_dynamic = compile_dynamic
        self._owns_process_group = False

        if self.height % 32 or self.width % 32:
            raise ValueError(f"AlayaWorld height/width must be divisible by 32, got {self.height}x{self.width}")
        if self.requested_frames is not None and self.requested_frames <= 0:
            raise ValueError("num_frames must be positive when provided")
        if self.rounds <= 0 or self.history_frames <= 0 or self.condition_frames <= 0:
            raise ValueError("rounds, history_latent_frames and condition_latent_frames must be positive")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if self.condition_frames > self.history_frames:
            raise ValueError("condition_latent_frames cannot exceed history_latent_frames")
        if len(self.intrinsic) != 4:
            raise ValueError("intrinsic must be [fx, fy, cx, cy]")
        if self.spatial_enabled and self.depth_backend == "da3" and not self.da3_path:
            raise ValueError("da3_path is required when spatial_enabled=true and depth_backend='da3'")

        self.device = self._setup_device(device)

    def _setup_device(self, requested: str | torch.device | None) -> torch.device:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and not dist.is_initialized():
            if not torch.cuda.is_available():
                raise RuntimeError("AlayaWorld context parallelism requires CUDA")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            self._owns_process_group = True
        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        if requested is not None:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _check_assets(self) -> None:
        for label, path in (("AlayaWorld checkpoint", self.checkpoint_path), ("Gemma root", self.gemma_root)):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.spatial_enabled and self.depth_backend == "da3" and not Path(self.da3_path).exists():
            raise FileNotFoundError(f"DA3 checkpoint does not exist: {self.da3_path}")

    def _load_camera(self, required_frames: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.camera_path:
            metadata = torch.load(self.camera_path, map_location="cpu", weights_only=False)
            if not isinstance(metadata, dict) or "cam_c2w" not in metadata:
                raise ValueError(f"camera file must contain a cam_c2w tensor: {self.camera_path}")
            camera = torch.as_tensor(metadata["cam_c2w"], dtype=torch.float32)
            intrinsic_value = metadata.get("intrinsic")
            if intrinsic_value is None:
                fx, fy, cx, cy = self.intrinsic
                intrinsic_value = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            intrinsic = torch.as_tensor(intrinsic_value, dtype=torch.float32)
            if camera.ndim == 3:
                camera = camera.unsqueeze(0)
            if camera.ndim != 4 or camera.shape[-2:] != (4, 4):
                raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {tuple(camera.shape)}")
            if intrinsic.ndim == 2:
                intrinsic = intrinsic.unsqueeze(0)
            if intrinsic.ndim not in (3, 4) or intrinsic.shape[-2:] != (3, 3):
                raise ValueError(f"intrinsic must be [3,3], [B,3,3], or [B,F,3,3], got {tuple(intrinsic.shape)}")
            if intrinsic.ndim == 3 and camera.shape[0] == 1 and intrinsic.shape[0] == camera.shape[1]:
                intrinsic = intrinsic.unsqueeze(0)
            if intrinsic.shape[0] not in (1, camera.shape[0]):
                raise ValueError(f"intrinsic batch {intrinsic.shape[0]} does not match camera batch {camera.shape[0]}")
            available_latents = (camera.shape[1] - 1) // 8 + 1
            # Official rollout planning keeps one latent beyond the final
            # generated chunk.  This look-ahead preserves the exact camera/VAE
            # window contract used to prepare Alaya inference cases.
            maximum_rounds = (available_latents - 1 - (1 + self.history_frames)) // self.output_frames
            rounds = min(self.rounds, int(maximum_rounds))
            if rounds < 1:
                raise ValueError(
                    f"camera trajectory has {camera.shape[1]} frames, insufficient for one AlayaWorld chunk"
                )
            needed = (1 + self.history_frames + rounds * self.output_frames) * 8 + 1
            return camera[:, :needed].contiguous(), intrinsic, rounds

        motions = parse_camera_trajectory(
            self.camera_trajectory,
            translation_step=self.camera_translation_step,
            rotation_step_degrees=self.camera_rotation_step_degrees,
        )
        needed_motions = required_frames - 1
        if len(motions) < needed_motions:
            motions.extend(dict(motions[-1]) if motions else {} for _ in range(needed_motions - len(motions)))
        view = camera_trajectory_view_matrices(motions[:needed_motions])
        camera = torch.linalg.inv(torch.from_numpy(view)).unsqueeze(0).float()
        fx, fy, cx, cy = self.intrinsic
        intrinsic = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        return camera, intrinsic.unsqueeze(0), self.rounds

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        model_folder = Path(self.gemma_root)
        weight_paths = tuple(str(path) for path in sorted(model_folder.glob("model-*.safetensors")))
        if not weight_paths:
            raise FileNotFoundError(f"no Gemma safetensors shards found below {model_folder}")
        policy = self._prompt_runtime_policy(device=self.device, dtype=self.dtype)
        gemma = load_ltx_gemma_module(CheckpointSpec(source=weight_paths), policy)
        processor = load_ltx_embedding_processor_module(
            CheckpointSpec(source=self.checkpoint_path),
            policy,
            video_only=True,
        )
        if not isinstance(processor, LTXVideoEmbeddingProcessorModule):
            raise TypeError("Alaya prompt encoding requires the video-only LTX processor")
        tokenizer = LTXVGemmaTokenizer(self.gemma_root, max_length=1024)
        token_pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
        execution_device = next(tensor.device for tensor in gemma.parameters() if tensor.device.type != "meta")
        input_ids = torch.tensor(
            [[int(pair[0]) for pair in token_pairs]],
            device=execution_device,
        )
        attention_mask = torch.tensor(
            [[int(pair[1]) for pair in token_pairs]],
            device=execution_device,
        )
        encoded = gemma.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        output = processor.processor.process_hidden_states(
            encoded.hidden_states,
            attention_mask,
            padding_side="left",
        )
        context = output.video_encoding.detach().to(device="cpu", dtype=self.dtype)
        del gemma, processor, tokenizer, encoded, output, input_ids, attention_mask
        _cleanup_cuda()
        return context

    @staticmethod
    def _prompt_runtime_policy(*, device: torch.device, dtype: torch.dtype) -> RuntimePolicy:
        """Keep the 12B Gemma prompt encoder below the generation VRAM peak."""

        del device
        return RuntimePolicy(device=torch.device("cpu"), dtype=dtype)

    def _build_vae(self):
        checkpoint = CheckpointSpec(source=self.checkpoint_path)
        policy = RuntimePolicy(device=self.device, dtype=self.dtype)
        encoder = load_ltx_video_encoder_checkpoint(checkpoint, policy)
        decoder = load_ltx_video_decoder_checkpoint(checkpoint, policy)
        return encoder.encoder, decoder.decoder

    def _build_transformer(self) -> tuple[AlayaWorldModel, AlayaHistoryEncoder]:
        checkpoint = CheckpointSpec(source=self.checkpoint_path)
        policy = RuntimePolicy(device=self.device, dtype=self.dtype)
        transformer = load_alaya_world_model(checkpoint, policy)
        transformer.action_adaln_embedder.freq_scale = self.action_freq_scale
        history = load_alaya_history_encoder(
            checkpoint,
            transformer.patchify_proj,
            policy=policy,
            **self.history_options,
        )
        if self.context_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            transformer.enable_context_parallel(dist.group.WORLD)
        if self.compile_mode != "none":
            # CUDA graphs in reduce-overhead mode are a poor fit for Ulysses'
            # all-to-all collectives.  Match the official Alaya runtime and keep
            # Inductor fusion while disabling graph capture under CP.
            configure_persistent_compile_cache(namespace="alayaworld-ltx2")
            mode = self.compile_mode
            if transformer._context_parallel_group is not None and mode == "reduce-overhead":
                mode = "default"
            compile_transformer(
                transformer,
                CompilationConfig(
                    mode=None if mode == "default" else mode,
                    backend=self.compile_backend,
                    fullgraph=self.compile_fullgraph,
                    dynamic=self.compile_dynamic,
                ),
            )
        return transformer, history

    def _positions(
        self,
        *,
        frames: int,
        height: int,
        width: int,
        time_indices: torch.Tensor | Sequence[float] | None = None,
        bounds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bounds is None:
            shape = VideoLatentShape(batch=1, channels=128, frames=frames, height=height, width=width)
            bounds = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(shape, device=self.device).float()
            if time_indices is not None:
                values = torch.as_tensor(time_indices, device=self.device, dtype=bounds.dtype).flatten()
                if values.numel() != frames:
                    raise ValueError(f"expected {frames} time indices, got {values.numel()}")
                per_token = values[:, None, None].expand(frames, height, width).reshape(-1)
                bounds[:, 0] = torch.stack((per_token, per_token + 1), dim=-1).unsqueeze(0)
        else:
            bounds = bounds.to(device=self.device, dtype=torch.float32)
        positions = get_pixel_coords(bounds, scale_factors=_LTX_SCALE_FACTORS, causal_fix=True).float()
        positions[:, 0] /= self.fps
        return positions.to(dtype=self.dtype)

    def _prefix_from_latent(
        self,
        latent: torch.Tensor,
        *,
        time_indices: Sequence[float] | torch.Tensor,
        actions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> AlayaPrefix:
        batch, _, frames, height, width = latent.shape
        if batch != 1:
            raise ValueError("AlayaWorld runtime currently supports batch size 1")
        patchifier = VideoLatentPatchifier(patch_size=1)
        return AlayaPrefix(
            latent=patchifier.patchify(latent),
            positions=self._positions(
                frames=frames,
                height=height,
                width=width,
                time_indices=time_indices,
            ),
            actions=actions,
            valid_mask=valid_mask,
        )

    def _target_modality(self, latent: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> Modality:
        batch, _, frames, height, width = latent.shape
        patchifier = VideoLatentPatchifier(patch_size=1)
        tokens = patchifier.patchify(latent)
        local_start = 1 + self.history_frames
        return Modality(
            latent=tokens,
            sigma=sigma.expand(batch),
            timesteps=torch.ones(batch, tokens.shape[1], 1, device=self.device, dtype=self.dtype) * sigma,
            positions=self._positions(
                frames=frames,
                height=height,
                width=width,
                time_indices=torch.arange(local_start, local_start + frames, device=self.device),
            ),
            context=context,
            context_mask=None,
        )

    @torch.inference_mode()
    def _denoise_chunk(
        self,
        *,
        transformer: AlayaWorldModel,
        history_encoder: AlayaHistoryEncoder,
        history: torch.Tensor,
        sink: torch.Tensor,
        context: torch.Tensor,
        full_actions: torch.Tensor,
        global_target_start: int,
        spatial_context: tuple[torch.Tensor, torch.Tensor] | None,
        generator: torch.Generator,
        history_action_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, channels, _, height, width = history.shape
        patchifier = VideoLatentPatchifier(patch_size=1)
        history_tokens, history_bounds = history_encoder(history)
        history_bounds = history_bounds.clone()
        history_bounds[:, 0] += 1
        history_actions = (
            select_adaln_actions(full_actions, history_action_indices, device=self.device, dtype=self.dtype)
            if history_action_indices is not None
            else None
        )
        history_prefix = AlayaPrefix(
            tokens=history_tokens,
            positions=self._positions(
                frames=self.history_frames,
                height=height // self.history_options["compress_h"],
                width=width // self.history_options["compress_w"],
                bounds=history_bounds,
            ),
            actions=history_actions,
        )
        sink_prefix = self._prefix_from_latent(sink, time_indices=(0,))
        nearby = history[:, :, -self.condition_frames :].contiguous()
        nearby_global = torch.arange(
            global_target_start - self.condition_frames,
            global_target_start,
            device=self.device,
        )
        nearby_actions = select_adaln_actions(
            full_actions,
            nearby_global,
            device=self.device,
            dtype=self.dtype,
        )
        nearby_start = 1 + self.history_frames - self.condition_frames
        nearby_prefix = self._prefix_from_latent(
            nearby,
            time_indices=torch.arange(nearby_start, nearby_start + self.condition_frames, device=self.device),
            actions=nearby_actions,
        )
        spatial_prefix = None
        if spatial_context is not None:
            spatial_latent, spatial_valid = spatial_context
            spatial_prefix = self._prefix_from_latent(
                spatial_latent,
                time_indices=torch.arange(
                    1 + self.history_frames,
                    1 + self.history_frames + self.output_frames,
                    device=self.device,
                ),
                valid_mask=spatial_valid,
            )
        target_indices = torch.arange(
            global_target_start,
            global_target_start + self.output_frames,
            device=self.device,
        )
        target_actions = select_adaln_actions(
            full_actions,
            target_indices,
            device=self.device,
            dtype=self.dtype,
        )
        conditioning = AlayaConditioning(
            sink=sink_prefix,
            history=history_prefix,
            spatial=spatial_prefix,
            nearby=nearby_prefix,
            target_actions=target_actions,
        )

        latent = torch.randn(
            batch,
            channels,
            self.output_frames,
            height,
            width,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        sigmas = torch.linspace(1.0, 0.0, self.sampling_steps + 1, device=self.device, dtype=torch.float32)
        for index in range(self.sampling_steps):
            sigma, next_sigma = sigmas[index], sigmas[index + 1]
            modality = self._target_modality(latent, sigma, context)
            velocity_tokens, _ = transformer(video=modality, conditioning=conditioning)
            velocity = patchifier.unpatchify(
                velocity_tokens,
                VideoLatentShape(
                    batch=batch,
                    channels=channels,
                    frames=self.output_frames,
                    height=height,
                    width=width,
                ),
            )
            if index + 1 < self.sampling_steps:
                latent = latent - (sigma - next_sigma).to(latent.dtype) * velocity
            else:
                latent = (latent.float() - velocity.float() * sigma).to(latent.dtype)
        return latent

    def _decode_video(self, decoder, latent: torch.Tensor) -> torch.Tensor:
        total = int(latent.shape[2])
        temporal_stride = 8
        frames: list[torch.Tensor] = []
        generator = torch.Generator(device=self.device).manual_seed(self.seed + 9_000_000)
        for start in range(0, total, self.decode_chunk):
            end = min(total, start + self.decode_chunk)
            context_start = max(0, start - self.decode_overlap)
            context_end = min(total, end + self.decode_overlap)
            pixels = decoder(
                latent[:, :, context_start:context_end].to(device=self.device, dtype=self.dtype),
                generator=generator,
            )
            local_start = start - context_start
            low = 0 if start == 0 else temporal_stride * (local_start - 1) + 1
            high = temporal_stride * (end - 1 - context_start) + 1
            tile = pixels[:, :, low:high].detach().float().cpu()
            frames.append(tile)
            del pixels
            _cleanup_cuda()
        video = torch.cat(frames, dim=2)[0].permute(1, 2, 3, 0).contiguous()
        return ((video * 0.5 + 0.5).clamp_(0, 1).mul_(255).round_()).to(torch.uint8)

    def _broadcast_decoded_frames(self, frames: torch.Tensor | None) -> torch.Tensor:
        """Broadcast rank-0 decoded uint8 frames without duplicating VAE work."""

        if not dist.is_initialized() or dist.get_world_size() <= 1:
            if frames is None:
                raise RuntimeError("decoded frames are missing on the single-rank path")
            return frames
        rank = dist.get_rank()
        shape = torch.tensor(
            tuple(frames.shape) if rank == 0 and frames is not None else (0, 0, 0, 0),
            device=self.device,
            dtype=torch.int64,
        )
        dist.broadcast(shape, src=0)
        resolved_shape = tuple(int(value) for value in shape.cpu().tolist())
        if len(resolved_shape) != 4 or any(value <= 0 for value in resolved_shape):
            raise RuntimeError(f"rank 0 broadcast an invalid decoded video shape: {resolved_shape}")
        payload = (
            frames.to(device=self.device, dtype=torch.uint8).contiguous()
            if rank == 0 and frames is not None
            else torch.empty(resolved_shape, device=self.device, dtype=torch.uint8)
        )
        dist.broadcast(payload, src=0)
        return payload.cpu()

    def _generate_video(self, prompt: str, image_path: str | None) -> torch.Tensor:
        if image_path is None:
            raise ValueError("AlayaWorld image-to-video inference requires image_path")
        self._check_assets()
        image = _fit_image(image_path, height=self.height, width=self.width)
        target_base = 1 + self.history_frames
        requested_camera_frames = (target_base + self.rounds * self.output_frames) * 8 + 1
        camera, intrinsic, rounds = self._load_camera(requested_camera_frames)
        needed_camera_frames = (target_base + rounds * self.output_frames) * 8 + 1
        camera = _pad_or_trim_frames(camera, needed_camera_frames, dim=1)
        full_actions = camera_poses_to_adaln_actions(
            camera,
            action_scale=self.action_scale,
            temporal_stride=8,
        )
        context = self._encode_prompt(prompt)

        encoder, decoder = self._build_vae()
        seed_pixel_frames = (target_base - 1) * 8 + 1
        seed_pixels = image[None, :, None].expand(1, 3, seed_pixel_frames, self.height, self.width)
        initial_latent = encoder.tiled_encode(seed_pixels, self.vae_tiling).to(dtype=self.dtype)
        if initial_latent.shape[2] != target_base:
            raise RuntimeError(f"LTX VAE returned {initial_latent.shape[2]} seed latents, expected {target_base}")
        sink = initial_latent[:, :, :1].contiguous()
        history = initial_latent[:, :, 1:target_base].contiguous()

        spatial = None
        spatial_bank: SpatialBank | None = None
        if self.spatial_enabled:
            spatial = AlayaSpatialMemory(
                video_encoder=encoder,
                video_decoder=decoder,
                encoder_tiling=self.vae_tiling,
                device=self.device,
                dtype=self.dtype,
                height=self.height,
                width=self.width,
                temporal_stride=8,
                sink_latent_frames=1,
                depth_backend=self.depth_backend,
                da3_path=self.da3_path,
                **self.spatial_options,
            )
            repeated = image[None, :, None].expand(1, 3, needed_camera_frames, self.height, self.width)
            spatial_bank = spatial.initialize(
                video_pixels=repeated,
                camera_to_world=camera,
                intrinsic=intrinsic,
                target_latent_start=target_base,
            )
            # DA3 is only needed again while appending generated RGB frames.
            # Keeping it on CUDA here overlaps it with the Alaya transformer,
            # history encoder and both VAE halves, which exceeds an 80GB card.
            # Preserve the exact DA3 depth path but run later updates from CPU.
            spatial.offload_depth_model()
            _cleanup_cuda()

        transformer, history_encoder = self._build_transformer()
        context = context.to(device=self.device, dtype=self.dtype)
        full_actions = full_actions.to(device=self.device, dtype=self.dtype)
        predictions: list[torch.Tensor] = []
        history_action_indices = (
            torch.arange(1, target_base, device=self.device) if self.action_history_memory else None
        )
        for round_index in range(rounds):
            target_start = target_base + round_index * self.output_frames
            spatial_context = (
                spatial.build_context(
                    spatial_bank,
                    camera_to_world=camera,
                    intrinsic=intrinsic,
                    target_latent_start=target_start,
                    latent_frames=self.output_frames,
                )
                if spatial is not None
                else None
            )
            generator = torch.Generator(device=self.device).manual_seed(self.seed * 1000 + round_index)
            prediction = self._denoise_chunk(
                transformer=transformer,
                history_encoder=history_encoder,
                history=history,
                sink=sink,
                context=context,
                full_actions=full_actions,
                global_target_start=target_start,
                spatial_context=spatial_context,
                generator=generator,
                history_action_indices=history_action_indices,
            ).detach()
            predictions.append(prediction)
            if spatial is not None:
                spatial.append(
                    spatial_bank,
                    latent=prediction,
                    camera_to_world=camera,
                    intrinsic=intrinsic,
                    target_latent_start=target_start,
                )
            history = torch.cat((history, prediction.to(history.dtype)), dim=2)[
                :, :, -self.history_frames :
            ].contiguous()
            if history_action_indices is not None:
                new_indices = torch.arange(
                    target_start,
                    target_start + self.output_frames,
                    device=self.device,
                )
                history_action_indices = torch.cat((history_action_indices, new_indices))[-self.history_frames :]

        del transformer, history_encoder, encoder, context, full_actions
        if spatial is not None:
            spatial.release()
        del spatial, spatial_bank
        _cleanup_cuda()

        rank0_decode = not (
            self.decode_rank0_only and dist.is_initialized() and dist.get_world_size() > 1 and dist.get_rank() != 0
        )
        frames = None
        if rank0_decode:
            keep = min(self.history_frames, 16)
            decode_latent = torch.cat(
                (
                    initial_latent[:, :, target_base - keep : target_base],
                    *(value.to(initial_latent.dtype) for value in predictions),
                ),
                dim=2,
            ).contiguous()
            prefix_pixels = (keep - 1) * 8 + 1
            frames = self._decode_video(decoder, decode_latent)[prefix_pixels:].contiguous()
            if self.requested_frames is not None:
                frames = frames[: self.requested_frames].contiguous()
            del decode_latent
        del decoder, initial_latent, predictions
        _cleanup_cuda()
        if self.decode_rank0_only and dist.is_initialized() and dist.get_world_size() > 1:
            return self._broadcast_decoded_frames(frames)
        if frames is None:
            raise RuntimeError("AlayaWorld did not decode any output frames")
        return frames

    @torch.inference_mode()
    def generate_video(self, prompt: str, image_path: str | None = None) -> torch.Tensor:
        """Generate ``rounds * output_latent_frames * 8`` uint8 RGB frames."""

        try:
            return self._generate_video(prompt, image_path)
        finally:
            # The runtime deliberately loads the large model components only for
            # one call.  Always release cached allocator blocks after an input,
            # checkpoint, DA3, or distributed failure as well as on success.
            _cleanup_cuda()

    def close(self) -> None:
        _cleanup_cuda()
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["AlayaWorldRuntime"]
