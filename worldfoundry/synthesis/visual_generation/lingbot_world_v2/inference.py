"""Inference-only LingBot-World-V2 causal-fast pipeline.

Adapted from ``Robbyant/lingbot-world-v2`` commit ``94f43115`` under
CC BY-NC-SA 4.0. Common model and runtime components are imported from
WorldFoundry rather than duplicated here.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import random
import sys
import tempfile
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming import (
    Wan2_1_VAE,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import (
    T5EncoderModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.lingbot_world_v2 import (
    LingBotWorldV2Model,
)
from worldfoundry.base_models.diffusion_model.recipes.wan_configs.lingbot_world_v2 import (
    LINGBOT_WORLD_V2_CONFIG,
)
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import (
    FlowUniPCMultistepScheduler,
)
from worldfoundry.core.attention.parallel.causal_rope_sequence_parallel import (
    sp_attn_forward_causal_chunked,
    sp_dit_forward_causal_chunked,
)
from worldfoundry.core.distributed.sharding.block_fsdp import shard_model
from worldfoundry.core.distributed.sequence_parallel.ops import get_world_size
from worldfoundry.core.geometry.transforms import ray_condition
from worldfoundry.operators.lingbot_world_operator import (
    compute_relative_poses,
    get_Ks_transformed,
    interpolate_camera_poses,
)


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        return isinstance(module, FullyShardedDataParallel)
    except (ImportError, RuntimeError):
        return "FullyShardedDataParallel" in type(module).__name__


class LingBotWorldV2Inference:
    """Public LingBot-World-V2 14B causal-fast inference implementation."""

    @staticmethod
    @contextmanager
    def _transformer_load_root(checkpoint_dir: Path, subfolder: str):
        """Present split model assets in the layout expected by Diffusers.

        Released HFD snapshots keep ``config.json`` at repository root while
        transformer shards live below ``transformers/``.  Diffusers expects
        both in one directory.  A process-local symlink view avoids mutating
        shared checkpoints and adds no large-file copy.
        """

        transformer_root = checkpoint_dir / subfolder
        if (transformer_root / "config.json").is_file():
            yield transformer_root
            return
        root_config = checkpoint_dir / "config.json"
        if not root_config.is_file():
            raise FileNotFoundError(
                f"LingBot-World-V2 transformer config is missing: {root_config}"
            )
        with tempfile.TemporaryDirectory(prefix="worldfoundry_lingbot_v2_model_") as raw:
            load_root = Path(raw)
            (load_root / "config.json").symlink_to(root_config)
            for source in transformer_root.iterdir():
                (load_root / source.name).symlink_to(source)
            yield load_root

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device_id: int = 0,
        rank: int = 0,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_sp: bool = False,
        t5_cpu: bool = False,
        init_on_cpu: bool = True,
        convert_model_dtype: bool = False,
        pipe_dtype: torch.dtype = torch.bfloat16,
        local_attn_size: int = 18,
        sink_size: int = 6,
    ) -> None:
        self.config = LINGBOT_WORLD_V2_CONFIG
        self.device = torch.device(f"cuda:{device_id}")
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.param_dtype = self.config.param_dtype
        self.pipe_dtype = pipe_dtype
        # Resident interactive sessions consume the same small core contract
        # as other causal Wan models.  Keep these aliases on the loaded core so
        # realtime adapters never need to special-case config layouts.
        self.num_train_timesteps = self.config.num_train_timesteps
        self.vae_stride = self.config.vae_stride
        self.patch_size = self.config.patch_size
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.sp_size = get_world_size() if use_sp else 1
        checkpoint_dir = Path(checkpoint_dir)

        if t5_fsdp or dit_fsdp or use_sp:
            init_on_cpu = False
        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(checkpoint_dir / self.config.t5_checkpoint),
            tokenizer_path=str(checkpoint_dir / self.config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        self.vae = Wan2_1_VAE(
            vae_pth=str(checkpoint_dir / self.config.vae_checkpoint),
            device=self.device,
        )

        logging.info("Loading LingBot-World-V2 causal-fast DiT from %s", checkpoint_dir)
        with self._transformer_load_root(checkpoint_dir, self.config.fast_checkpoint) as load_root:
            model = LingBotWorldV2Model.from_pretrained(
                load_root,
                torch_dtype=torch.bfloat16,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            )
        model.eval().requires_grad_(False)
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal_chunked,
                    block.self_attn,
                )
            model.forward = types.MethodType(sp_dit_forward_causal_chunked, model)
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not init_on_cpu:
                model.to(self.device)
        self.model = model if dit_fsdp else model.to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._text_cache: dict[str, list[torch.Tensor]] = {}
        self._cross_attention_initialized = False

    def clear_text_cache(self) -> None:
        """Release cached T5 embeddings."""
        self._text_cache.clear()

    def _encode_prompt(self, prompt: str, offload_model: bool) -> list[torch.Tensor]:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.t5_cpu:
            context = [item.to(self.device) for item in self.text_encoder([prompt], torch.device("cpu"))]
        else:
            text_model = self.text_encoder.model
            text_is_fsdp = _is_fsdp_module(text_model)
            if not text_is_fsdp:
                text_model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            if offload_model and not text_is_fsdp:
                text_model.cpu()
        self._text_cache[cache_key] = context
        return context

    @staticmethod
    def _flow_to_x0(flow_pred, latent, timestep, scheduler):
        original_dtype = flow_pred.dtype
        flow_pred, latent, sigmas, timesteps = (
            item.double().to(flow_pred.device) for item in (flow_pred, latent, scheduler.sigmas, scheduler.timesteps)
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (latent - sigma * flow_pred).to(original_dtype)

    @staticmethod
    def _self_attention_cache(num_layers, shape, dtype, device):
        return [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(num_layers)
        ]

    @staticmethod
    def _cross_attention_cache(num_layers, shape, dtype, device):
        return [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "is_init": torch.tensor([0], dtype=torch.bool, device=device),
            }
            for _ in range(num_layers)
        ]

    _initialize_self_kv_cache = _self_attention_cache
    _initialize_crossattn_cache = _cross_attention_cache

    @staticmethod
    def _load_actions(action_path: str | Path, frame_num: int) -> tuple[np.ndarray, torch.Tensor, int]:
        action_dir = Path(action_path)
        poses_path = action_dir / "poses.npy"
        intrinsics_path = action_dir / "intrinsics.npy"
        if not poses_path.is_file() or not intrinsics_path.is_file():
            raise FileNotFoundError(
                f"LingBot-World-V2 action_path must contain poses.npy and intrinsics.npy: {action_dir}"
            )
        poses = np.load(poses_path)
        intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"Expected poses.npy shape [F,4,4], got {poses.shape}.")
        if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
            raise ValueError(f"Expected intrinsics.npy shape [F,4], got {tuple(intrinsics.shape)}.")
        if len(intrinsics) < 1:
            raise ValueError("intrinsics.npy must contain at least one camera calibration row.")
        if frame_num < 5 or (frame_num - 1) % 4:
            raise ValueError("frame_num must be at least 5 and follow the 4n+1 layout.")
        if len(poses) < frame_num:
            raise ValueError(f"Requested {frame_num} frames but only {len(poses)} camera poses were supplied.")
        return poses[:frame_num], intrinsics, frame_num

    def _camera_condition(
        self,
        poses: np.ndarray,
        intrinsics: torch.Tensor,
        *,
        height: int,
        width: int,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        chunk_size: int,
    ) -> torch.Tensor:
        intrinsics = get_Ks_transformed(
            intrinsics,
            height_org=480,
            width_org=832,
            height_resize=height,
            width_resize=width,
            height_final=height,
            width_final=width,
        )[0]
        target_frames = (len(poses) - 1) // 4 + 1
        target_frames -= target_frames % chunk_size
        camera_poses = interpolate_camera_poses(
            src_indices=np.linspace(0, len(poses) - 1, len(poses)),
            src_rot_mat=poses[:, :3, :3],
            src_trans_vec=poses[:, :3, 3],
            tgt_indices=np.linspace(0, len(poses) - 1, target_frames),
        )
        camera_poses = compute_relative_poses(camera_poses, framewise=True).to(self.device)
        intrinsics = intrinsics.repeat(len(camera_poses), 1).to(self.device)
        camera = ray_condition(
            intrinsics[None],
            camera_poses[None],
            height,
            width,
            self.device,
            use_ray_o=True,
        )[0]
        camera = rearrange(
            camera,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=height // latent_height,
            c2=width // latent_width,
        )
        camera = rearrange(
            camera[None],
            "b (f h w) c -> b c f h w",
            f=latent_frames,
            h=latent_height,
            w=latent_width,
        )
        return camera.to(self.param_dtype)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image,
        action_path: str | Path,
        *,
        chunk_size: int = 4,
        max_area: int = 480 * 832,
        frame_num: int = 361,
        timestep_indices: tuple[int, ...] = (0, 250, 500, 750),
        shift: float = 10.0,
        seed: int = 42,
        offload_model: bool = False,
        max_sequence_length: int = 512,
        max_attention_size: int | None = None,
    ) -> torch.Tensor | None:
        """Generate a camera-controlled video on rank zero."""
        if not isinstance(prompt, str):
            raise TypeError("LingBot-World-V2 currently supports one string prompt per run.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        poses, intrinsics, frame_num = self._load_actions(action_path, frame_num)
        requested_frame_num = frame_num
        image = TF.to_tensor(image).sub_(0.5).div_(0.5).to(self.device)

        image_height, image_width = image.shape[1:]
        aspect_ratio = image_height / image_width
        latent_height = round(
            np.sqrt(max_area * aspect_ratio)
            // self.config.vae_stride[1]
            // self.config.patch_size[1]
            * self.config.patch_size[1]
        )
        latent_width = round(
            np.sqrt(max_area / aspect_ratio)
            // self.config.vae_stride[2]
            // self.config.patch_size[2]
            * self.config.patch_size[2]
        )
        height = latent_height * self.config.vae_stride[1]
        width = latent_width * self.config.vae_stride[2]
        latent_frames = (frame_num - 1) // self.config.vae_stride[0] + 1
        # The causal model requires complete chunks. Extend the final chunk
        # with a stationary camera and trim decoded frames, rather than
        # silently dropping requested frames (49 previously became 45).
        latent_frames = math.ceil(latent_frames / chunk_size) * chunk_size
        if latent_frames < chunk_size:
            raise ValueError(f"frame_num={frame_num} is too short for chunk_size={chunk_size}.")
        frame_num = (latent_frames - 1) * self.config.vae_stride[0] + 1
        if frame_num > len(poses):
            poses = np.concatenate((poses, np.repeat(poses[-1:], frame_num - len(poses), axis=0)))
        frame_seq_len = latent_height * latent_width // (self.config.patch_size[1] * self.config.patch_size[2])
        max_seq_len = math.ceil(chunk_size * frame_seq_len / self.sp_size) * self.sp_size
        self._cross_attention_initialized = False

        generator = torch.Generator(device=self.device).manual_seed(
            seed if seed >= 0 else random.randint(0, sys.maxsize)
        )
        noise = torch.randn(
            16,
            latent_frames,
            latent_height,
            latent_width,
            dtype=torch.float32,
            generator=generator,
            device=self.device,
        )
        mask = torch.ones(1, frame_num, latent_height, latent_width, device=self.device)
        mask[:, 1:] = 0
        mask = torch.cat([mask[:, :1].repeat_interleave(4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, latent_height, latent_width).transpose(1, 2)[0]

        self.scheduler.set_timesteps(self.config.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[list(timestep_indices)]
        context = self._encode_prompt(prompt, offload_model)
        camera = self._camera_condition(
            poses,
            intrinsics,
            height=height,
            width=width,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            chunk_size=chunk_size,
        )
        condition = self.vae.encode(
            [
                torch.cat(
                    [
                        torch.nn.functional.interpolate(
                            image[None].cpu(), size=(height, width), mode="bicubic"
                        ).transpose(0, 1),
                        torch.zeros(3, frame_num - 1, height, width),
                    ],
                    dim=1,
                ).to(self.device)
            ]
        )[0]
        condition = torch.cat([mask, condition])

        model_config = self.model.config
        kv_size = frame_seq_len * (self.local_attn_size if self.local_attn_size > -1 else latent_frames)
        head_dim = model_config.dim // model_config.num_heads
        self_cache = self._self_attention_cache(
            model_config.num_layers,
            [1, kv_size, model_config.num_heads // self.sp_size, head_dim],
            self.pipe_dtype,
            self.device,
        )
        cross_cache = self._cross_attention_cache(
            model_config.num_layers,
            [1, max_sequence_length, model_config.num_heads, head_dim],
            self.pipe_dtype,
            self.device,
        )

        @contextmanager
        def no_sync():
            yield

        sync_context = getattr(self.model, "no_sync", no_sync)
        predicted_chunks = []
        with torch.amp.autocast("cuda", dtype=self.param_dtype), sync_context():
            for chunk_id, (latent, current_condition, current_camera) in enumerate(
                tqdm(
                    zip(
                        noise.split(chunk_size, dim=1),
                        condition.split(chunk_size, dim=1),
                        camera.split(chunk_size, dim=2),
                    ),
                    total=latent_frames // chunk_size,
                )
            ):
                model_kwargs = {
                    "context": [context[0]],
                    "seq_len": max_seq_len,
                    "y": [current_condition],
                    "dit_cond_dict": {"c2ws_plucker_emb": current_camera.chunk(1, dim=0)},
                    "kv_cache": self_cache,
                    "crossattn_cache": cross_cache,
                    "current_start": chunk_id * chunk_size * frame_seq_len,
                    "max_attention_size": kv_size if max_attention_size is None else max_attention_size,
                    "frame_seqlen": frame_seq_len,
                }
                for timestep_index, timestep in enumerate(timesteps):
                    noise_prediction = self.model(
                        x=[latent.to(self.device)],
                        t=timestep[None].to(self.device),
                        cross_attn_first_call=not self._cross_attention_initialized,
                        **model_kwargs,
                    )[0]
                    self._cross_attention_initialized = True
                    x0 = self._flow_to_x0(noise_prediction, latent, timestep, self.scheduler)
                    if timestep_index + 1 < len(timesteps):
                        latent = self.scheduler.add_noise(
                            x0,
                            torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype),
                            timesteps[timestep_index + 1],
                        )
                predicted_chunks.append(x0)
                self.model(
                    x=[x0],
                    t=torch.zeros(1, device=self.device),
                    cross_attn_first_call=False,
                    **model_kwargs,
                )

        predicted = torch.cat(predicted_chunks, dim=1)
        if offload_model:
            self.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        videos = self.vae.decode([predicted]) if self.rank == 0 else None
        if dist.is_initialized():
            dist.barrier()
        return videos[0][:, :requested_frame_num] if videos is not None else None


__all__ = ["LingBotWorldV2Inference"]
