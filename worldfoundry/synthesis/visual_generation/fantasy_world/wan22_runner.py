"""Native dual-denoiser inference runner for FantasyWorld Wan2.2."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .native_pipeline import build_fantasy_world_wan22_models
from .runtime_env import (
    WAN22_LORA_HIGH_NAME,
    WAN22_LORA_LOW_NAME,
    ensure_moge2_runtime,
    resolve_moge_pretrained,
)
from .worldfoundry_runtime import normalize_wan_num_frames, pad_camera_params_to_frames


class FantasyWorldWan22Runner:
    """Run the released high/low Wan2.2 models on one native architecture."""

    @staticmethod
    def _canonical_cuda_device(device: str) -> str:
        return "cuda:0" if device == "cuda" else device

    @classmethod
    def _auto_device_layout(
        cls,
        base_device: str,
        high_model_device: Optional[str],
        low_model_device: Optional[str],
        moge_device: Optional[str],
    ) -> tuple[str, str, str]:
        resolved_base = cls._canonical_cuda_device(base_device)
        if not resolved_base.startswith("cuda"):
            return resolved_base, resolved_base, resolved_base
        count = torch.cuda.device_count()
        if count <= 1:
            high = cls._canonical_cuda_device(high_model_device or resolved_base)
            return (
                high,
                cls._canonical_cuda_device(low_model_device or high),
                cls._canonical_cuda_device(moge_device or high),
            )
        base_index = int(resolved_base.split(":", 1)[1])
        high = cls._canonical_cuda_device(high_model_device or f"cuda:{base_index}")
        high_index = int(high.split(":", 1)[1])
        low = cls._canonical_cuda_device(low_model_device or f"cuda:{(high_index + 1) % count}")
        if moge_device is None:
            free = [index for index in range(count) if f"cuda:{index}" not in {high, low}]
            moge = f"cuda:{free[0]}" if free else high
        else:
            moge = cls._canonical_cuda_device(moge_device)
        return high, low, moge

    def __init__(
        self,
        *,
        base_dir: str,
        lora_dir: str,
        model_ckpt_high: str,
        model_ckpt_low: str,
        moge_path: Optional[str] = None,
        moge_pretrained: Optional[str] = None,
        base_seed: int = -1,
        sample_steps: int = 50,
        cfg_scale: float = 5.0,
        timestep_boundary: int = 900,
        frames: int = 81,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        device: str = "cuda",
        high_model_device: Optional[str] = None,
        low_model_device: Optional[str] = None,
        moge_device: Optional[str] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("FantasyWorld Wan2.2 official inference requires a CUDA device.")
        ensure_moge2_runtime(moge_path)

        from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.variants.fantasy_world.utils.pose_enc import (
            extri_intri_to_pose_encoding,
            pose_encoding_to_extri_intri,
        )
        from worldfoundry.core.geometry.pose import RealEstate10KPoseProcessor

        from . import utils as fw_utils

        self.base_seed = int(base_seed) if int(base_seed) >= 0 else random.randint(0, sys.maxsize)
        self.sample_steps = int(sample_steps)
        self.cfg_scale = float(cfg_scale)
        self.fps = int(fps)
        self.high_device, self.low_device, self.moge_device = self._auto_device_layout(
            str(device), high_model_device, low_model_device, moge_device
        )
        self.device = self.high_device
        self.torch_dtype = weight_dtype
        self.num_frames = normalize_wan_num_frames(frames)
        self.height = int(height)
        self.width = int(width)
        self.timestep_boundary = int(timestep_boundary)
        self._fw_utils = fw_utils
        self._extri_intri_to_pose_encoding = extri_intri_to_pose_encoding

        lora_root = Path(lora_dir).expanduser().resolve()
        models = build_fantasy_world_wan22_models(
            base_model_root=base_dir,
            high_lora_path=lora_root / WAN22_LORA_HIGH_NAME,
            low_lora_path=lora_root / WAN22_LORA_LOW_NAME,
            high_checkpoint_path=model_ckpt_high,
            low_checkpoint_path=model_ckpt_low,
            high_device=self.high_device,
            low_device=self.low_device,
            torch_dtype=self.torch_dtype,
        )
        self.model_high = models.high.to(self.high_device).eval()
        self.model_low = models.low.to(self.low_device).eval()
        self.model_high.pipe.device = self.high_device
        self.model_high.pipe.torch_dtype = self.torch_dtype
        self.model_low.pipe.device = self.low_device
        self.model_low.pipe.torch_dtype = self.torch_dtype

        self.pose_processor = RealEstate10KPoseProcessor(
            sample_stride=1,
            sample_n_frames=self.num_frames,
            relative_pose=True,
            zero_t_first_frame=True,
            sample_size=[self.height, self.width],
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            is_i2v=True,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        )
        self.moge = MoGeModel.from_pretrained(
            resolve_moge_pretrained(moge_pretrained)
        ).to(self.moge_device).eval()

    def _prepare_control_latents(
        self,
        plucker_embedding: torch.Tensor,
        target_device: str,
    ) -> torch.Tensor:
        camera_video = plucker_embedding.to(
            device=target_device,
            dtype=self.torch_dtype,
        )[0].permute(3, 0, 1, 2).unsqueeze(0)
        values = torch.cat(
            (
                torch.repeat_interleave(camera_video[:, :, :1], repeats=4, dim=2),
                camera_video[:, :, 1:],
            ),
            dim=2,
        ).transpose(1, 2)
        batch, frames, channels, height, width = values.shape
        if frames % 4:
            raise ValueError("FantasyWorld Wan2.2 camera-control frames must be divisible by four")
        values = values.reshape(batch, frames // 4, 4, channels, height, width).transpose(2, 3)
        return values.reshape(batch, frames // 4, channels * 4, height, width).transpose(1, 2).contiguous()

    def generate_video_with_dual_models(
        self,
        *,
        context_pos: torch.Tensor,
        context_neg: torch.Tensor | None,
        y: torch.Tensor,
        plucker_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict | None]:
        high_pipe = self.model_high.pipe
        low_pipe = self.model_low.pipe
        high_pipe.scheduler.set_timesteps(self.sample_steps)
        low_pipe.scheduler.set_timesteps(self.sample_steps)
        latent_channels = int(high_pipe.vae.model.z_dim)
        latents = high_pipe.generate_noise(
            (
                1,
                latent_channels,
                (self.num_frames - 1) // 4 + 1,
                self.height // high_pipe.vae.upsampling_factor,
                self.width // high_pipe.vae.upsampling_factor,
            ),
            seed=self.base_seed,
            device=self.high_device,
            dtype=torch.float32,
        ).to(device=self.high_device, dtype=self.torch_dtype)

        devices = {self.high_device, self.low_device}
        control_by_device = {
            target: self._prepare_control_latents(plucker_embedding, target)
            for target in devices
        }
        y_by_device = {
            target: y.to(device=target, dtype=self.torch_dtype)
            for target in devices
        }
        positive_by_device = {
            target: context_pos.to(device=target, dtype=self.torch_dtype)
            for target in devices
        }
        negative_by_device = (
            None
            if context_neg is None
            else {
                target: context_neg.to(device=target, dtype=self.torch_dtype)
                for target in devices
            }
        )
        final_prediction = None

        for progress_id, step_t in enumerate(tqdm(high_pipe.scheduler.timesteps)):
            use_high = float(step_t) > self.timestep_boundary
            model = self.model_high if use_high else self.model_low
            target = self.high_device if use_high else self.low_device
            latents = latents.to(target)
            timestep = step_t.unsqueeze(0).to(device=target, dtype=self.torch_dtype)
            positive, prediction = model.joint_forward(
                latents,
                timestep=timestep,
                context=positive_by_device[target],
                y=y_by_device[target],
                control_camera_latents_input=control_by_device[target],
                return_prediction=progress_id == self.sample_steps - 1,
            )
            if negative_by_device is None or self.cfg_scale == 1.0:
                noise_prediction = positive
            else:
                negative, _ = model.joint_forward(
                    latents,
                    timestep=timestep,
                    context=negative_by_device[target],
                    y=y_by_device[target],
                    control_camera_latents_input=control_by_device[target],
                )
                noise_prediction = negative + self.cfg_scale * (positive - negative)
            latents = model.pipe.scheduler.step(noise_prediction, step_t.to(target), latents)
            final_prediction = prediction
        return latents, final_prediction

    def generate_video(
        self,
        *,
        image: Image.Image,
        end_image: Optional[Image.Image],
        prompt: str,
        neg_prompt: Optional[str],
        camera_params,
        using_scale: bool = True,
    ):
        neg_prompt = neg_prompt or ""
        with torch.no_grad():
            input_image = image.convert("RGB")
            camera_params = pad_camera_params_to_frames(camera_params, self.num_frames)
            input_image_tensor = torch.tensor(
                np.array(input_image) / 255,
                dtype=torch.float32,
                device=self.moge_device,
            ).permute(2, 0, 1)
            moge_output = self.moge.infer(input_image_tensor)
            moge = {key: value.cpu().contiguous() for key, value in moge_output.items()}
            intrinsics = torch.from_numpy(
                np.stack([self._fw_utils.get_intrinsic_matrix(camera) for camera in camera_params]).astype(
                    np.float32
                )
            )
            extrinsics = torch.from_numpy(
                np.stack([camera.w2c_mat for camera in camera_params]).astype(np.float32)
            )
            if using_scale:
                first_world, first_mask = self._fw_utils.batch_depth_to_world(
                    prediction=moge,
                    extrinsics=extrinsics[0, :3, :].unsqueeze(0),
                    intrinsics=intrinsics[0].unsqueeze(0),
                )
                extrinsics = self._fw_utils.normalize_scene(
                    extrinsics=extrinsics.unsqueeze(0)[:, :, :3, :],
                    first_moge_world=first_world.unsqueeze(0),
                    first_moge_mask=first_mask.unsqueeze(0),
                ).squeeze(0)
            pose_encoding = self._extri_intri_to_pose_encoding(
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                [self.height, self.width],
                pose_encoding_type="absT_quaR_FoV",
            ).squeeze(0)
            plucker_embedding = self.pose_processor.get_plucker_embedding_direct_from_cam_params(
                pose_encoding.unsqueeze(0),
                image_size=(self.height, self.width),
            ).to(dtype=self.torch_dtype)

            image_embedding = self.model_high.pipe.encode_image(
                input_image,
                end_image,
                self.num_frames,
                self.height,
                self.width,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            y = image_embedding["y"]
            context_pos = self.model_high.pipe.encode_prompt(prompt or "")["context"]
            context_neg = self.model_high.pipe.encode_prompt(neg_prompt)["context"]

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            latent_video, prediction = self.generate_video_with_dual_models(
                context_pos=context_pos,
                context_neg=context_neg,
                y=y,
                plucker_embedding=plucker_embedding,
            )
            decoded = self.model_high.pipe.decode_video(
                latent_video.to(self.high_device),
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
        video = decoded.squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu()
        frames = ((video + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        return frames, prediction


def build_wan22_runner(
    *,
    base_dir: str,
    lora_dir: str,
    model_ckpt_high: str,
    model_ckpt_low: str,
    moge_path: Optional[str] = None,
    moge_pretrained: Optional[str] = None,
    base_seed: int = -1,
    sample_steps: int = 50,
    cfg_scale: float = 5.0,
    timestep_boundary: int = 900,
    frames: int = 81,
    fps: int = 16,
    height: int = 480,
    width: int = 832,
    device: str = "cuda",
    high_model_device: Optional[str] = None,
    low_model_device: Optional[str] = None,
    moge_device: Optional[str] = None,
    weight_dtype: torch.dtype = torch.bfloat16,
) -> FantasyWorldWan22Runner:
    return FantasyWorldWan22Runner(
        base_dir=base_dir,
        lora_dir=lora_dir,
        model_ckpt_high=model_ckpt_high,
        model_ckpt_low=model_ckpt_low,
        moge_path=moge_path,
        moge_pretrained=moge_pretrained,
        base_seed=base_seed,
        sample_steps=sample_steps,
        cfg_scale=cfg_scale,
        timestep_boundary=timestep_boundary,
        frames=frames,
        fps=fps,
        height=height,
        width=width,
        device=device,
        high_model_device=high_model_device,
        low_model_device=low_model_device,
        moge_device=moge_device,
        weight_dtype=weight_dtype,
    )


__all__ = ["FantasyWorldWan22Runner", "build_wan22_runner"]
