"""Native inference runner for FantasyWorld Wan2.1."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .native_pipeline import build_fantasy_world_wan21_model
from .runtime_env import (
    DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT,
    ensure_moge2_runtime,
    resolve_moge_pretrained,
)
from .worldfoundry_runtime import normalize_wan_num_frames, pad_camera_params_to_frames


class FantasyWorldWan21Runner:
    """Compose native Wan/VGGT/MoGe roles for released Wan2.1 inference."""

    def __init__(
        self,
        *,
        ckpt_dir: str,
        model_ckpt: str,
        moge_path: Optional[str] = None,
        moge_pretrained: Optional[str] = None,
        sample_steps: int = 50,
        sample_guide_scale: float = 5.0,
        frames: int = 81,
        fps: int = 16,
        height: int = 336,
        width: int = 592,
        start_index: int = 16,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("FantasyWorld Wan2.1 official inference requires a CUDA device.")
        if int(start_index) != 16:
            raise ValueError("the released FantasyWorld Wan2.1 checkpoint starts fusion at Wan block 16")
        ensure_moge2_runtime(moge_path)

        from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.variants.fantasy_world.utils.pose_enc import (
            extri_intri_to_pose_encoding,
            pose_encoding_to_extri_intri,
        )
        from worldfoundry.core.geometry.pose import RealEstate10KPoseProcessor

        from . import utils as fw_utils

        self.sample_steps = int(sample_steps)
        self.sample_guide_scale = float(sample_guide_scale)
        self.fps = int(fps)
        self.device = str(device)
        self.torch_dtype = weight_dtype
        self.num_frames = normalize_wan_num_frames(frames)
        self.height = int(height)
        self.width = int(width)
        self.start_index = 16
        self.default_negative_prompt = DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT
        self._fw_utils = fw_utils
        self._extri_intri_to_pose_encoding = extri_intri_to_pose_encoding

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
        self.model = build_fantasy_world_wan21_model(
            base_model_root=ckpt_dir,
            checkpoint_path=model_ckpt,
            device=self.device,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()
        self.model.pipe.device = self.device
        self.model.pipe.torch_dtype = self.torch_dtype
        self.moge = MoGeModel.from_pretrained(
            resolve_moge_pretrained(moge_pretrained)
        ).to(self.device).eval()

    def _sample_latents(
        self,
        *,
        context_positive: torch.Tensor,
        context_negative: torch.Tensor | None,
        clip_feature: torch.Tensor,
        condition_latents: torch.Tensor,
        plucker_embedding: torch.Tensor,
        seed: int,
    ) -> tuple[torch.Tensor, dict | None]:
        pipe = self.model.pipe
        pipe.scheduler.set_timesteps(self.sample_steps)
        latent_frames = (self.num_frames - 1) // 4 + 1
        latent_channels = int(pipe.vae.model.z_dim)
        latents = pipe.generate_noise(
            (
                1,
                latent_channels,
                latent_frames,
                self.height // pipe.vae.upsampling_factor,
                self.width // pipe.vae.upsampling_factor,
            ),
            seed=seed,
            device=self.device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        plucker_features = self.model.camera_condition.get_pose_fea(plucker_embedding)
        plucker_context_lens = torch.ones(
            latent_frames,
            dtype=torch.long,
            device=self.device,
        )
        plucker_context_lens[1:] = 4
        final_prediction = None

        for progress_id, step_t in enumerate(tqdm(pipe.scheduler.timesteps)):
            timestep = step_t.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            positive, prediction = self.model.joint_forward(
                latents,
                timestep=timestep,
                context=context_positive,
                clip_feature=clip_feature,
                y=condition_latents,
                plucker_fea=plucker_features,
                plucker_context_lens=plucker_context_lens,
                return_prediction=progress_id == self.sample_steps - 1,
            )
            if context_negative is None or self.sample_guide_scale == 1.0:
                noise_prediction = positive
            else:
                negative, _ = self.model.joint_forward(
                    latents,
                    timestep=timestep,
                    context=context_negative,
                    clip_feature=clip_feature,
                    y=condition_latents,
                    plucker_fea=plucker_features,
                    plucker_context_lens=plucker_context_lens,
                )
                noise_prediction = negative + self.sample_guide_scale * (positive - negative)
            latents = pipe.scheduler.step(noise_prediction, step_t, latents)
            final_prediction = prediction
        return latents, final_prediction

    def generate_video(
        self,
        *,
        image: Image.Image,
        camera_params,
        prompt: str,
        neg_prompt: Optional[str] = None,
        using_scale: bool = True,
        seed: int = 1024,
    ):
        neg_prompt = self.default_negative_prompt if neg_prompt is None else neg_prompt
        with torch.no_grad():
            input_image = image.convert("RGB")
            camera_params = pad_camera_params_to_frames(camera_params, self.num_frames)
            input_tensor = torch.tensor(
                np.array(input_image) / 255,
                dtype=torch.float32,
                device=self.device,
            ).permute(2, 0, 1)
            moge_output = self.moge.infer(input_tensor)
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
            ).to(device=self.device, dtype=self.torch_dtype)
            image_embedding = self.model.pipe.encode_image(
                input_image,
                None,
                self.num_frames,
                self.height,
                self.width,
            )
            clip_feature = image_embedding["clip_feature"].to(self.device, self.torch_dtype)
            condition_latents = image_embedding["y"].to(self.device, self.torch_dtype)
            context_positive = self.model.pipe.encode_prompt(prompt or "")["context"].to(
                self.device, self.torch_dtype
            )
            context_negative = self.model.pipe.encode_prompt(neg_prompt or "")["context"].to(
                self.device, self.torch_dtype
            )

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            latent_video, prediction = self._sample_latents(
                context_positive=context_positive,
                context_negative=context_negative,
                clip_feature=clip_feature,
                condition_latents=condition_latents,
                plucker_embedding=plucker_embedding,
                seed=int(seed),
            )
            decoded = self.model.pipe.decode_video(
                latent_video,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
        video = decoded.squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu()
        frames = ((video + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
        return frames, prediction


__all__ = ["FantasyWorldWan21Runner"]
