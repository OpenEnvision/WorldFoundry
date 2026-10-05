# Adapted from Tencent Hunyuan-GameCraft-1.0, under the Tencent Hunyuan Community License.
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed
import torchvision
from einops import rearrange

from worldfoundry.core.geometry.transforms import ray_condition
from worldfoundry.core.geometry.transforms import torch_meshgrid_ij as custom_meshgrid

ACTION_DICT = {
    "w": "forward",
    "a": "left",
    "d": "right",
    "s": "backward",
    "left_rot": "left_rot",
    "right_rot": "right_rot",
    "up_rot": "up_rot",
    "down_rot": "down_rot",
}


def _load_inference_cls():
    from worldfoundry.synthesis.visual_generation.hunyuan_world.gamecraft_inference.inference import (
        Inference,
    )

    return Inference


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= 10
    return np.array(ret_poses, dtype=np.float32)


def get_c2w(w2cs, transform_matrix, relative_c2w):
    if relative_c2w:
        target_cam_c2w = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
        for pose in ret_poses:
            pose[:3, -1:] *= 2
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def generate_motion_segment(current_pose, motion_type: str, value: float, duration: int = 30):
    positions = []
    rotations = []

    if motion_type in ["forward", "backward"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        pitch_rad = np.radians(current_pose["rotation"][0])
        forward_vec = np.array(
            [
                -math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                -math.cos(yaw_rad) * math.cos(pitch_rad),
            ]
        )
        direction = 1 if motion_type == "forward" else -1
        step = forward_vec * value * direction / duration

        for i in range(1, duration + 1):
            positions.append((current_pose["position"] + step * i).copy())
            rotations.append(current_pose["rotation"].copy())
        current_pose["position"] = positions[-1]

    elif motion_type in ["left", "right"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        right_vec = np.array([math.cos(yaw_rad), 0, -math.sin(yaw_rad)])
        direction = -1 if motion_type == "right" else 1
        step = right_vec * value * direction / duration

        for i in range(1, duration + 1):
            positions.append((current_pose["position"] + step * i).copy())
            rotations.append(current_pose["rotation"].copy())
        current_pose["position"] = positions[-1]

    elif motion_type.endswith("rot"):
        axis = motion_type.split("_")[0]
        total_rotation = np.zeros(3)
        if axis == "left":
            total_rotation[0] = value
        elif axis == "right":
            total_rotation[0] = -value
        elif axis == "up":
            total_rotation[2] = -value
        elif axis == "down":
            total_rotation[2] = value

        step = total_rotation / duration
        for i in range(1, duration + 1):
            positions.append(current_pose["position"].copy())
            rotations.append((current_pose["rotation"] + step * i).copy())
        current_pose["rotation"] = rotations[-1]

    return positions, rotations, current_pose


def euler_to_quaternion(angles):
    pitch, yaw, roll = np.radians(angles)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ]


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def ActionToPoseFromID(action_id, value=0.2, duration=33):
    current_pose = {
        "position": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0]),
    }
    intrinsic = [0.50505, 0.8979, 0.5, 0.5]
    positions, rotations, _ = generate_motion_segment(
        current_pose,
        ACTION_DICT[action_id],
        value,
        duration,
    )

    pose_list = [
        " ".join(
            map(
                str,
                [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
        )
    ]
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        rotation = quaternion_to_rotation_matrix(euler_to_quaternion(rot))
        extrinsic = np.hstack([rotation, pos.reshape(3, 1)])
        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))

    return pose_list


class Camera:
    def __init__(self, entry):
        self.fx, self.fy, self.cx, self.cy = entry[1:5]
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def _pose_embeddings_from_rows(poses, h, w, target_length, *, flip=False, start_index=0, step=1):
    sample_id = [start_index + i * step for i in range(target_length)]
    poses = [poses[i] for i in sample_id]

    cam_params = [[float(x) for x in pose] for pose in poses]
    if len(cam_params) != target_length:
        raise ValueError(f"Expected {target_length} camera poses, got {len(cam_params)}")
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    ratio_w = w / (cam_params[0].cx * 2)
    ratio_h = h / (cam_params[0].cy * 2)
    intrinsics = np.asarray(
        [
            [
                cam_param.fx * ratio_w,
                cam_param.fy * ratio_h,
                cam_param.cx * ratio_w,
                cam_param.cy * ratio_h,
            ]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )
    intrinsics = torch.as_tensor(intrinsics)[None]
    c2w = torch.as_tensor(get_relative_pose(cam_params))[None]
    uncond_c2w = torch.zeros_like(c2w)
    uncond_c2w[:, :] = torch.eye(4, device=c2w.device)
    flip_flag_value = torch.ones if flip else torch.zeros
    flip_flag = flip_flag_value(target_length, dtype=torch.bool, device=c2w.device)

    plucker_embedding = ray_condition(intrinsics, c2w, h, w, device="cpu", flip_flag=flip_flag)
    uncond_plucker_embedding = ray_condition(
        intrinsics,
        uncond_c2w,
        h,
        w,
        device="cpu",
        flip_flag=flip_flag,
    )

    return (
        plucker_embedding[0].permute(0, 3, 1, 2).contiguous(),
        uncond_plucker_embedding[0].permute(0, 3, 1, 2).contiguous(),
        poses,
    )


def GetPoseEmbedsFromPoses(poses, h, w, target_length, flip=False, start_index=None):
    rows = [pose.split(" ") for pose in poses]
    return _pose_embeddings_from_rows(
        rows,
        h,
        w,
        target_length,
        flip=flip,
        start_index=0 if start_index is None else start_index,
    )


def GetPoseEmbedsFromTxt(pose_dir, h, w, target_length, flip=False, start_index=None, step=1):
    with Path(pose_dir).open("r", encoding="utf-8") as f:
        rows = [pose.strip().split(" ") for pose in f.readlines()[1:]]
    return _pose_embeddings_from_rows(
        rows,
        h,
        w,
        target_length,
        flip=flip,
        start_index=0 if start_index is None else start_index,
        step=step,
    )


def convert_videos_to_grid(videos: torch.Tensor, rescale=False, n_rows=6):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for frame in videos:
        grid = torchvision.utils.make_grid(frame, nrow=n_rows)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            grid = (grid + 1.0) / 2.0
        grid = torch.clamp(grid, 0, 1).detach().cpu()
        outputs.append((grid * 255).numpy().astype(np.uint8))

    return outputs


class HunyuanGameCraftRuntime:
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        device=0,
        logger=None,
        weight_dtype=torch.bfloat16,
    ):
        _load_inference_cls().__init__(
            self,
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            device=device,
            logger=logger,
        )

        self.args = args
        self.weight_dtype = weight_dtype
        from worldfoundry.synthesis.visual_generation.hunyuan_world.gamecraft_inference.diffusion import (
            load_diffusion_pipeline,
        )

        self.pipeline = load_diffusion_pipeline(
            args,
            0,
            self.vae,
            self.text_encoder,
            self.text_encoder_2,
            self.model,
            device=self.device,
        )
        self._log_info("Loaded Hunyuan GameCraft runtime.")

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device, **kwargs):
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            raise FileNotFoundError(
                "Hunyuan GameCraft requires a local model directory. "
                f"Runtime downloads are disabled for strict in-tree execution: {pretrained_model_path}"
            )

        model_base = f"{model_root}/stdmodels"
        synthesis_t2v_model_path = f"{model_root}/gamecraft_models/mp_rank_00_model_states.pt"
        return _load_inference_cls().from_pretrained.__func__(
            cls,
            synthesis_t2v_model_path,
            model_base,
            args,
            device,
            **kwargs,
        )

    def _log_info(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _torch_device(self) -> torch.device:
        if isinstance(self.device, torch.device):
            return self.device
        if isinstance(self.device, int):
            return torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        return torch.device(str(self.device))

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size

    def get_rotary_pos_embed(self, video_length, height, width, concat_dict=None):
        from worldfoundry.synthesis.visual_generation.hunyuan_world.gamecraft_inference.helpers import (
            get_nd_rotary_pos_embed_new,
        )

        target_ndim = 3
        ndim = 5 - 2
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size"
                f"({self.model.patch_size}), but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size"
                f"({self.model.patch_size}), but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]
        else:
            raise TypeError(f"Unsupported patch_size type: {type(self.model.patch_size)!r}")

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes
        head_dim = self.model.hidden_size // self.model.num_heads
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        return get_nd_rotary_pos_embed_new(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
            concat_dict=concat_dict or {},
        )

    @torch.no_grad()
    def predict_per_action(
        self,
        prompt,
        is_image=True,
        size=(720, 1280),
        video_length=129,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        batch_size=1,
        num_videos_per_prompt=1,
        verbose=1,
        output_type="pil",
        **kwargs,
    ):
        del video_length
        out_dict = {}

        prompt_embeds = kwargs.get("prompt_embeds")
        attention_mask = kwargs.get("attention_mask")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_attention_mask = kwargs.get("negative_attention_mask")
        uncond_ref_latents = kwargs.get("uncond_ref_latents")
        return_latents = kwargs.get("return_latents", False)
        negative_prompt = kwargs.get("negative_prompt", negative_prompt)

        action_id = kwargs.get("action_id")
        action_speed = kwargs.get("action_speed")
        start_index = kwargs.get("start_index")
        last_latents = kwargs.get("last_latents")
        ref_latents = kwargs.get("ref_latents")
        input_pose = kwargs.get("input_pose")
        step = kwargs.get("step", 1)
        use_sage = kwargs.get("use_sage", False)

        size = self.parse_size(size)
        target_height = align_to(size[0], 16)
        target_width = align_to(size[1], 16)

        if input_pose is not None:
            pose_embeds, uncond_pose_embeds, poses = GetPoseEmbedsFromTxt(
                input_pose,
                target_height,
                target_width,
                33,
                kwargs.get("flip", False),
                start_index,
                step,
            )
        else:
            pose = ActionToPoseFromID(action_id, value=action_speed)
            pose_embeds, uncond_pose_embeds, poses = GetPoseEmbedsFromPoses(
                pose,
                target_height,
                target_width,
                33,
                kwargs.get("flip", False),
                0,
            )

        target_length = 34 if is_image else 66
        runtime_device = self._torch_device()
        pose_embeds = pose_embeds.unsqueeze(0).to(torch.bfloat16).to(runtime_device)
        uncond_pose_embeds = uncond_pose_embeds.unsqueeze(0).to(torch.bfloat16).to(runtime_device)

        cpu_offload = self.args.cpu_offload
        use_deepcache = kwargs.get("use_deepcache", 1)
        denoise_strength = kwargs.get("denoise_strength", 1.0)
        init_latents = kwargs.get("init_latents")
        mask = kwargs.get("mask")
        if prompt is None:
            prompt = None
            negative_prompt = None
            batch_size = prompt_embeds.shape[0]
            assert prompt_embeds is not None
        else:
            if isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif isinstance(prompt, (list, tuple)):
                batch_size = len(prompt)
            else:
                raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

        from worldfoundry.base_models.diffusion_model.schedulers.hunyuan_compat import (
            HunyuanVideoFlowMatchDiscreteScheduler as FlowMatchDiscreteScheduler,
        )

        self.pipeline.scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver,
        )

        seed = self.args.seed
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        generator = [torch.Generator(device=runtime_device).manual_seed(seed) for seed in seeds]

        out_dict["frame"] = target_length
        out_dict["size"] = (target_height, target_width)
        out_dict["video_length"] = target_length
        out_dict["seeds"] = seeds
        out_dict["negative_prompt"] = negative_prompt

        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            37 if is_image else 69,
            target_height,
            target_width,
        )
        n_tokens = freqs_cos.shape[0]

        if verbose == 1:
            self._log_info(
                "\n".join(
                    [
                        f"size: {out_dict['size']}",
                        f"video_length: {target_length}",
                        f"prompt: {prompt}",
                        f"neg_prompt: {negative_prompt}",
                        f"seed: {seed}",
                        f"infer_steps: {infer_steps}",
                        f"denoise_strength: {denoise_strength}",
                        f"use_deepcache: {use_deepcache}",
                        f"use_sage: {use_sage}",
                        f"cpu_offload: {cpu_offload}",
                        f"num_images_per_prompt: {num_videos_per_prompt}",
                        f"guidance_scale: {guidance_scale}",
                        f"n_tokens: {n_tokens}",
                        f"flow_shift: {flow_shift}",
                    ]
                )
            )

        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            last_latents=last_latents,
            cam_latents=pose_embeds,
            uncond_cam_latents=uncond_pose_embeds,
            height=target_height,
            width=target_width,
            video_length=target_length,
            gt_latents=ref_latents,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            prompt_embeds=prompt_embeds,
            ref_latents=ref_latents,
            latents=init_latents,
            denoise_strength=denoise_strength,
            mask=mask,
            uncond_ref_latents=uncond_ref_latents,
            ip_cfg_scale=self.args.ip_cfg_scale,
            use_deepcache=use_deepcache,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            output_type=output_type,
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            data_type="video" if target_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            cpu_offload=cpu_offload,
            return_latents=return_latents,
            use_sage=use_sage,
        )
        if samples is None:
            return None

        out_dict["samples"] = []
        out_dict["prompts"] = prompt
        out_dict["pose"] = poses

        if return_latents:
            latents, timesteps, last_latents, ref_latents = samples[1], samples[2], samples[3], samples[4]
            samples = samples[0][0] if samples[0] is not None and len(samples[0]) > 0 else None
            out_dict["denoised_lantents"] = latents
            out_dict["timesteps"] = timesteps
            out_dict["ref_latents"] = ref_latents
            out_dict["last_latents"] = last_latents
        else:
            samples = samples[0]

        if samples is not None:
            for sample in samples:
                out_dict["samples"].append(sample.unsqueeze(0))

        self._log_info(f"Success, time: {time.time() - start_time}")
        return out_dict

    def predict(
        self,
        ref_images,
        last_latents,
        ref_latents,
        action_list,
        action_speed_list,
        prompt,
        negative_prompt,
        size,
        video_length,
        guidance_scale,
        infer_steps,
        flow_shift,
        first_is_image: bool = True,
        return_latents: bool = False,
        **kwargs,
    ):
        del video_length
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )

        if len(action_list) == 0:
            raise ValueError("action_list is empty")

        out_cat = None
        for idx, action_id in enumerate(action_list):
            outputs = self.predict_per_action(
                is_image=first_is_image and idx == 0,
                ref_images=ref_images,
                last_latents=last_latents,
                ref_latents=ref_latents,
                action_id=action_id,
                action_speed=action_speed_list[idx],
                prompt=prompt,
                negative_prompt=negative_prompt,
                return_latents=True,
                size=size,
                guidance_scale=guidance_scale,
                infer_steps=infer_steps,
                flow_shift=flow_shift,
                **kwargs,
            )

            ref_latents = outputs["ref_latents"]
            last_latents = outputs["last_latents"]

            if rank == 0:
                sub_samples = outputs["samples"][0]
                out_cat = sub_samples if out_cat is None else torch.cat([out_cat, sub_samples], dim=2)

        video_frames = convert_videos_to_grid(out_cat, n_rows=1) if rank == 0 and out_cat is not None else None
        if return_latents:
            return {
                "video": video_frames,
                "last_latents": last_latents,
                "ref_latents": ref_latents,
            }
        return video_frames


def load_runtime(*args: Any, **kwargs: Any) -> HunyuanGameCraftRuntime:
    return HunyuanGameCraftRuntime.from_pretrained(*args, **kwargs)


__all__ = [
    "ACTION_DICT",
    "ActionToPoseFromID",
    "Camera",
    "GetPoseEmbedsFromPoses",
    "GetPoseEmbedsFromTxt",
    "HunyuanGameCraftRuntime",
    "align_to",
    "convert_videos_to_grid",
    "custom_meshgrid",
    "euler_to_quaternion",
    "generate_motion_segment",
    "get_c2w",
    "get_relative_pose",
    "load_runtime",
    "quaternion_to_rotation_matrix",
    "ray_condition",
]
