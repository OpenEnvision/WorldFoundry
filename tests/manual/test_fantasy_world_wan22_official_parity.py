from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.fantasy_world.runtime_env import (
    ensure_fantasy_world_runtime,
)


def _should_run() -> bool:
    return os.environ.get("RUN_FANTASYWORLD_WAN22_PARITY", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_required_path(env_name: str, default_path: Path) -> Path | None:
    value = os.environ.get(env_name)
    candidate = Path(value).expanduser().resolve() if value else default_path.resolve()
    return candidate if candidate.exists() else None


def _select_devices() -> tuple[str, str, str] | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None

    free_by_gpu: list[tuple[int, int]] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_by_gpu.append((index, int(free_bytes)))
    free_by_gpu.sort(key=lambda item: item[1], reverse=True)

    high_index, high_free = free_by_gpu[0]
    low_index, low_free = free_by_gpu[1]
    min_free_bytes = 60 * 1024**3
    if high_free < min_free_bytes or low_free < min_free_bytes:
        return None

    high_device = f"cuda:{high_index}"
    low_device = f"cuda:{low_index}"
    moge_device = high_device
    return high_device, low_device, moge_device


def _build_camera_inputs(frames: int, height: int, width: int):
    ensure_fantasy_world_runtime()
    from worldfoundry.synthesis.visual_generation.fantasy_world import utils as fw_utils

    fx = fy = 500.0
    K = np.array(
        [
            [fx, 0.0, (width - 1) / 2.0],
            [0.0, fy, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    poses = []
    for idx in range(frames):
        ratio = idx / max(frames - 1, 1)
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = 0.15 * ratio
        c2w[1, 3] = 0.02 * np.sin(ratio * np.pi)
        poses.append(c2w)

    camera_params = fw_utils.cameras_json_to_camera_list(
        {"cameras_interp": [pose.tolist() for pose in poses]},
        image_size=(height, width),
        K=K,
    )
    return K, camera_params


def _reference_generate_video_with_dual_models(
    runner,
    *,
    context_pos: torch.Tensor,
    context_neg: torch.Tensor,
    y: torch.Tensor,
    plucker_embedding: torch.Tensor,
):
    num_frames = runner.num_frames
    if num_frames % 4 != 1:
        num_frames = (num_frames + 2) // 4 * 4 + 1

    runner.model_high.pipe.scheduler.set_timesteps(runner.sample_steps)
    runner.model_low.pipe.scheduler.set_timesteps(runner.sample_steps)

    latents = runner.model_high.pipe.generate_noise(
        (1, 16, (num_frames - 1) // 4 + 1, runner.height // 8, runner.width // 8),
        seed=runner.base_seed,
    ).to(dtype=runner.torch_dtype, device=runner.high_device)

    control_camera_video = plucker_embedding[0].to(
        dtype=runner.torch_dtype,
        device=runner.high_device,
    ).permute([3, 0, 1, 2]).unsqueeze(0)
    control_camera_latents = torch.concat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
            control_camera_video[:, :, 1:],
        ],
        dim=2,
    ).transpose(1, 2)
    bsz, frames_local, channels, height_local, width_local = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(
        bsz, frames_local // 4, 4, channels, height_local, width_local
    ).transpose(2, 3)
    control_camera_latents = control_camera_latents.contiguous().view(
        bsz, frames_local // 4, channels * 4, height_local, width_local
    ).transpose(1, 2)

    final_prediction = None
    for progress_id, _ in enumerate(range(runner.sample_steps)):
        step_t = runner.model_high.pipe.scheduler.timesteps[progress_id]
        current_model = (
            runner.model_high if step_t.item() > runner.timestep_boundary else runner.model_low
        )
        current_device = (
            runner.high_device if current_model is runner.model_high else runner.low_device
        )
        t = step_t.unsqueeze(0).to(dtype=runner.torch_dtype, device=current_device)
        latents = latents.to(current_device)

        current_control = control_camera_latents.to(
            device=current_device,
            dtype=runner.torch_dtype,
        )
        current_y = y.to(dtype=runner.torch_dtype, device=current_device)
        current_ctx_pos = context_pos.to(dtype=runner.torch_dtype, device=current_device)

        noise_pred_posi, prediction = current_model.joint_forward(
            latents,
            timestep=t,
            context=current_ctx_pos,
            y=current_y,
            use_gradient_checkpointing=False,
            camera_token=None,
            control_camera_latents_input=current_control,
            uncond=False,
            return_prediction=progress_id == runner.sample_steps - 1,
        )

        if runner.cfg_scale != 1.0 and context_neg is not None:
            current_ctx_neg = context_neg.to(dtype=runner.torch_dtype, device=current_device)
            noise_pred_nega, _ = current_model.joint_forward(
                latents,
                timestep=t,
                context=current_ctx_neg,
                y=current_y,
                use_gradient_checkpointing=False,
                camera_token=None,
                control_camera_latents_input=current_control,
                uncond=False,
            )
            noise_pred = noise_pred_nega + runner.cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        latents = current_model.pipe.scheduler.step(
            noise_pred,
            current_model.pipe.scheduler.timesteps[progress_id],
            latents,
        )
        final_prediction = prediction

    return latents, final_prediction


def _reference_generate_video(
    runner,
    *,
    image: Image.Image,
    prompt: str,
    neg_prompt: str,
    camera_params,
    using_scale: bool = True,
):
    ensure_fantasy_world_runtime()
    from worldfoundry.synthesis.visual_generation.fantasy_world import fantasy_world_utils as fw_utils

    neg_prompt = neg_prompt or ""
    with torch.no_grad():
        input_image = image.convert("RGB")
        input_image_pt = torch.tensor(
            np.array(input_image) / 255,
            dtype=torch.float32,
            device=runner.moge_device,
        ).permute(2, 0, 1)
        output = runner.moge.infer(input_image_pt)
        moge = {key: value.cpu().contiguous() for key, value in output.items()}

        intrinsics = []
        extrinsics = []
        for camera in camera_params:
            intrinsics.append(fw_utils.get_intrinsic_matrix(camera))
            extrinsics.append(camera.w2c_mat)
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
        extrinsics_4x4 = extrinsics.unsqueeze(0)

        if using_scale:
            first_intrinsic = intrinsics[0, :, :].unsqueeze(0)
            first_extrinsic = extrinsics[0, :3, :].unsqueeze(0)
            first_moge_world, first_moge_mask = fw_utils.batch_depth_to_world(
                prediction=moge,
                extrinsics=first_extrinsic,
                intrinsics=first_intrinsic,
            )
            extrinsics = fw_utils.normalize_scene(
                extrinsics=extrinsics_4x4[:, :, :3, :],
                first_moge_world=first_moge_world.unsqueeze(0),
                first_moge_mask=first_moge_mask.unsqueeze(0),
            ).squeeze(0)

        pose_enc = runner._extri_intri_to_pose_encoding(
            extrinsics.unsqueeze(0),
            intrinsics.unsqueeze(0),
            [runner.height, runner.width],
            pose_encoding_type="absT_quaR_FoV",
        ).squeeze(0)
        plucker_embedding = runner.pose_processor.get_plucker_embedding_direct_from_cam_params(
            pose_enc.unsqueeze(0),
            image_size=(runner.height, runner.width),
        ).to(runner.high_device, runner.torch_dtype)

        inputs_shared, inputs_posi, inputs_nega = runner.model_high.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=runner.base_seed,
            tiled=True,
            input_image=input_image,
            end_image=None,
            height=runner.height,
            width=runner.width,
            return_condition=True,
        )
        ctx_pos, ctx_neg = inputs_posi["context"], inputs_nega["context"]
        y = inputs_shared["y"]

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=runner.torch_dtype)
        if runner.device.startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        latent_video, prediction = _reference_generate_video_with_dual_models(
            runner,
            context_pos=ctx_pos,
            context_neg=ctx_neg,
            y=y,
            plucker_embedding=plucker_embedding,
        )
        latent_video = latent_video.to(runner.high_device)
        frames = runner.model_high.pipe.vae.decode(
            latent_video,
            device=runner.high_device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )

    video = frames.squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu()
    video = (video + 1.0) / 2.0
    video = (video * 255.0).clamp(0, 255)
    return video.numpy().astype(np.uint8), prediction


if not _should_run():
    print("skip fantasy world wan22 official parity test: set RUN_FANTASYWORLD_WAN22_PARITY=1")
else:
    repo_root = _repo_root()
    model_path = _resolve_required_path(
        "FANTASYWORLD_WAN22_MODEL_PATH",
        repo_root / "cache/hfd/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera",
    )
    base_path = _resolve_required_path(
        "FANTASYWORLD_WAN22_BASE_PATH",
        repo_root / "cache/hfd/Wan2.2-Fun-A14B-Control-Camera",
    )
    lora_path = _resolve_required_path(
        "FANTASYWORLD_WAN22_LORA_PATH",
        repo_root / "cache/hfd/Wan2.2-Fun-Reward-LoRAs",
    )
    moge_pretrained = _resolve_required_path(
        "FANTASYWORLD_WAN22_MOGE_PRETRAINED",
        repo_root / "cache/hfd/moge-2-vitl-normal",
    )
    image_path = _resolve_required_path(
        "FANTASYWORLD_WAN22_IMAGE_PATH",
        repo_root / "worldfoundry/data/test_cases/test_image_case1/ref_image.png",
    )

    required_paths = {
        "model_path": model_path,
        "base_path": base_path,
        "lora_path": lora_path,
        "moge_pretrained": moge_pretrained,
        "image_path": image_path,
    }
    missing = [name for name, value in required_paths.items() if value is None]
    if missing:
        print(f"skip fantasy world wan22 official parity test: missing {missing}")
    else:
        device_layout = _select_devices()
        if device_layout is None:
            print("skip fantasy world wan22 official parity test: need two mostly free CUDA GPUs")
        else:
            from worldfoundry.synthesis.visual_generation.fantasy_world.wan22_runner import (
                build_wan22_runner,
            )

            frames = int(os.environ.get("FANTASYWORLD_WAN22_PARITY_FRAMES", "81"))
            height = int(os.environ.get("FANTASYWORLD_WAN22_PARITY_HEIGHT", "480"))
            width = int(os.environ.get("FANTASYWORLD_WAN22_PARITY_WIDTH", "832"))
            sample_steps = int(os.environ.get("FANTASYWORLD_WAN22_PARITY_SAMPLE_STEPS", "2"))
            base_seed = int(os.environ.get("FANTASYWORLD_WAN22_PARITY_SEED", "123"))
            high_device, low_device, moge_device = device_layout

            K, camera_params = _build_camera_inputs(frames=frames, height=height, width=width)
            image = Image.open(image_path).convert("RGB")
            prompt = "A cinematic fantasy landscape with gentle forward camera motion."

            high_ckpt = model_path / "high_noise_model.pth"
            low_ckpt = model_path / "low_noise_model.pth"
            moge_checkpoint = (
                moge_pretrained
                if moge_pretrained.is_file()
                else moge_pretrained / "model.pt"
            )
            runner = build_wan22_runner(
                base_dir=str(base_path),
                lora_dir=str(lora_path),
                model_ckpt_high=str(high_ckpt),
                model_ckpt_low=str(low_ckpt),
                moge_pretrained=str(moge_checkpoint),
                base_seed=base_seed,
                sample_steps=sample_steps,
                cfg_scale=5.0,
                timestep_boundary=900,
                frames=frames,
                fps=16,
                height=height,
                width=width,
                device=high_device,
                high_model_device=high_device,
                low_model_device=low_device,
                moge_device=moge_device,
            )

            frames_ref, pred_ref = _reference_generate_video(
                runner,
                image=image,
                prompt=prompt,
                neg_prompt="",
                camera_params=camera_params,
                using_scale=True,
            )
            frames_int, pred_int = runner.generate_video(
                image=image,
                end_image=None,
                prompt=prompt,
                neg_prompt="",
                camera_params=camera_params,
                using_scale=True,
            )

            assert np.array_equal(frames_ref, frames_int), "FantasyWorld Wan2.2 frames diverged from official reference logic."
            for key in ["depth", "depth_conf"]:
                ref_val = pred_ref[key].detach().to(torch.float32).cpu()
                int_val = pred_int[key].detach().to(torch.float32).cpu()
                assert torch.equal(ref_val, int_val), f"FantasyWorld Wan2.2 `{key}` diverged from official reference logic."

            print(
                "fantasy world wan22 official parity passed",
                high_device,
                low_device,
                moge_device,
                K.shape,
            )
