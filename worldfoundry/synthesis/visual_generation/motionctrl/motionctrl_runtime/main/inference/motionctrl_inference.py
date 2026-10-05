from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim import DDIMSampler
from worldfoundry.core.model_loading import load_torch_state_dict

DEFAULT_NEGATIVE_PROMPT = (
    "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
    "sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, "
    "disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, "
    "floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
)
POST_PROMPT = (
    "Ultra-detail, masterpiece, best quality, cinematic lighting, 8k uhd, dslr, "
    "soft lighting, film grain, Fujifilm XT3"
)


def _normalize_checkpoint_state(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "module" in checkpoint and isinstance(checkpoint["module"], dict):
        return OrderedDict((key[16:] if key.startswith("module.") else key, value) for key, value in checkpoint["module"].items())
    return checkpoint


def load_model_checkpoint(model: Any, ckpt: str | Path, adapter_ckpt: str | Path | None = None) -> Any:
    checkpoint = _normalize_checkpoint_state(load_torch_state_dict(ckpt, map_location="cpu"))
    incompatible = model.load_state_dict(checkpoint, strict=False)
    missing_text = [name for name in incompatible.missing_keys if name.startswith("cond_stage_model.")]
    if missing_text:
        raise RuntimeError(f"MotionCtrl checkpoint is missing text encoder weights: {missing_text}")
    if adapter_ckpt is not None:
        adapter_state = _normalize_checkpoint_state(load_torch_state_dict(adapter_ckpt, map_location="cpu"))
        model.adapter.load_state_dict(adapter_state, strict=True)
    return model


def _resolve_traj_file(cond_dir: str | Path, traj: str) -> Path:
    base_path = Path(cond_dir) / "trajectories" / traj
    candidates = [base_path.with_suffix(".npy"), base_path.with_suffix(".txt")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MotionCtrl trajectory fixture is missing for {traj}: tried {candidates}")


def _trajectory_from_control_points(traj_file: Path, *, frames: int = 16, height: int = 256, width: int = 256) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for line in traj_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        x_text, y_text = stripped.replace(",", " ").split()[:2]
        points.append((float(x_text), float(y_text)))
    if len(points) < 2:
        raise ValueError(f"MotionCtrl trajectory fixture must contain at least two points: {traj_file}")

    source = np.asarray(points, dtype=np.float32)
    sample_positions = np.linspace(0, len(source) - 1, num=frames, dtype=np.float32)
    resampled = np.stack(
        [
            np.interp(sample_positions, np.arange(len(source), dtype=np.float32), source[:, axis])
            for axis in range(2)
        ],
        axis=1,
    ).astype(np.float32)
    origin = resampled[0]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sigma = float(max(height, width)) / 8.0
    traj = np.zeros((frames, height, width, 2), dtype=np.float32)

    for frame_index, point in enumerate(resampled):
        if frame_index == 0:
            continue
        dx = (point[0] - origin[0]) / float(width)
        dy = (point[1] - origin[1]) / float(height)
        weight = np.exp(-(((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / (2.0 * sigma * sigma))).astype(np.float32)
        traj[frame_index, :, :, 0] = dx * weight
        traj[frame_index, :, :, 1] = dy * weight
    return traj


def _load_traj_array(traj_file: Path) -> np.ndarray:
    if traj_file.suffix == ".npy":
        return np.load(traj_file).astype(np.float32, copy=False)
    if traj_file.suffix == ".txt":
        return _trajectory_from_control_points(traj_file)
    raise ValueError(f"Unsupported MotionCtrl trajectory format: {traj_file}")


def load_trajs(cond_dir: str | Path, trajs: Iterable[str]) -> tuple[list[torch.Tensor], list[str]]:
    data_list: list[torch.Tensor] = []
    traj_names: list[str] = []
    for traj in trajs:
        traj_file = _resolve_traj_file(cond_dir, traj)
        traj_names.append(traj_file.stem)
        data_list.append(torch.from_numpy(_load_traj_array(traj_file)).permute(3, 0, 1, 2).float())
    return data_list, traj_names


def load_camera_pose(cond_dir: str | Path, camera_poses: Iterable[str]) -> tuple[list[torch.Tensor], list[str]]:
    data_list: list[torch.Tensor] = []
    pose_names: list[str] = []
    for camera_pose in camera_poses:
        pose_file = Path(cond_dir) / "camera_poses" / f"{camera_pose}.json"
        pose_names.append(camera_pose.replace("test_camera_", ""))
        pose = np.asarray(json.loads(pose_file.read_text(encoding="utf-8")), dtype=np.float32)
        data_list.append(torch.as_tensor(pose).float())
    return data_list, pose_names


def _print_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    total_count = data.numel()
    if finite_count == 0:
        print(f"{label}: shape={tuple(data.shape)} finite=0/{total_count}", flush=True)
        return
    values = data[finite]
    print(
        f"{label}: shape={tuple(data.shape)} finite={finite_count}/{total_count} "
        f"min={float(values.min().item()):.6g} max={float(values.max().item()):.6g} "
        f"mean={float(values.mean().item()):.6g} std={float(values.std(unbiased=False).item()):.6g}",
        flush=True,
    )


def motionctrl_sample(
    model: Any,
    prompts: str | list[str],
    noise_shape: list[int],
    camera_poses: torch.Tensor | None = None,
    trajs: torch.Tensor | None = None,
    n_samples: int = 1,
    unconditional_guidance_scale: float = 1.0,
    unconditional_guidance_scale_temporal: float | None = None,
    ddim_steps: int = 50,
    ddim_eta: float = 1.0,
    **kwargs: Any,
) -> torch.Tensor:
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    prompt_list = [f"{prompt}, {POST_PROMPT}" for prompt in prompt_list]

    cond = model.get_learned_conditioning(prompt_list)
    pose_emb = camera_poses[..., None] if camera_poses is not None else None
    traj_features = model.get_traj_features(trajs) if trajs is not None else None
    if unconditional_guidance_scale != 1.0:
        uc_prompt = batch_size * [DEFAULT_NEGATIVE_PROMPT]
        uc = model.get_learned_conditioning(uc_prompt)
        un_motion = model.get_traj_features(torch.zeros_like(trajs)) if traj_features is not None else None
        uc = {"features_adapter": un_motion, "uc": uc}
    else:
        uc = None

    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            temporal_length=noise_shape[2],
            conditional_guidance_scale_temporal=unconditional_guidance_scale_temporal,
            features_adapter=traj_features,
            pose_emb=pose_emb,
            **kwargs,
        )
        _print_tensor_stats("MotionCtrl latent samples after DDIM", samples)
        batch_images = model.decode_first_stage(samples)
        _print_tensor_stats("MotionCtrl decoded samples after VAE", batch_images)
        batch_variants.append(batch_images)
    return torch.stack(batch_variants).permute(1, 0, 2, 3, 4, 5)


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "load_camera_pose",
    "load_model_checkpoint",
    "load_trajs",
    "motionctrl_sample",
]
