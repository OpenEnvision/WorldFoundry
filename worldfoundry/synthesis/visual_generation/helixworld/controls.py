"""HelixWorld latent-step action plans and calibrated camera conditions.

Adapted from NoizAI/HelixWorld's preview controls (Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.media.processing.image_utils import center_crop_resize_geometry

from .modeling import VideoControlCondition

_NAVIGATION = {
    "W": (1, 0, 0, 0), "S": (-1, 0, 0, 0), "A": (0, -1, 0, 0), "D": (0, 1, 0, 0),
    "left": (0, 0, -1, 0), "right": (0, 0, 1, 0), "up": (0, 0, 0, 1),
    "down": (0, 0, 0, -1), "stop": (0, 0, 0, 0),
}
_DIRECTION_IDS = {(0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, 1): 3, (0, -1): 4,
                  (1, 1): 5, (1, -1): 6, (-1, 1): 7, (-1, -1): 8}


def navigation(action: str) -> tuple[int, int, int, int]:
    motion = np.zeros(4, dtype=np.int64)
    for key in action.split("+"):
        key = key.strip()
        key = key.upper() if key.upper() in {"W", "A", "S", "D"} else key.lower()
        if key not in _NAVIGATION:
            raise ValueError(f"unsupported HelixWorld action: {key!r}")
        motion += _NAVIGATION[key]
    return tuple(int(value) for value in np.sign(motion))


def action_id(action: str) -> int:
    forward, right, yaw, pitch = navigation(action)
    return _DIRECTION_IDS[forward, right] * 9 + _DIRECTION_IDS[yaw, pitch]


def expand_action_plan(spec: str, transition_count: int) -> list[str]:
    segments = [segment.strip() for segment in spec.split(",") if segment.strip()]
    if not segments:
        raise ValueError("HelixWorld requires a non-empty action plan")
    result: list[str] = []
    for index, segment in enumerate(segments):
        if ":" in segment:
            action, duration = segment.rsplit(":", 1)
            count = int(duration)
        elif index == len(segments) - 1:
            action, count = segment, transition_count - len(result)
        else:
            raise ValueError("only the final action may omit its latent-step duration")
        navigation(action)
        if count <= 0:
            raise ValueError("action durations must be positive")
        result.extend([action] * count)
    if len(result) != transition_count:
        raise ValueError(f"action durations total {len(result)}, expected {transition_count} latent transitions")
    return result


def camera_poses(actions: list[str], perspective: str) -> torch.Tensor:
    speed, angle = 0.08, np.deg2rad(3.0)
    if perspective == "first_person":
        pose = np.eye(4)
        poses = [pose.copy()]
        for action in actions:
            forward, right, yaw, pitch = navigation(action)
            cosine, sine = np.cos(yaw * angle), np.sin(yaw * angle)
            pose[:3, :3] = pose[:3, :3] @ np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
            cosine, sine = np.cos(pitch * angle), np.sin(pitch * angle)
            pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
            pose[:3, 3] += pose[:3, :3] @ np.array([right * speed, 0, forward * speed])
            poses.append(pose.copy())
    elif perspective == "third_person":
        azimuth, elevation = np.pi, 0.0
        character = np.zeros(3)
        radius, orbit_height = speed / angle, 0.3

        def orbit_pose():
            position = character + np.array([
                radius * np.cos(elevation) * np.sin(azimuth),
                orbit_height + radius * np.sin(elevation),
                radius * np.cos(elevation) * np.cos(azimuth),
            ])
            forward = character + np.array([0, orbit_height * 0.5, 0]) - position
            forward /= np.linalg.norm(forward) + 1e-8
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            right /= np.linalg.norm(right) + 1e-8
            pose = np.eye(4)
            pose[:3, :3] = np.stack((right, np.cross(right, forward), forward), axis=-1)
            pose[:3, 3] = position
            return pose

        poses = [orbit_pose()]
        for action in actions:
            forward, right, yaw, pitch = navigation(action)
            azimuth -= yaw * angle
            elevation = np.clip(elevation - pitch * angle, np.deg2rad(-60), np.deg2rad(60))
            character += np.array([-right * speed, 0, forward * speed])
            poses.append(orbit_pose())
    else:
        raise ValueError("perspective must be 'first_person' or 'third_person'")
    return torch.from_numpy(np.stack(poses)).float()


def _source_size(image) -> tuple[int, int]:
    if isinstance(image, (str, Path)):
        with Image.open(Path(image).expanduser()) as source:
            return source.height, source.width
    if isinstance(image, Image.Image):
        return image.height, image.width
    if isinstance(image, torch.Tensor):
        return tuple(image.shape[-3:-1] if image.shape[-1] in (1, 3, 4) else image.shape[-2:])
    raise TypeError("HelixWorld requires a path, PIL image, or tensor first frame")


def prepare_video_control(request, *, device, dtype) -> VideoControlCondition:
    if request.batch_size != 1:
        raise ValueError("HelixWorld preview supports one sample per request")
    latent_frames = (request.num_frames - 1) // 8 + 1
    if request.num_frames < 121 or request.num_frames % 8 != 1 or latent_frames % 4:
        raise ValueError("HelixWorld preview supports 121, 153, 185, ... frames")
    if request.height % 32 or request.width % 32:
        raise ValueError("HelixWorld height and width must be divisible by 32")
    tokens_per_frame = (request.height // 32) * (request.width // 32)
    control_path = request.inputs.get("control_path")
    if control_path is not None:
        payload = torch.load(control_path, map_location="cpu", weights_only=True)
        ids = payload["action_ids"].reshape(1, -1)
        if ids.shape[1] != latent_frames or int(ids[0, 0]) != 0 or bool(((ids < 0) | (ids > 80)).any()):
            raise ValueError("control action IDs must match latent frames, start at 0, and lie in [0, 80]")
        control = VideoControlCondition(
            camera_intrinsics=payload["camera_intrinsics"].to(device=device, dtype=dtype),
            camera_w2c=payload["camera_w2c"].to(device=device, dtype=dtype),
            camera_valid_mask=payload["camera_valid_mask"].to(device=device, dtype=torch.bool),
            action_ids=ids.repeat_interleave(tokens_per_frame, dim=1).to(device),
            action_valid_mask=payload["action_valid_mask"].to(device=device, dtype=torch.bool),
        )
    else:
        actions = expand_action_plan(str(request.inputs.get("actions", "W")), latent_frames - 1)
        poses = camera_poses(actions, str(request.inputs.get("perspective", "first_person")))
        # The released input path uses first-frame-relative camera coordinates.
        w2c = torch.linalg.inv(poses)
        relative = w2c @ torch.linalg.inv(w2c[0])
        relative[0] = torch.eye(4)
        centers = torch.linalg.solve(relative[:, :3, :3], -relative[:, :3, 3, None]).squeeze(-1)
        step = torch.quantile(torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1), 0.75)
        radius = torch.linalg.vector_norm(centers, dim=-1).max()
        scale = min(1.0, 0.03 / float(step) if step > 1e-8 else 1.0, 1.5 / float(radius) if radius > 1e-8 else 1.0)
        relative[:, :3, 3] *= scale
        image = request.inputs.get("images", request.inputs.get("image"))
        if isinstance(image, (list, tuple)):
            image = image[0] if len(image) == 1 else None
        source_height, source_width = _source_size(image)
        resized_height, resized_width, top, left = center_crop_resize_geometry(
            source_height, source_width, request.height, request.width,
        )
        intrinsics = torch.tensor([
            [969.6969696969696 * resized_width / 1920 / request.width, 0, (resized_width / 2 - left) / request.width],
            [0, 969.6969696969696 * resized_height / 1080 / request.height, (resized_height / 2 - top) / request.height],
            [0, 0, 1],
        ])
        tokens = latent_frames * tokens_per_frame
        control = VideoControlCondition(
            camera_intrinsics=intrinsics.expand(request.batch_size, tokens, -1, -1).to(device=device, dtype=dtype),
            camera_w2c=relative.repeat_interleave(tokens_per_frame, dim=0).unsqueeze(0).expand(request.batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            camera_valid_mask=torch.ones(request.batch_size, tokens, device=device, dtype=torch.bool),
            action_ids=torch.tensor([0, *(action_id(action) for action in actions)], device=device).repeat_interleave(tokens_per_frame).expand(request.batch_size, -1),
            action_valid_mask=torch.ones(request.batch_size, tokens, device=device, dtype=torch.bool),
        )
    expected = (request.batch_size, latent_frames * tokens_per_frame)
    if control.camera_w2c.shape != (*expected, 4, 4) or control.camera_intrinsics.shape != (*expected, 3, 3):
        raise ValueError(f"camera controls must match the video batch/token grid {expected}")
    if control.camera_valid_mask.shape != expected or control.action_valid_mask.shape != expected:
        raise ValueError("camera and action validity masks must match the video tokens")
    return control.with_projective_matrices()
