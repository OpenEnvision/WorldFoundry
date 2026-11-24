"""Small, standalone WASD/IJKL camera DSL used by UCPE inference."""

from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_TRANSLATION_SPEED = 0.025
DEFAULT_ROTATION_SPEED_DEG = 0.6
DEFAULT_PITCH_SPEED_DEG = 0.2
DEFAULT_PITCH_LIMIT_DEG = 60.0
ALLOWED_ACTION_KEYS = frozenset("wsadikjl")


def parse_action_string(action: str) -> list[list[str]]:
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    frames: list[list[str]] = []
    for segment in cleaned.split(","):
        if "-" not in segment:
            raise ValueError(f"Invalid action segment {segment!r}; expected '<keys>-<duration>'")
        keys_part, duration = segment.rsplit("-", 1)
        if not duration.isdigit() or int(duration) <= 0:
            raise ValueError(f"Invalid action duration in {segment!r}")
        keys = [] if keys_part.lower() == "none" else sorted(set(keys_part.lower()))
        invalid = sorted(set(keys) - ALLOWED_ACTION_KEYS)
        if invalid:
            raise ValueError(f"Unknown action keys {invalid}; allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}")
        frames.extend([keys] * int(duration))
    return frames


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def action_string_to_c2w(
    frames: list[list[str]],
    *,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_speed_deg: float,
    pitch_limit_deg: float,
    fps: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pitch = 0.0
    velocity = np.zeros(4, dtype=np.float64)
    poses = [pose.copy()]
    previous: set[str] = set()
    dt = 1.0 / fps
    for keys in frames:
        current = set(keys)
        target = np.array(
            [
                float("w" in current) - float("s" in current),
                float("d" in current) - float("a" in current),
                float("l" in current) - float("j" in current),
                float("i" in current) - float("k" in current),
            ]
        )
        target *= np.array(
            [translation_speed, translation_speed, math.radians(rotation_speed_deg), math.radians(pitch_speed_deg)]
        )
        if current - previous:
            velocity = target
        else:
            velocity += (target - velocity) * (1.0 - math.exp(-dt / (0.45 if np.any(target) else 1.0)))
        previous = current
        new_pitch = np.clip(pitch + velocity[3], -math.radians(pitch_limit_deg), math.radians(pitch_limit_deg))
        pitch_step = new_pitch - pitch
        pitch = new_pitch
        rotation = _rot_y(velocity[2]) @ pose[:3, :3] @ _rot_x(pitch_step)
        forward = rotation[:, 2].copy()
        forward[1] = 0
        right = rotation[:, 0].copy()
        right[1] = 0
        forward /= max(np.linalg.norm(forward), 1e-6)
        right /= max(np.linalg.norm(right), 1e-6)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = poses[-1][:3, 3] + forward * velocity[0] + right * velocity[1]
        poses.append(pose.copy())
    return np.stack(poses).astype(np.float32)


def default_k_pix(width: int, height: int, fov_deg: float) -> torch.Tensor:
    focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return torch.tensor([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def build_action_pt_from_string(
    action: str,
    *,
    num_frames: int,
    image_width: int,
    image_height: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    fov_deg: float,
    fps: float,
) -> dict[str, torch.Tensor | str]:
    keys = parse_action_string(action)
    c2ws = action_string_to_c2w(
        keys,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_speed_deg=DEFAULT_PITCH_SPEED_DEG,
        pitch_limit_deg=pitch_limit_deg,
        fps=fps,
    )
    if len(c2ws) < num_frames:
        c2ws = np.concatenate([c2ws, np.repeat(c2ws[-1:], num_frames - len(c2ws), axis=0)])
    c2ws = c2ws[:num_frames]
    return {
        "c2ws_raw": torch.from_numpy(c2ws),
        "K_pix": default_k_pix(image_width, image_height, fov_deg),
        "schema": "wasd_ijkl_ucpe_v3",
    }
