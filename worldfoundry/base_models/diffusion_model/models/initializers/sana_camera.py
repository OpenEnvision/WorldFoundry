"""SANA-WM camera-action rollout.

SANA-WM assigns a/d to yaw and j/l to strafe and coasts held-key velocity
after release, unlike the shared WASD convention in
:mod:`worldfoundry.core.geometry.trajectory`. The initializer rolls out OpenCV
camera-to-world poses from this model-specific action DSL.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _rotation_x(theta: float) -> np.ndarray:
    """Right-handed rotation about +X (pitch); used by WASD rollout, not by Plücker encoding."""
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    """Right-handed rotation about +Y (yaw)."""
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)


_SANA_WM_NAMED_ACTIONS = {
    "forward": "w",
    "back": "s",
    "backward": "s",
    "left": "j",
    "right": "l",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "a",
    "camera_left": "a",
    "camera_r": "d",
    "camera_right": "d",
}


def rollout_sana_wm_camera_actions(
    actions: str | Sequence[str],
    *,
    num_frames: int | None = None,
    translation_step: float = 0.025,
    rotation_step_degrees: float = 0.6,
    pitch_limit_degrees: float = 60.0,
) -> np.ndarray:
    """Roll out the current official SANA-WM action DSL as OpenCV c2w poses.

    SANA-WM assigns ``a/d`` to yaw and ``j/l`` to strafe, unlike the older
    shared WASD convention. Its held-key velocity coasts after a key release.
    Keep this model-specific contract separate from the generic rollout.
    """
    allowed = frozenset("wasdijkl")
    if isinstance(actions, str):
        cleaned = "".join(actions.replace("，", ",").split())
        per_frame = []
        if cleaned and "-" not in cleaned:
            per_frame = [segment.lower() for segment in cleaned.split(",") if segment]
        else:
            for segment in cleaned.split(",") if cleaned else []:
                if "-" not in segment:
                    raise ValueError(f"Invalid SANA-WM action segment {segment!r}; expected '<keys>-<duration>'")
                keys, duration = segment.rsplit("-", 1)
                if not duration.isdigit() or int(duration) <= 0:
                    raise ValueError(f"SANA-WM action duration must be positive: {segment!r}")
                per_frame.extend([keys.lower()] * int(duration))
    else:
        per_frame = [
            _SANA_WM_NAMED_ACTIONS.get(str(action).strip().lower(), str(action).strip().lower()) for action in actions
        ]
    if num_frames is not None:
        count = max(int(num_frames) - 1, 0)
        per_frame = (per_frame + ["none"] * count)[:count]

    rotation_step = np.deg2rad(float(rotation_step_degrees))
    pitch_limit = np.deg2rad(float(pitch_limit_degrees))
    pose = np.eye(4, dtype=np.float64)
    poses = [pose.copy()]
    velocity = np.zeros(4, dtype=np.float64)  # forward, strafe, yaw, pitch
    previous_controls: set[str] = set()
    pitch = 0.0
    for raw_keys in per_frame:
        keys = "" if raw_keys in {"", "none", "noop"} else raw_keys
        invalid = sorted(set(keys) - allowed)
        if invalid:
            raise ValueError(f"unknown SANA-WM camera action keys {invalid}")
        controls = set(keys)
        target = np.array(
            [
                (("w" in controls) - ("s" in controls)) * translation_step,
                (("l" in controls) - ("j" in controls)) * translation_step,
                (("d" in controls) - ("a" in controls)) * rotation_step,
                (("i" in controls) - ("k" in controls)) * rotation_step,
            ],
            dtype=np.float64,
        )
        if controls - previous_controls:
            velocity = target
        else:
            tau = np.where(np.abs(target) > 1e-12, 0.45, 1.0)
            velocity += (1.0 - np.exp(-1.0 / (16.0 * tau))) * (target - velocity)
        previous_controls = controls

        next_pitch = np.clip(pitch + velocity[3], -pitch_limit, pitch_limit)
        pitch_delta = next_pitch - pitch
        pitch = next_pitch
        rotation = _rotation_y(velocity[2]) @ pose[:3, :3] @ _rotation_x(pitch_delta)
        forward = rotation[:, 2].copy()
        right = rotation[:, 0].copy()
        forward[1] = right[1] = 0.0
        forward /= float(np.linalg.norm(forward)) + 1e-6
        right /= float(np.linalg.norm(right)) + 1e-6
        next_pose = np.eye(4, dtype=np.float64)
        next_pose[:3, :3] = rotation
        next_pose[:3, 3] = pose[:3, 3] + forward * velocity[0] + right * velocity[1]
        pose = next_pose
        poses.append(pose.copy())
    return np.stack(poses).astype(np.float32)
