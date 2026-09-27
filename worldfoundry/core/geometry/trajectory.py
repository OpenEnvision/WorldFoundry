"""Reusable camera-trajectory parsing and action-token conversion.

World models consume camera motion in several incompatible encodings:
held WASD/IJKL keys, compact ``w*4,j*2`` strings, Wan 19-D pose rows,
GEN3C named paths, 6-D AdaLN deltas, Echo 12-D RT, and 81-class
WorldPlay labels. This module converts between those encodings without
importing a model family.

Groups:

- **Rollout** — :func:`rollout_wasd_camera_actions` and
  :func:`rollout_sana_wm_camera_actions` (held keys → c2w),
  :func:`parse_camera_trajectory` / :func:`camera_trajectory_view_matrices`
  / :func:`camera_trajectory_tensors` (string → w2c + intrinsics).
- **Named / planar** — :func:`named_camera_trajectory_tensors` (GEN3C
  left/right/zoom/orbit), :func:`generate_planar_camera_coordinates`
  plus :func:`wan_camera_coordinates_to_plucker`.
- **Action tokens** — :func:`camera_poses_to_adaln_actions`,
  :func:`camera_poses_to_relative_rt12_actions`,
  :func:`discretize_camera_poses_to_actions`,
  :func:`camera_trajectory_action_labels`.

OpenCV convention throughout: +X right, +Y down, +Z forward.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from worldfoundry.core.geometry.transforms import rotation_matrix_to_euler_angles_zyx

_TRANSLATION_LABELS = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

_TRAJECTORY_ACTION_LABELS = {
    "w": (1, 0),
    "s": (2, 0),
    "a": (3, 0),
    "d": (4, 0),
    "j": (0, 2),
    "l": (0, 1),
    "i": (0, 3),
    "k": (0, 4),
}

# Studio / catalog tokens used by SANA-WM and other WASD world models.
_NAMED_WASD_ACTIONS = {
    "forward": "w",
    "back": "s",
    "backward": "s",
    "left": "a",
    "right": "d",
    "camera_up": "i",
    "camera_down": "k",
    "camera_l": "j",
    "camera_left": "j",
    "camera_r": "l",
    "camera_right": "l",
}

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

# ──────────────────────────────────────────────────────────────────────────
# Rollout — held WASD/IJKL keys or compact trajectory strings → poses
# ──────────────────────────────────────────────────────────────────────────


def _rotation_x(theta: float) -> np.ndarray:
    """Right-handed rotation about +X (pitch); used by WASD rollout, not by Plücker encoding."""
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    """Right-handed rotation about +Y (yaw)."""
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)


def rollout_wasd_camera_actions(
    actions: str | Sequence[str],
    *,
    num_frames: int | None = None,
    translation_step: float = 0.05,
    rotation_step_degrees: float = 1.2,
    pitch_limit_degrees: float = 85.0,
) -> np.ndarray:
    """Roll out OpenCV camera-to-world poses from held WASD/IJKL keys.

    A string uses ``<keys>-<duration>`` segments such as
    ``"w-10,iw-5,none-3"``.  A sequence provides one held-key string per
    output transition.  The returned trajectory includes the identity anchor.
    """

    allowed = frozenset("wasdijkl")
    if isinstance(actions, str):
        cleaned = "".join(actions.replace("，", ",").split())
        if cleaned and "-" not in cleaned:
            actions = [part for part in cleaned.split(",") if part]
        elif cleaned:
            per_frame: list[str] = []
            for segment in cleaned.split(","):
                if "-" not in segment:
                    raise ValueError(
                        f"Invalid camera action segment {segment!r}; expected '<keys>-<duration>'"
                    )
                keys, raw_duration = segment.rsplit("-", 1)
                if not raw_duration.isdigit() or int(raw_duration) <= 0:
                    raise ValueError(f"camera action duration must be positive: {segment!r}")
                per_frame.extend([keys] * int(raw_duration))
        else:
            per_frame = []
    if not isinstance(actions, str):
        per_frame = [
            _NAMED_WASD_ACTIONS.get(str(value).strip().lower(), str(value).strip().lower())
            for value in actions
        ]

    target_transitions = None if num_frames is None else max(int(num_frames) - 1, 0)
    if target_transitions is not None:
        per_frame = (per_frame + ["none"] * target_transitions)[:target_transitions]

    rotation_step = np.radians(float(rotation_step_degrees))
    pitch_limit = np.radians(float(pitch_limit_degrees))
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0
    for raw_keys in per_frame:
        keys = "" if raw_keys.lower() in {"", "none", "noop"} else raw_keys.lower()
        invalid = sorted(set(keys) - allowed)
        if invalid:
            raise ValueError(f"unknown camera action keys {invalid}; allowed: {''.join(sorted(allowed))}")
        held = set(keys)
        rotation = current[:3, :3]
        translation = current[:3, 3]
        pitch_delta = (rotation_step if "i" in held else 0.0) - (
            rotation_step if "k" in held else 0.0
        )
        if not -pitch_limit <= current_pitch + pitch_delta <= pitch_limit:
            pitch_delta = 0.0
        else:
            current_pitch += pitch_delta
        yaw_delta = (rotation_step if "l" in held else 0.0) - (
            rotation_step if "j" in held else 0.0
        )
        next_rotation = _rotation_y(yaw_delta) @ rotation @ _rotation_x(pitch_delta)
        forward = next_rotation[:, 2].copy()
        right = next_rotation[:, 0].copy()
        forward[1] = right[1] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1e-6)
        right /= max(float(np.linalg.norm(right)), 1e-6)
        movement = np.zeros(3, dtype=np.float64)
        movement += (("w" in held) - ("s" in held)) * float(translation_step) * forward
        movement += (("d" in held) - ("a" in held)) * float(translation_step) * right
        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = next_rotation
        current[:3, 3] = translation + movement
        poses.append(current.copy())
    return np.stack(poses).astype(np.float32)


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
            _SANA_WM_NAMED_ACTIONS.get(str(action).strip().lower(), str(action).strip().lower())
            for action in actions
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


def parse_camera_trajectory(
    trajectory: str,
    *,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
) -> list[dict[str, float]]:
    """Parse ``w*4,j*2`` camera controls into per-frame motion dicts.

    Each comma-separated segment is a direction and optional repeat
    (``w``, ``w*19``, ``left``, ``dn``). Directions include translation
    (``w/s/a/d/u/dn``) and rotation (``j/l/i/k`` plus named
    left/right/up/down). ``z`` is a no-op frame.

    Returns:
        One dict per output frame with a subset of ``forward``,
        ``right``, ``up``, ``yaw``, ``pitch`` in metres / radians.
    """

    rotation_step = np.radians(float(rotation_step_degrees))
    motions = {
        "w": {"forward": float(translation_step)},
        "s": {"forward": -float(translation_step)},
        "d": {"right": float(translation_step)},
        "a": {"right": -float(translation_step)},
        "u": {"up": float(translation_step)},
        "dn": {"up": -float(translation_step)},
        "j": {"yaw": -rotation_step},
        "l": {"yaw": rotation_step},
        "i": {"pitch": rotation_step},
        "k": {"pitch": -rotation_step},
        "left": {"yaw": -rotation_step},
        "right": {"yaw": rotation_step},
        "up": {"pitch": rotation_step},
        "down": {"pitch": -rotation_step},
        "z": {},
    }
    parsed: list[dict[str, float]] = []
    for raw_segment in str(trajectory).strip().split(","):
        segment = raw_segment.strip().lower()
        if not segment:
            continue
        match = re.fullmatch(r"([a-z]+)(?:\*(\d+))?", segment)
        if match is None:
            raise ValueError(f"Invalid camera trajectory segment {raw_segment!r}; expected e.g. 'w*19'")
        key, raw_count = match.groups()
        if key not in motions:
            raise ValueError(f"Unknown camera direction {key!r}; choices: {sorted(motions)}")
        count = int(raw_count or 1)
        if count < 0:
            raise ValueError(f"Camera trajectory count must be non-negative, got {count}")
        parsed.extend(dict(motions[key]) for _ in range(count))
    return parsed


def camera_trajectory_view_matrices(
    trajectory: str | Sequence[Mapping[str, float]],
    *,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
) -> np.ndarray:
    """Return OpenCV world-to-camera matrices, including the identity frame.

    Applies :func:`parse_camera_trajectory` (or a pre-parsed motion
    sequence) to an identity c2w, then inverts each pose so callers that
    expect w2c (Wan / GEN3C loaders) can use the result directly.

    Returns:
        ``[T, 4, 4]`` float32 world-to-camera matrices.
    """

    motions = (
        parse_camera_trajectory(
            trajectory,
            translation_step=translation_step,
            rotation_step_degrees=rotation_step_degrees,
        )
        if isinstance(trajectory, str)
        else [dict(item) for item in trajectory]
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    poses = [camera_to_world.copy()]
    for motion in motions:
        if "yaw" in motion:
            camera_to_world[:3, :3] = camera_to_world[:3, :3] @ _rotation_y(float(motion["yaw"]))
        if "pitch" in motion:
            camera_to_world[:3, :3] = camera_to_world[:3, :3] @ _rotation_x(float(motion["pitch"]))
        local_translation = np.asarray(
            [float(motion.get("right", 0.0)), -float(motion.get("up", 0.0)), float(motion.get("forward", 0.0))]
        )
        camera_to_world[:3, 3] += camera_to_world[:3, :3] @ local_translation
        poses.append(camera_to_world.copy())
    return np.stack([np.linalg.inv(pose) for pose in poses]).astype(np.float32)


def camera_trajectory_tensors(
    trajectory: str | Sequence[Mapping[str, float]],
    *,
    fx: float = 0.5050505,
    fy: float = 0.89786756,
    cx: float = 0.5,
    cy: float = 0.5,
    translation_step: float = 0.08,
    rotation_step_degrees: float = 3.0,
    device: Any = "cpu",
    dtype: Any = None,
):
    """Return batched view matrices and normalized camera intrinsics.

    Wraps :func:`camera_trajectory_view_matrices` and repeats a single
    pinhole intrinsic ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` across
    time. Defaults match the historical WorldPlay / SANA-WM cameras.

    Returns:
        ``(views, intrinsics)`` with shapes ``[1, T, 4, 4]`` and
        ``[1, T, 3, 3]`` on *device*.
    """

    import torch

    tensor_dtype = dtype or torch.float32
    view_matrices = camera_trajectory_view_matrices(
        trajectory,
        translation_step=translation_step,
        rotation_step_degrees=rotation_step_degrees,
    )
    intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.repeat(intrinsic[None], len(view_matrices), axis=0)
    return (
        torch.as_tensor(view_matrices, dtype=tensor_dtype, device=device).unsqueeze(0),
        torch.as_tensor(intrinsics, dtype=tensor_dtype, device=device).unsqueeze(0),
    )


# ──────────────────────────────────────────────────────────────────────────
# Named / planar paths — GEN3C left/right/zoom/orbit and Wan Plücker
# ──────────────────────────────────────────────────────────────────────────


def generate_planar_camera_coordinates(
    direction: str,
    length: int,
    *,
    speed: float = 1 / 54,
    origin: Sequence[float] = (
        0,
        0.532139961,
        0.946026558,
        0.5,
        0.5,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
    ),
) -> list[list[float]]:
    """Create the compact camera rows used by Wan camera-control checkpoints.

    Each row is the 19-D layout ``[id, fx, fy, cx, cy, 0, 0, *w2c_3x4]``.
    Planar directions translate the camera along world X (left/right)
    and Y (up/down) by *speed* per frame, starting from *origin*.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    supported = {
        "Left",
        "Right",
        "Up",
        "Down",
        "LeftUp",
        "LeftDown",
        "RightUp",
        "RightDown",
    }
    if direction not in supported:
        raise ValueError(f"unsupported camera direction {direction!r}")
    coordinates = [list(origin)]
    while len(coordinates) < length:
        row = coordinates[-1].copy()
        if "Left" in direction:
            row[9] += speed
        if "Right" in direction:
            row[9] -= speed
        if "Up" in direction:
            row[13] += speed
        if "Down" in direction:
            row[13] -= speed
        coordinates.append(row)
    return coordinates


def wan_camera_coordinates_to_plucker(
    coordinates: Sequence[Sequence[float]],
    *,
    width: int = 672,
    height: int = 384,
    original_width: int = 1280,
    original_height: int = 720,
    device: Any = "cpu",
):
    """Convert Wan 19-D camera rows to ``[T, H, W, 6]`` Plücker features.

    Rescales normalized fx/fy for letterboxed resize from
    ``original_width x original_height`` to ``width x height``, recenters
    poses so frame 0 is the identity, then calls
    :func:`worldfoundry.core.geometry.transforms.ray_condition`.
    """

    import torch

    from worldfoundry.core.geometry.transforms import ray_condition

    rows = np.asarray(coordinates, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 19:
        raise ValueError(f"camera coordinates must be [T,19], got {rows.shape}")
    intrinsics = rows[:, 1:5].copy()
    sample_ratio = float(width) / float(height)
    source_ratio = float(original_width) / float(original_height)
    if source_ratio > sample_ratio:
        intrinsics[:, 0] *= (height * source_ratio) / width
    else:
        intrinsics[:, 1] *= (width / source_ratio) / height
    intrinsics *= np.asarray((width, height, width, height), dtype=np.float32)

    world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None], len(rows), axis=0)
    world_to_camera[:, :3, :] = rows[:, 7:].reshape(-1, 3, 4)
    camera_to_world = np.linalg.inv(world_to_camera)
    target = np.eye(4, dtype=np.float32)
    relative_transform = target @ world_to_camera[0]
    relative_camera_to_world = np.stack(
        [target, *[relative_transform @ pose for pose in camera_to_world[1:]]]
    )
    k = torch.as_tensor(intrinsics, device=device).unsqueeze(0)
    c2w = torch.as_tensor(relative_camera_to_world, device=device).unsqueeze(0)
    return ray_condition(k, c2w, height, width, device=device)[0]


def named_camera_trajectory_tensors(
    trajectory: str,
    *,
    initial_world_to_camera: Any,
    initial_intrinsic: Any,
    num_frames: int,
    movement_distance: float = 0.3,
    camera_rotation: str = "center_facing",
    center_depth: float = 1.0,
    num_circles: int = 1,
):
    """Build the eight named GEN3C camera trajectories without a model runtime.

    The output is ``(world_to_camera, intrinsics)`` with batched shapes
    ``[1,T,4,4]`` and ``[1,T,3,3]``.  Keeping this geometry in ``core`` makes
    it reusable by camera-conditioned recipes instead of tying it to Cosmos.
    """

    import torch

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    trajectory = str(trajectory).strip().lower()
    supported = {
        "left",
        "right",
        "up",
        "down",
        "zoom_in",
        "zoom_out",
        "clockwise",
        "counterclockwise",
    }
    if trajectory not in supported:
        raise ValueError(f"unsupported named camera trajectory {trajectory!r}")
    if camera_rotation not in {"center_facing", "no_rotation", "trajectory_aligned"}:
        raise ValueError(
            "camera_rotation must be center_facing, no_rotation, or trajectory_aligned"
        )

    world_to_camera = torch.as_tensor(initial_world_to_camera)
    intrinsic = torch.as_tensor(initial_intrinsic, device=world_to_camera.device)
    if world_to_camera.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("initial camera matrices must have shapes [4,4] and [3,3]")
    dtype = world_to_camera.dtype
    device = world_to_camera.device
    steps = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
    positions = torch.zeros((num_frames, 3), device=device, dtype=dtype)
    distance = float(movement_distance) * float(center_depth)
    if trajectory in {"clockwise", "counterclockwise"}:
        direction = 1.0 if trajectory == "clockwise" else -1.0
        theta = steps * (2.0 * torch.pi * int(num_circles))
        positions[:, 0] = direction * distance * (torch.cos(theta) - 1.0)
        positions[:, 1] = distance * torch.sin(theta)
    else:
        axis, direction = {
            "left": (0, -1.0),
            "right": (0, 1.0),
            "up": (1, -1.0),
            "down": (1, 1.0),
            "zoom_in": (2, 1.0),
            "zoom_out": (2, -1.0),
        }[trajectory]
        positions[:, axis] = direction * distance * steps

    center = torch.tensor((0.0, 0.0, float(center_depth)), device=device, dtype=dtype)
    up_axis = torch.tensor((0.0, 1.0, 0.0), device=device, dtype=dtype)
    matrices = []
    for position in positions:
        if camera_rotation == "center_facing":
            target = center
        elif camera_rotation == "trajectory_aligned":
            target = center + position * 2.0
        else:
            target = center + position
        forward = torch.nn.functional.normalize(target - position, dim=0)
        right = torch.nn.functional.normalize(torch.linalg.cross(up_axis, forward), dim=0)
        up = torch.linalg.cross(forward, right)
        view = torch.eye(4, device=device, dtype=dtype)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = forward
        view[:3, 3] = -position
        matrices.append(view @ world_to_camera)
    views = torch.stack(matrices).unsqueeze(0)
    intrinsics = intrinsic.unsqueeze(0).unsqueeze(0).expand(1, num_frames, -1, -1).clone()
    return views, intrinsics


# ──────────────────────────────────────────────────────────────────────────
# Action tokens — AdaLN deltas, Echo RT12, WorldPlay 81-class labels
# ──────────────────────────────────────────────────────────────────────────


def camera_poses_to_adaln_actions(
    camera_to_world: Any,
    *,
    action_scale: str | Sequence[float],
    temporal_stride: int = 8,
):
    """Convert pixel-rate camera poses to latent-rate 6D AdaLN actions.

    The returned tensor has shape ``[B, T_latent, 6]`` and stores consecutive
    camera-local ``[tx, ty, tz, rx, ry, rz]`` deltas.  Frame zero is all zeros.
    This is the common representation used by AlayaWorld's action conditioner;
    keeping it in :mod:`worldfoundry.core` also avoids a SciPy runtime
    dependency for camera-conditioned world models.

    Args:
        camera_to_world: ``[F, 4, 4]`` or ``[B, F, 4, 4]`` camera-to-world
            matrices. Torch tensors are accepted and converted back to Torch.
        action_scale: Six positive normalization constants, either a
            comma-separated string or a sequence.
        temporal_stride: Pixel frames represented by one video latent frame.
    """

    import torch

    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    is_tensor = isinstance(camera_to_world, torch.Tensor)
    source_device = camera_to_world.device if is_tensor else None
    poses = camera_to_world.detach().cpu().numpy() if is_tensor else np.asarray(camera_to_world)
    if poses.ndim == 3:
        poses = poses[None]
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must be [F,4,4] or [B,F,4,4], got {poses.shape}")
    if poses.shape[1] < 1:
        raise ValueError("camera_to_world must contain at least one frame")

    raw_scale = action_scale.split(",") if isinstance(action_scale, str) else action_scale
    try:
        scale = np.asarray([float(value) for value in raw_scale], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("action_scale must contain six positive floats") from exc
    if scale.shape != (6,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"action_scale must contain six positive floats, got {action_scale!r}")

    result: list[np.ndarray] = []
    for batch_poses in poses.astype(np.float64, copy=False):
        latent_count = (len(batch_poses) - 1) // temporal_stride + 1
        sampled = batch_poses[np.minimum(np.arange(latent_count) * temporal_stride, len(batch_poses) - 1)]
        actions = np.zeros((latent_count, 6), dtype=np.float32)
        for index in range(1, latent_count):
            relative = np.linalg.inv(sampled[index - 1]) @ sampled[index]
            actions[index, :3] = relative[:3, 3]
            actions[index, 3:] = rotation_matrix_to_euler_angles_zyx(relative[:3, :3])
        result.append(actions / scale[None])

    stacked = np.stack(result)
    if is_tensor:
        return torch.as_tensor(stacked, dtype=torch.float32, device=source_device)
    return stacked


def camera_poses_to_relative_rt12_actions(
    camera_to_world: Any,
    *,
    temporal_stride: int = 4,
):
    """Convert pixel-rate camera poses to Echo-style relative 12D RT actions.

    Echo-Memory represents each latent frame as ``[tx, ty, tz, R.flatten()]``
    relative to the first frame of the current rollout chunk.  The returned
    value has shape ``[B, T_latent, 12]``.  Both NumPy arrays and Torch tensors
    are accepted; Torch input produces Torch output on the source device.
    """

    import torch

    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    is_tensor = isinstance(camera_to_world, torch.Tensor)
    source_device = camera_to_world.device if is_tensor else None
    poses = camera_to_world.detach().cpu().numpy() if is_tensor else np.asarray(camera_to_world)
    if poses.ndim == 3:
        poses = poses[None]
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must be [F,4,4] or [B,F,4,4], got {poses.shape}")
    if poses.shape[1] < 1:
        raise ValueError("camera_to_world must contain at least one frame")

    batches: list[np.ndarray] = []
    for batch_poses in poses.astype(np.float64, copy=False):
        latent_count = (len(batch_poses) - 1) // temporal_stride + 1
        indices = np.minimum(
            np.arange(latent_count, dtype=np.int64) * temporal_stride,
            len(batch_poses) - 1,
        )
        sampled = batch_poses[indices]
        reference_inverse = np.linalg.inv(sampled[0])
        rows = np.empty((latent_count, 12), dtype=np.float32)
        for index, pose in enumerate(sampled):
            relative = reference_inverse @ pose
            rows[index, :3] = relative[:3, 3]
            rows[index, 3:] = relative[:3, :3].reshape(-1)
        batches.append(rows)

    stacked = np.stack(batches)
    if is_tensor:
        return torch.as_tensor(stacked, dtype=torch.float32, device=source_device)
    return stacked


def select_adaln_actions(actions: Any, latent_indices: Any, *, device: Any = None, dtype: Any = None):
    """Select and clamp latent-rate action rows for a rollout segment.

    Args:
        actions: ``[B, T, 6]`` AdaLN action tensor (or array).
        latent_indices: 1-D index list; values are clamped into
            ``[0, T-1]`` so a short clip does not index-error.
        device / dtype: Optional relocation of the gathered rows.

    Returns:
        ``[B, len(indices), 6]`` on *device*.
    """

    import torch

    values = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError(f"actions must be [B,T,6], got {tuple(values.shape)}")
    target_device = device if device is not None else values.device
    indices = torch.as_tensor(latent_indices, device=target_device, dtype=torch.long).flatten()
    indices = indices.clamp(min=0, max=values.shape[1] - 1)
    values = values.to(device=target_device, dtype=dtype or values.dtype)
    return values.index_select(1, indices)


def one_hot_camera_actions_to_labels(one_hot: Any) -> np.ndarray:
    """Map forward/backward/left/right one-hot rows to labels 0–8.

    Combinations of the four translation bits become the WorldPlay
    9-way translation class (idle plus four cardinals plus four
    diagonals). Unknown bit patterns map to 0 (idle).
    """

    rows = np.asarray(one_hot)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"Expected camera action rows shaped (N, 4), got {rows.shape}")
    return np.asarray([_TRANSLATION_LABELS.get(tuple(int(value) for value in row), 0) for row in rows], dtype=np.int64)


def discretize_camera_poses_to_actions(view_matrices: Any) -> np.ndarray:
    """Derive the 81-class WorldPlay action labels from world-to-camera poses.

    Consecutive c2w relatives are thresholded into 4-bit translation
    (angle to +Z / +X) and 4-bit rotation (yaw/pitch degrees). Each
    nibble is mapped through :func:`one_hot_camera_actions_to_labels`
    and combined as ``translation * 9 + rotation``. Frame 0 is idle.
    Requires SciPy only on this path.
    """

    from scipy.spatial.transform import Rotation

    matrices = np.asarray(view_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera matrices shaped (T, 4, 4), got {matrices.shape}")
    camera_to_world = np.linalg.inv(matrices)
    count = len(matrices)
    translation = np.zeros((count, 4), dtype=np.int32)
    rotation = np.zeros((count, 4), dtype=np.int32)
    for index in range(1, count):
        relative = np.linalg.inv(camera_to_world[index - 1]) @ camera_to_world[index]
        direction = relative[:3, 3]
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            angles = np.degrees(np.arccos(np.clip(direction / norm, -1.0, 1.0)))
            if angles[2] < 60:
                translation[index, 0] = 1
            elif angles[2] > 120:
                translation[index, 1] = 1
            if angles[0] < 60:
                translation[index, 2] = 1
            elif angles[0] > 120:
                translation[index, 3] = 1
        rotation_angles = Rotation.from_matrix(relative[:3, :3]).as_euler("xyz", degrees=True)
        if rotation_angles[1] > 0.05:
            rotation[index, 0] = 1
        elif rotation_angles[1] < -0.05:
            rotation[index, 1] = 1
        if rotation_angles[0] > 0.05:
            rotation[index, 2] = 1
        elif rotation_angles[0] < -0.05:
            rotation[index, 3] = 1
    return one_hot_camera_actions_to_labels(translation) * 9 + one_hot_camera_actions_to_labels(rotation)


def camera_trajectory_action_labels(trajectory: str, num_frames: int):
    """Convert a compact trajectory string to a padded int64 Torch tensor.

    Parses ``w*4,j*2``-style segments via :data:`_TRAJECTORY_ACTION_LABELS`
    and writes labels into ``result[1:]``, leaving frame 0 as idle.
    Shorter strings are padded with zeros; longer ones are truncated to
    *num_frames*.
    """

    import torch

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    labels: list[int] = []
    for raw_segment in str(trajectory).strip().split(","):
        segment = raw_segment.strip().lower()
        if not segment:
            continue
        match = re.fullmatch(r"([a-z]+)(?:\*(\d+))?", segment)
        if match is None:
            raise ValueError(f"Invalid camera trajectory segment {raw_segment!r}")
        key, raw_count = match.groups()
        translation_label, rotation_label = _TRAJECTORY_ACTION_LABELS.get(key, (0, 0))
        labels.extend([translation_label * 9 + rotation_label] * int(raw_count or 1))
    result = np.zeros(num_frames, dtype=np.int64)
    fill_length = min(len(labels), num_frames - 1)
    result[1 : 1 + fill_length] = labels[:fill_length]
    return torch.from_numpy(result)


__all__ = [
    "camera_poses_to_adaln_actions",
    "camera_trajectory_action_labels",
    "camera_trajectory_tensors",
    "camera_trajectory_view_matrices",
    "generate_planar_camera_coordinates",
    "named_camera_trajectory_tensors",
    "discretize_camera_poses_to_actions",
    "one_hot_camera_actions_to_labels",
    "parse_camera_trajectory",
    "rollout_sana_wm_camera_actions",
    "rollout_wasd_camera_actions",
    "select_adaln_actions",
    "wan_camera_coordinates_to_plucker",
]
