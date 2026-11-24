"""Camera trajectory generators for inference evaluation.

All poses are w2c (world-to-camera) OpenCV convention, matching the training
pipeline (build_worldplaygen_lmdb.py: c2w -> w2c via np.linalg.inv).

Trajectory string format (each step = 0.08 unit translation or 3° rotation):
  w*N  -- move forward  (+Z in camera local frame)
  s*N  -- move backward (-Z)
  a*N  -- move left     (-X)
  d*N  -- move right    (+X)
  u*N  -- move up       (-Y, OpenCV Y-down)
  dn*N -- move down     (+Y)
  j*N  -- yaw left      (rotate around Y, positive)
  l*N  -- yaw right     (rotate around Y, negative)
  i*N  -- pitch up      (rotate around X, negative)
  k*N  -- pitch down    (rotate around X, positive)
  h*N  -- hold          (camera does not move)

``h`` exists for evaluation cases whose camera is fixed -- WBench's
``nav_cate: none`` cases, for instance. Those still need viewmats: PRoPE was
trained with the camera pathway always active, so passing ``viewmats=None``
runs a different network than the one that was trained, and a bad sample then
says nothing about the checkpoint.

Example: "w*19" -> 20-frame forward dolly (1 identity + 19 steps).
Chain segments: "w*10,d*9", "w*9,j*10"
"""

import re

import numpy as np

_STEP = 0.08
_ROT_STEP = np.radians(3.0)  # 3.0 degrees per latent frame


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Direction -> per-step motion dict (same convention as HY-WorldPlay)
_MOTIONS = {
    "w": {"forward": _STEP},
    "s": {"forward": -_STEP},
    "d": {"right": _STEP},
    "a": {"right": -_STEP},
    "u": {"up": _STEP},
    "dn": {"up": -_STEP},
    "j": {"yaw": -_ROT_STEP},  # yaw left
    "l": {"yaw": _ROT_STEP},  # yaw right
    "i": {"pitch": _ROT_STEP},  # pitch up
    "k": {"pitch": -_ROT_STEP},  # pitch down
    "h": {},  # hold: identity step, fixed camera
}


def _generate_c2w_trajectory(motions):
    """Build c2w 4x4 matrices from motion dicts.

    Exact equivalent of HY-WorldPlay/hyvideo/generate_custom_trajectory.py
    to ensure poses match the training pipeline.
    """
    T = np.eye(4)
    poses = [T.copy()]
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0, 0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0, 0])
        up = move.get("up", 0.0)
        if up:
            # up in camera frame = -Y (OpenCV Y-down)
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return poses


def parse_trajectory(traj_str: str) -> np.ndarray:
    """Parse trajectory string into (T, 4, 4) w2c viewmats.

    Builds c2w via _generate_c2w_trajectory (matching training pipeline),
    then inverts to w2c via np.linalg.inv. First frame is always identity.
    """
    segments = traj_str.strip().split(",")
    motions = []
    for seg in segments:
        seg = seg.strip()
        m = re.fullmatch(r"([a-z+]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: '{seg}'. Expected 'w*19'.")
        key, n = m.group(1), int(m.group(2))
        # '+' composes simultaneous motions in one step: 'w+k' walks forward
        # while pitching down. WBench's navigation vocabulary needs this --
        # 43 of its 601 navigation turns are compound actions ("W+down",
        # "S+D"), and running them as consecutive segments would produce an
        # L-shaped path where the benchmark asks for a diagonal one.
        #
        # Components address distinct axes (forward / right / up / yaw /
        # pitch), so merging the dicts is the composition; a repeated axis
        # would silently drop one of the two and is rejected instead.
        parts = [p for p in key.split("+") if p]
        unknown = [p for p in parts if p not in _MOTIONS]
        if unknown:
            raise ValueError(f"Unknown direction(s) {unknown} in '{key}'. Valid: {list(_MOTIONS.keys())}")
        motion = {}
        for p in parts:
            for axis, value in _MOTIONS[p].items():
                if axis in motion:
                    raise ValueError(f"'{key}' drives '{axis}' twice; compound actions must address distinct axes")
                motion[axis] = value
        motions.extend([motion] * n)

    c2w_list = _generate_c2w_trajectory(motions)
    T = len(c2w_list)
    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        viewmats[i] = np.linalg.inv(c2w)
    return viewmats
