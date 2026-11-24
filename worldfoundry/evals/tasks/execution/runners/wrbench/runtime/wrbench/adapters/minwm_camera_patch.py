"""Runtime camera-token patch helpers for minWM native camera tokens."""

from __future__ import annotations

from pathlib import Path

_PATCH_MODULE = r'''\
"""WRBench runtime override for minWM camera trajectory tokens.

Adds a static no-op token and allows WRBench to scale the yaw rotation step.
"""

import re

import numpy as np
import torch

_STEP = 0.08
_ROT_STEP_DEG = __WRBENCH_ROT_STEP_DEG__
_ROT_STEP = np.radians(float(_ROT_STEP_DEG))


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


_MOTIONS = {
    "w":  {"forward":  _STEP},
    "s":  {"forward": -_STEP},
    "d":  {"right":    _STEP},
    "a":  {"right":   -_STEP},
    "u":  {"up":       _STEP},
    "dn": {"up":      -_STEP},
    "j":  {"yaw":     -_ROT_STEP},
    "l":  {"yaw":      _ROT_STEP},
    "i":  {"pitch":    _ROT_STEP},
    "k":  {"pitch":   -_ROT_STEP},
    "z":  {},
}


def _generate_c2w_trajectory(motions):
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
            T[:3, 3] += T[:3, :3] @ np.array([0, -up, 0])
        poses.append(T.copy())
    return poses


def parse_trajectory(traj_str: str) -> np.ndarray:
    segments = traj_str.strip().split(",")
    motions = []
    for seg in segments:
        seg = seg.strip()
        m = re.fullmatch(r"([a-z]+)\*(\d+)", seg)
        if m is None:
            raise ValueError(f"Cannot parse trajectory segment: {seg!r}. Expected 'w*19'.")
        key, n = m.group(1), int(m.group(2))
        if key not in _MOTIONS:
            raise ValueError(f"Unknown direction {key!r}. Valid: {list(_MOTIONS.keys())}")
        motions.extend([_MOTIONS[key]] * n)

    c2w_list = _generate_c2w_trajectory(motions)
    T = len(c2w_list)
    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i, c2w in enumerate(c2w_list):
        viewmats[i] = np.linalg.inv(c2w)
    return viewmats


def make_camera_tensors(
    traj_str: str,
    fx: float = 0.5050505,
    fy: float = 0.89786756,
    cx: float = 0.5,
    cy: float = 0.5,
    device="cpu",
    dtype=torch.float32,
):
    viewmats_np = parse_trajectory(traj_str)
    T = len(viewmats_np)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    Ks_np = np.tile(K, (T, 1, 1))

    viewmats = torch.tensor(viewmats_np, dtype=dtype, device=device).unsqueeze(0)
    Ks = torch.tensor(Ks_np, dtype=dtype, device=device).unsqueeze(0)
    return viewmats, Ks
'''


_SITECUSTOMIZE = r'''\
"""WRBench sitecustomize: force minWM wan_utils.camera_trajectory to the patched module.

A meta_path finder takes priority over sys.path, so this overrides the upstream
module even though `torchrun Wan21/wan_inference.py` puts `Wan21/` at sys.path[0].
"""

import importlib.abc
import importlib.util
import sys
from pathlib import Path

_TARGET = "wan_utils.camera_trajectory"
_PATCH_FILE = Path(__file__).resolve().parent / "wan_utils" / "camera_trajectory.py"


class _WRBenchCameraFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == _TARGET and _PATCH_FILE.is_file():
            return importlib.util.spec_from_file_location(name, _PATCH_FILE)
        return None


if not any(
    isinstance(f, _WRBenchCameraFinder) for f in sys.meta_path
):
    sys.meta_path.insert(0, _WRBenchCameraFinder())
    sys.stderr.write("WRBENCH_PATCH_SENTINEL meta_path finder installed\n")
'''


def write_rotation_step_patch(
    work_dir: str | Path,
    *,
    runtime_yaw_deg_per_token: float,
) -> dict[str, str]:
    """Materialize a meta_path override module + launcher for minWM yaw scaling.

    Uses a ``sitecustomize.py`` meta_path finder (priority over ``sys.path``) so
    the patched ``wan_utils.camera_trajectory`` wins even though
    ``torchrun Wan21/wan_inference.py`` places ``Wan21/`` at ``sys.path[0]``.
    """
    root = Path(work_dir)
    patch_root = root / "minwm_camera_patch"
    module_path = patch_root / "wan_utils" / "camera_trajectory.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        _PATCH_MODULE.replace("__WRBENCH_ROT_STEP_DEG__", repr(float(runtime_yaw_deg_per_token))),
        encoding="utf-8",
    )
    sitecustomize_path = patch_root / "sitecustomize.py"
    sitecustomize_path.write_text(_SITECUSTOMIZE, encoding="utf-8")
    launcher_path = root / "minwm_wan_launch_with_rot_step.sh"
    launcher_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'PATCH_ROOT="${HERE}/minwm_camera_patch"\n'
        'PATCH_FILE="${PATCH_ROOT}/wan_utils/camera_trajectory.py"\n'
        'if [[ ! -f "${PATCH_FILE}" ]]; then\n'
        '  echo "FATAL: rotation-step patch module missing at ${PATCH_FILE}" >&2\n'
        "  exit 3\n"
        "fi\n"
        'if [[ ! -f "${PATCH_ROOT}/sitecustomize.py" ]]; then\n'
        '  echo "FATAL: sitecustomize meta_path override missing at ${PATCH_ROOT}/sitecustomize.py" >&2\n'
        "  exit 3\n"
        "fi\n"
        'if [[ -z "${PYTHONPATH+x}" ]]; then\n'
        '  echo "FATAL: PYTHONPATH must be set by the WRBench runtime before applying the minWM patch" >&2\n'
        "  exit 3\n"
        "fi\n"
        'export PYTHONPATH="${PATCH_ROOT}:${PYTHONPATH}"\n'
        'exec "$@"\n',
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)
    return {
        "patch_root": str(patch_root),
        "launcher_path": str(launcher_path),
    }


def apply_launcher_to_command(command: list[str], launcher_path: str) -> list[str]:
    """Wrap a command template so the rotation-step patch is active at runtime."""
    return [launcher_path, *command]
