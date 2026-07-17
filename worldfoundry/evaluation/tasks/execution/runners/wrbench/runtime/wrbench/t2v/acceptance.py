"""Formal T2V intake acceptance gates and calibration checks."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from wrbench.actions import parse_camera_script
from wrbench.builder import build_camera_trajectory

T2V_GATE_NAMES = (
    "subject_present",
    "scene_present",
    "action_judgeable",
    "camera_visible",
    "camera_amplitude_ok",
)

_EVENT_PRONOUN_START_RE = re.compile(
    r"(?<=\.\s)(He|She|It)\b|^(He|She|It)\b|(?<=\,\s)(he|she|it)\b"
)


def validate_subject_anchored_prompt(prompt: str) -> bool:
    """Return True when the event sentence does not start with a bare subject pronoun."""
    text = str(prompt or "").strip()
    if not text:
        return False
    if "background." not in text:
        return True
    event_part = text.split("background.", 1)[1].strip()
    return _EVENT_PRONOUN_START_RE.search(event_part) is None


def expected_yaw_peak_deg(camera_script: str) -> float | None:
    script = parse_camera_script(str(camera_script))
    if all(action.kind == "static" for action in script.actions):
        return 0.0
    if any(action.kind == "yaw" for action in script.actions):
        trajectory = build_camera_trajectory(script, width=2, height=2, fps=script.fps)
        c2w = trajectory.to_c2w()
        rel = np.linalg.inv(c2w[0]) @ c2w
        yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
        return abs(float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0)
    return None


def verify_minwm_rotation_calibration(
    token_details: dict[str, Any],
    *,
    tolerance_deg: float = 1.5,
) -> tuple[bool, dict[str, Any]]:
    """Check adapter token metadata against requested/effective yaw peaks."""
    requested = token_details.get("requested_yaw_peak_deg")
    effective = token_details.get("effective_yaw_peak_deg")
    runtime_step = token_details.get("runtime_yaw_deg_per_token")
    requires_patch = bool(token_details.get("requires_runtime_rot_step_patch"))
    if requested is None or effective is None:
        return False, {"reason": "missing_yaw_peak_metadata"}
    delta = abs(float(requested) - float(effective))
    ok = delta <= tolerance_deg
    return ok, {
        "requested_yaw_peak_deg": requested,
        "effective_yaw_peak_deg": effective,
        "runtime_yaw_deg_per_token": runtime_step,
        "requires_runtime_rot_step_patch": requires_patch,
        "delta_deg": delta,
        "tolerance_deg": tolerance_deg,
    }


def assess_t2v_gates(
    *,
    subject_present: bool,
    scene_present: bool,
    action_judgeable: bool,
    camera_visible: bool,
    camera_amplitude_ok: bool,
) -> dict[str, Any]:
    gates = {
        "subject_present": bool(subject_present),
        "scene_present": bool(scene_present),
        "action_judgeable": bool(action_judgeable),
        "camera_visible": bool(camera_visible),
        "camera_amplitude_ok": bool(camera_amplitude_ok),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "passed": passed,
        "failed_gates": [name for name, ok in gates.items() if not ok],
    }
