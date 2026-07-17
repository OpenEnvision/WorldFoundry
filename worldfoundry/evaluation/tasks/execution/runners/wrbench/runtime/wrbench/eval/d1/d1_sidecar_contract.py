"""D1 camera-control sidecar contract validation.

This module validates the target side of the camera-control metric. It does not
run MegaSAM and it does not infer targets from labels; it only decides whether a
payload is eligible to serve as the canonical intended OpenCV C2W control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OPENCV_C2W_VALUES = {
    "opencv_c2w",
    "opencv camera-to-world, x-right y-down z-forward",
    "opencv camera-to-world, x right y down z forward",
}
BENCHMARK_TARGET_ROLE = "benchmark_intended_control"
LEGACY_TARGET_ROLE = "canonical_intended_control"
MAIN_ROTATION_PROFILE = "canonical_60deg"
LEGACY_MAIN_ROTATION_PROFILE = "yaw_go_return_60deg"
CERTIFIED_STATUS = "certified"
MAIN_EVIDENCE_LEVELS = {
    "benchmark_intent",
    "payload_audited",
}
ALLOWED_ADAPTER_PROVENANCE = {
    "exact_model_payload_exported",
    "deterministic_adapter",
    "uncertified_estimate",
}
CONTROL_FAMILY_BY_DIRECTION = {
    "yaw_LR": "rotation",
    "yaw_RL": "rotation",
    "pan_LR": "translation",
    "pan_RL": "translation",
    "static": "static",
}


@dataclass(frozen=True)
class SidecarValidation:
    ok: bool
    status: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    target_pose_path: Path | None = None
    camera_trajectory_path: Path | None = None
    camera_sidecar_path: Path | None = None
    yaw_peak_deg: float | None = None
    adapter_provenance: str | None = None
    evidence_level: str | None = None
    control_family: str | None = None
    control_direction: str | None = None
    control_profile: str | None = None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def is_opencv_c2w(value: Any) -> bool:
    raw = _norm(value)
    if raw in OPENCV_C2W_VALUES:
        return True
    normalized = raw.replace("_", " ").replace("-", " ")
    return all(
        token in normalized
        for token in ("opencv", "camera to world", "x right", "y down", "z forward")
    )


def payload_convention(payload: dict[str, Any]) -> Any:
    return (
        payload.get("target_coordinate_convention")
        or payload.get("coordinate_convention")
        or payload.get("coordinate")
        or payload.get("d1_metric_tier")
        or payload.get("metric_tier")
    )


def yaw_peak_deg(payload: dict[str, Any]) -> float | None:
    value = payload.get("yaw_peak_deg")
    if value is None:
        value = payload.get("target_yaw_peak_deg")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def adapter_provenance_kind(payload: dict[str, Any]) -> str | None:
    value = payload.get("adapter_provenance")
    if isinstance(value, dict):
        for key in ("kind", "type", "status", "provenance", "mode"):
            item = value.get(key)
            if item:
                return str(item).strip()
        return None
    if value is None:
        return None
    return str(value).strip()


def evidence_level(payload: dict[str, Any]) -> str | None:
    value = payload.get("evidence_level")
    if isinstance(value, dict):
        for key in ("level", "kind", "status"):
            item = value.get(key)
            if item:
                return str(item).strip()
        return None
    if value:
        return str(value).strip()
    if _norm(payload.get("target_certification_status")) == CERTIFIED_STATUS:
        return "payload_audited"
    return None


def control_direction(payload: dict[str, Any]) -> str | None:
    for key in ("control_direction", "camera_type", "camera", "source_action"):
        value = payload.get(key)
        if value in CONTROL_FAMILY_BY_DIRECTION:
            return str(value)
    return None


def control_family(payload: dict[str, Any], direction: str | None = None) -> str | None:
    value = payload.get("control_family")
    if value:
        return str(value).strip()
    if direction in CONTROL_FAMILY_BY_DIRECTION:
        return CONTROL_FAMILY_BY_DIRECTION[str(direction)]
    return None


def control_profile(payload: dict[str, Any]) -> str | None:
    value = payload.get("control_profile")
    if value:
        return str(value).strip()
    legacy = payload.get("canonical_profile")
    if legacy == LEGACY_MAIN_ROTATION_PROFILE:
        return MAIN_ROTATION_PROFILE
    if isinstance(legacy, str) and legacy.startswith("yaw_go_return_") and legacy.endswith("deg"):
        return "diagnostic_" + legacy.removeprefix("yaw_go_return_")
    return None


def _first_path(payload: dict[str, Any], fields: tuple[str, ...]) -> Path | None:
    for field_name in fields:
        value = payload.get(field_name)
        if value:
            return Path(str(value))
    return None


def _default_path(video_path: str | Path | None, suffix: str) -> Path | None:
    if not video_path:
        return None
    return Path(str(video_path) + suffix)


def validate_camera_control_target(
    payload: dict[str, Any],
    *,
    video_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    require_artifacts: bool = False,
    require_camera_sidecar: bool = False,
    require_main_profile: bool = True,
) -> SidecarValidation:
    """Validate a camera-control target payload for D1 main-score eligibility."""

    notes: list[str] = []
    if not isinstance(payload, dict):
        return SidecarValidation(False, "invalid_sidecar", ("payload is not an object",))

    convention = payload_convention(payload)
    if not is_opencv_c2w(convention):
        notes.append(f"target convention is not OpenCV C2W: {convention}")

    target_path = _first_path(payload, ("target_pose_path", "trajectory_c2w_path"))
    if target_path is None:
        target_path = _default_path(video_path, ".target_c2w.npy")
    trajectory_path = _first_path(payload, ("camera_trajectory_path",))
    if trajectory_path is None:
        trajectory_path = _default_path(video_path, ".camera_trajectory.json")
    camera_path = Path(sidecar_path) if sidecar_path else _default_path(video_path, ".camera.json")

    if require_main_profile:
        role = payload.get("target_role")
        if role not in {BENCHMARK_TARGET_ROLE, LEGACY_TARGET_ROLE}:
            notes.append(f"target_role is not {BENCHMARK_TARGET_ROLE}: {role}")
        direction = control_direction(payload)
        family = control_family(payload, direction)
        profile = control_profile(payload)
        level = evidence_level(payload)
        if direction is None:
            notes.append("control_direction is missing or invalid")
        if family not in {"rotation", "translation", "static"}:
            notes.append(f"control_family is missing or invalid: {family}")
        elif direction is not None and CONTROL_FAMILY_BY_DIRECTION.get(direction) != family:
            notes.append(f"control_family does not match control_direction: {family}/{direction}")
        if profile is None:
            notes.append("control_profile is missing")
        if level is None:
            notes.append("evidence_level is missing")
        elif level not in MAIN_EVIDENCE_LEVELS:
            notes.append(f"evidence_level is not main-score eligible: {level}")
        yaw = yaw_peak_deg(payload)
        if family == "rotation":
            if profile != MAIN_ROTATION_PROFILE:
                notes.append(f"control_profile is not {MAIN_ROTATION_PROFILE}: {profile}")
            if yaw is None:
                notes.append("yaw_peak_deg is missing or invalid")
            elif abs(abs(yaw) - 60.0) > 1e-6:
                notes.append(f"yaw_peak_deg is not 60: {yaw}")
        elif family == "translation":
            if profile not in {"canonical_pan", "canonical_translation"}:
                notes.append(f"control_profile is not canonical_pan: {profile}")
        elif family == "static":
            if profile != "canonical_static":
                notes.append(f"control_profile is not canonical_static: {profile}")
        for field_name in ("num_frames", "fps", "image_size", "fov", "trajectory_sampling_rule"):
            if payload.get(field_name) in (None, ""):
                notes.append(f"{field_name} is missing")
        provenance = adapter_provenance_kind(payload)
        if provenance is not None and provenance not in ALLOWED_ADAPTER_PROVENANCE:
            notes.append(f"adapter_provenance is not recognized: {provenance}")
    else:
        yaw = yaw_peak_deg(payload)
        provenance = adapter_provenance_kind(payload)
        direction = control_direction(payload)
        family = control_family(payload, direction)
        profile = control_profile(payload)
        level = evidence_level(payload)

    if require_artifacts:
        if target_path is None or not target_path.exists():
            notes.append("target_c2w artifact is missing")
        if trajectory_path is None or not trajectory_path.exists():
            notes.append("camera_trajectory artifact is missing")
    if require_camera_sidecar and (camera_path is None or not camera_path.exists()):
        notes.append("camera sidecar artifact is missing")

    if any("convention" in note for note in notes):
        status = "coordinate_mismatch"
    elif any("profile" in note or "yaw_peak_deg is not 60" in note or "control_family does not match" in note for note in notes):
        status = "profile_mismatch"
    elif any("artifact is missing" in note or "is missing" in note for note in notes):
        status = "missing_gt"
    else:
        status = "ok" if not notes else "invalid_sidecar"

    ok = not notes
    return SidecarValidation(
        ok=ok,
        status=status,
        notes=tuple(notes),
        target_pose_path=target_path,
        camera_trajectory_path=trajectory_path,
        camera_sidecar_path=camera_path,
        yaw_peak_deg=yaw,
        adapter_provenance=provenance,
        evidence_level=level,
        control_family=family,
        control_direction=direction,
        control_profile=profile,
    )
