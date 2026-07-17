"""Small helpers shared by unified camera adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wrbench.payload import ModelControlTimeline
from wrbench.registry import resolve_model_amplitude
from wrbench.trajectory import CameraTrajectory


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    out = u @ vh
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def _scale_rotation_matrix(rotation: np.ndarray, gain: float) -> np.ndarray:
    rotation = _orthonormalize_rotation(np.asarray(rotation, dtype=np.float64))
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return rotation
    axis /= norm
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    scaled_theta = theta * float(gain)
    return np.eye(3, dtype=np.float64) + np.sin(scaled_theta) * skew + (1.0 - np.cos(scaled_theta)) * (skew @ skew)


def _apply_relative_rotation_gain(c2w: np.ndarray, gain: float) -> np.ndarray:
    if abs(float(gain) - 1.0) < 1e-8:
        return c2w
    out = c2w.copy()
    base_rotation = _orthonormalize_rotation(out[0, :3, :3])
    for index, rotation in enumerate(out[:, :3, :3]):
        relative = base_rotation.T @ _orthonormalize_rotation(rotation)
        out[index, :3, :3] = (base_rotation @ _scale_rotation_matrix(relative, float(gain))).astype(out.dtype)
    return out


def ensure_work_dir(work_dir: str | Path | None) -> Path:
    out = Path(work_dir) if work_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    return out


def model_target_trajectory(trajectory: CameraTrajectory, model_name: str, num_frames: int) -> tuple[CameraTrajectory, object]:
    amp = resolve_model_amplitude(model_name)
    target = trajectory.resample(int(num_frames))
    c2w = target.to_c2w()
    c2w = _apply_relative_rotation_gain(c2w, float(amp.rotation_gain))
    t = c2w[:, :3, 3]
    max_abs = float(np.max(np.abs(t))) if t.size else 0.0
    if max_abs > 0:
        gain = float(amp.translation_gain)
        capped = np.clip(t * gain, -float(amp.max_amount), float(amp.max_amount))
        c2w[:, :3, 3] = capped
    if max_abs > 0 or abs(float(amp.rotation_gain) - 1.0) >= 1e-8:
        target = CameraTrajectory(
            c2w=c2w,
            intrinsics=target.intrinsics,
            camera_type=target.camera_type,
            fps=target.fps,
            source=target.source,
            conversion_mode=target.conversion_mode,
        )
    return target, amp


def cameractrl_rows(trajectory: CameraTrajectory) -> np.ndarray:
    w2c = trajectory.to_w2c()
    intr = trajectory.intrinsics
    rows = np.zeros((trajectory.frame_count, 19), dtype=np.float32)
    rows[:, 0] = intr[:, 0, 0]
    rows[:, 1] = intr[:, 1, 1]
    rows[:, 2] = intr[:, 0, 2]
    rows[:, 3] = intr[:, 1, 2]
    rows[:, 4:16] = w2c[:, :3, :4].reshape(trajectory.frame_count, 12)
    rows[:, 16] = intr[:, 0, 0]
    rows[:, 17] = intr[:, 1, 1]
    rows[:, 18] = 1.0
    return rows


def write_cameractrl_txt(path: str | Path, trajectory: CameraTrajectory) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, cameractrl_rows(trajectory), fmt="%.8f")
    return out


def write_json(path: str | Path, payload: object) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def base_metadata(model_name: str, amp: object) -> dict:
    return {
        "adapter_version": 1,
        "rotation_gain": amp.rotation_gain,
        "translation_gain": amp.translation_gain,
        "translation_unit": amp.translation_unit,
        "calibration_status": amp.calibration_status,
    }


def megasam_precision_inputs_from_trajectory(trajectory: CameraTrajectory) -> dict:
    """Structured peak tuple for MegaSAM precision QC from a post-remap target trajectory."""
    c2w = trajectory.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    trans = rel[:, :3, 3]
    camera_type = str(trajectory.camera_type or "")

    yaw_peak = float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0
    trans_norms = np.linalg.norm(trans, axis=1) if trans.size else np.zeros(0)
    trans_peak_row = int(np.argmax(trans_norms)) if len(trans_norms) else 0
    trans_peak_vec = trans[trans_peak_row] if trans.size else np.zeros(3)
    trans_axis_idx = int(np.argmax(np.abs(trans_peak_vec))) if trans_peak_vec.size else 0
    trans_peak = float(trans_peak_vec[trans_axis_idx]) if trans_peak_vec.size else 0.0

    if camera_type.startswith("pan") or (abs(trans_peak) > 1e-6 and abs(yaw_peak) < 1e-3):
        return {
            "rotation_axis": None,
            "rotation_peak_signed_deg": 0.0,
            "translation_peak_signed": trans_peak,
            "translation_axis": ["x", "y", "z"][trans_axis_idx],
        }
    return {
        "rotation_axis": "y",
        "rotation_peak_signed_deg": yaw_peak,
        "translation_peak_signed": 0.0,
        "translation_axis": None,
    }


def linspace_frame_indices(requested_frame_count: int, sample_count: int) -> list[int]:
    if int(requested_frame_count) <= 0 or int(sample_count) <= 0:
        raise ValueError("requested_frame_count and sample_count must be positive")
    if int(sample_count) == 1:
        return [0]
    values = np.linspace(0, int(requested_frame_count) - 1, int(sample_count))
    return [int(round(float(v))) for v in values]


def build_model_control_timeline_metadata(
    *,
    payload_type: str,
    control_sample_kind: str,
    requested_frame_count: int,
    target_frame_count: int,
    control_sample_count: int | None = None,
    source_frame_indices: list[int] | None = None,
    sampling_rule: str,
    coordinate_convention: str = "opencv_c2w",
    target_c2w_is_model_effective: bool = True,
    extra: dict | None = None,
) -> dict:
    count = int(control_sample_count if control_sample_count is not None else target_frame_count)
    if source_frame_indices is not None:
        indices = source_frame_indices
    elif count == 0:
        indices = []
    else:
        indices = linspace_frame_indices(int(requested_frame_count), count)
    if len(indices) != count:
        raise ValueError("source_frame_indices length must match control_sample_count")
    if len(indices) > 1:
        stride_hint = float((indices[-1] - indices[0]) / (len(indices) - 1))
    elif len(indices) == 1:
        stride_hint = 0.0
    else:
        stride_hint = None
    timeline = ModelControlTimeline(
        schema_version=1,
        control_sample_kind=str(control_sample_kind),
        payload_type=str(payload_type),
        requested_frame_count=int(requested_frame_count),
        target_frame_count=int(target_frame_count),
        control_sample_count=count,
        source_frame_indices=indices,
        model_control_indices=list(range(count)),
        sampling_rule=str(sampling_rule),
        stride_hint=stride_hint,
        coordinate_convention=str(coordinate_convention),
        target_c2w_is_model_effective=bool(target_c2w_is_model_effective),
        extra=extra or {},
    )
    return timeline.to_metadata()


def adapter_taxonomy_metadata(
    *,
    model_name: str,
    amp: object,
    target: CameraTrajectory,
    requested_frames: int,
    payload_type: str = "",
    certification_kind: str = "direct_frame_pose_payload",
    model_payload_summary: dict | None = None,
    target_c2w_is_model_effective: bool = True,
    target_frame_indices: list[int] | None = None,
    control_sample_kind: str = "dense_pose",
    control_sample_count: int | None = None,
    source_frame_indices: list[int] | None = None,
    sampling_rule: str = "one_control_per_target_frame",
    model_control_extra: dict | None = None,
) -> dict:
    model_frames = int(target.frame_count)
    meta = base_metadata(model_name, amp)
    meta.update(
        {
            "certification_kind": certification_kind,
            "target_certification_kind": certification_kind,
            "target_c2w_is_model_effective": bool(target_c2w_is_model_effective),
            "megasam_precision_inputs": megasam_precision_inputs_from_trajectory(target),
            "frame_mapping": {
                "requested_frames": int(requested_frames),
                "model_frames": model_frames,
                "target_frame_indices": target_frame_indices or list(range(model_frames)),
            },
            "model_control_timeline": build_model_control_timeline_metadata(
                payload_type=payload_type,
                control_sample_kind=control_sample_kind,
                requested_frame_count=int(requested_frames),
                target_frame_count=model_frames,
                control_sample_count=control_sample_count if control_sample_count is not None else model_frames,
                source_frame_indices=source_frame_indices,
                sampling_rule=sampling_rule,
                target_c2w_is_model_effective=target_c2w_is_model_effective,
                extra=model_control_extra,
            ),
            "canonical_control": str(target.camera_type or ""),
        }
    )
    if model_payload_summary is not None:
        meta["model_payload_summary"] = model_payload_summary
    return meta


REQUIRED_UNIFIED_SIDECAR_FIELDS = (
    "camera_control_source",
    "frame_action_script",
    "target_certification_kind",
    "model_payload_type",
    "model_payload_summary",
    "frame_mapping",
    "canonical_control",
    "target_c2w_is_model_effective",
    "megasam_precision_inputs",
    "model_control_timeline",
)


def unified_sidecar_extra(
    *,
    payload_metadata: dict,
    payload_type: str,
    camera_script: str,
    target_yaw_peak_deg: float | None = None,
) -> dict:
    """Merge adapter metadata into the unified sidecar contract."""
    extra = dict(payload_metadata)
    extra.update(
        {
            "camera_control_source": "frame_action_script",
            "frame_action_script": camera_script,
            "target_certification_kind": extra.get(
                "target_certification_kind",
                extra.get("certification_kind", "direct_frame_pose_payload"),
            ),
            "model_payload_type": payload_type,
            "model_payload_summary": extra.get("model_payload_summary") or {},
        }
    )
    timeline = dict(extra.get("model_control_timeline") or {})
    if timeline:
        timeline["payload_type"] = payload_type
        extra["model_control_timeline"] = timeline
    if target_yaw_peak_deg is not None and "target_yaw_peak_deg" not in extra:
        extra["target_yaw_peak_deg"] = float(target_yaw_peak_deg)
    return extra
