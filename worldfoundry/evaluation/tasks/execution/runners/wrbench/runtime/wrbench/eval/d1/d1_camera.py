#!/usr/bin/env python3
"""D1 camera movement target loading, diagnostics, and JSONL sidecar scoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wrbench.eval.d1.d1_sidecar_contract import validate_camera_control_target
from wrbench.eval.d1.geometry import safe_video_id
from wrbench.eval.d1.pose import (
    PRIMARY_TARGET_POSE_PATH_FIELDS,
    POSE_INLINE_FIELDS,
    _expand_poses,
    _normalize_predicted_poses,
    score_trajectory_pose,
)


ELIGIBLE_CAMERAS = frozenset({"static", "yaw_LR", "yaw_RL", "pan_LR", "pan_RL"})
CERTIFIED_TARGET_STATUS = "certified"
UNCERTIFIED_TARGET_NOTE = "target_certification_status is not certified"
D1_TARGET_SIDECAR_SUFFIX = ".d1_target.json"
D1_TARGET_SIDECAR_FIELDS = ("target_pose_path", "trajectory_c2w_path")
OPENCV_C2W_VALUES = {
    "opencv_c2w",
    "opencv camera-to-world, x-right y-down z-forward",
    "opencv camera-to-world, x right y down z forward",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _camera_type(row_or_camera: dict[str, Any] | str) -> str:
    if isinstance(row_or_camera, dict):
        return str(row_or_camera.get("camera_type") or row_or_camera.get("camera") or "").strip()
    return str(row_or_camera or "").strip()


def _target_certification_status(payload: dict[str, Any]) -> str:
    return str(payload.get("target_certification_status") or "").strip().lower()


def _is_certified_target(payload: dict[str, Any]) -> bool:
    return _target_certification_status(payload) == CERTIFIED_TARGET_STATUS


def _norm_convention(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _is_opencv_c2w(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if raw in OPENCV_C2W_VALUES:
        return True
    normalized = _norm_convention(value)
    normalized = normalized.replace("-", " ")
    return all(
        token in normalized
        for token in ("opencv", "camera to world", "x right", "y down", "z forward")
    )


def _target_convention(payload: dict[str, Any]) -> Any:
    return (
        payload.get("target_coordinate_convention")
        or payload.get("coordinate_convention")
        or payload.get("coordinate")
        or payload.get("d1_metric_tier")
        or payload.get("metric_tier")
    )


def _is_loaded_certified_opencv(target: dict[str, Any] | None) -> bool:
    if not target or target.get("status") != "ok":
        return False
    return _is_opencv_c2w(target.get("target_coordinate_convention") or target.get("metric_tier"))


def _uncertified_target(source: str, *, metric_tier: str = "opencv_c2w") -> dict[str, Any]:
    return {
        "poses_c2w": None,
        "metric_tier": metric_tier,
        "target_source": source,
        "target_certification_status": "uncertified",
        "notes": [UNCERTIFIED_TARGET_NOTE],
        "status": "uncertified_target",
    }


def _validation_metadata(payload: dict[str, Any], validation: Any) -> dict[str, Any]:
    status = _target_certification_status(payload) or validation.evidence_level or "benchmark_intent"
    return {
        "target_certification_status": status,
        "target_coordinate_convention": _target_convention(payload),
        "evidence_level": validation.evidence_level,
        "control_family": validation.control_family,
        "control_direction": validation.control_direction,
        "control_profile": validation.control_profile,
    }


def _load_explicit_target(row: dict[str, Any], *, require_main_profile: bool) -> dict[str, Any] | None:
    for field in PRIMARY_TARGET_POSE_PATH_FIELDS:
        value = row.get(field)
        if not value:
            continue
        metric_tier = str(row.get("metric_tier") or row.get("d1_metric_tier") or "opencv_c2w")
        convention = _target_convention(row)
        if not _is_opencv_c2w(convention):
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                "target_certification_status": CERTIFIED_TARGET_STATUS,
                "target_coordinate_convention": convention,
                "notes": [f"target convention is not OpenCV C2W: {convention}"],
                "status": "coordinate_mismatch",
            }
        validation = validate_camera_control_target(row, require_main_profile=require_main_profile)
        metadata = _validation_metadata(row, validation)
        if not validation.ok:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": list(validation.notes),
                "status": validation.status,
            }
        path = Path(str(value))
        if not path.exists():
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [f"{field} does not exist"],
                "status": "missing_target_pose",
            }
        try:
            return {
                "poses_c2w": _expand_poses(np.load(path, allow_pickle=False)),
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [],
                "status": "ok",
            }
        except ValueError:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [f"{field} is not a valid pose stack"],
                "status": "invalid_pose",
            }
        except Exception:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [f"failed to read {field}"],
                "status": "error",
            }

    for field in POSE_INLINE_FIELDS:
        if row.get(field) is None:
            continue
        metric_tier = str(row.get("metric_tier") or row.get("d1_metric_tier") or "opencv_c2w")
        convention = _target_convention(row)
        if not _is_opencv_c2w(convention):
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                "target_certification_status": CERTIFIED_TARGET_STATUS,
                "target_coordinate_convention": convention,
                "notes": [f"target convention is not OpenCV C2W: {convention}"],
                "status": "coordinate_mismatch",
            }
        validation = validate_camera_control_target(row, require_main_profile=require_main_profile)
        metadata = _validation_metadata(row, validation)
        if not validation.ok:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": list(validation.notes),
                "status": validation.status,
            }
        try:
            return {
                "poses_c2w": _expand_poses(row[field]),
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [],
                "status": "ok",
            }
        except ValueError:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": field,
                **metadata,
                "notes": [f"{field} is not a valid pose stack"],
                "status": "invalid_pose",
            }
    return None


def _load_sidecar_target(
    row: dict[str, Any],
    *,
    suffix: str = ".camera.json",
    source: str = "camera_sidecar",
    fields: tuple[str, ...] = ("target_pose_path", "trajectory_c2w_path", "input_pose_path"),
    require_main_profile: bool,
) -> dict[str, Any] | None:
    path = row.get("path") or row.get("video_path")
    if not path:
        return None
    sidecar_path = Path(str(path) + suffix)
    if not sidecar_path.exists():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "poses_c2w": None,
            "metric_tier": "opencv_c2w",
            "target_source": source,
            "notes": ["failed to read camera sidecar"],
            "status": "error",
        }
    if not isinstance(sidecar, dict):
        return {
            "poses_c2w": None,
            "metric_tier": "opencv_c2w",
            "target_source": source,
            "target_certification_status": "uncertified",
            "notes": ["camera sidecar is not an object"],
            "status": "invalid_pose",
        }
    metric_tier = str(sidecar.get("metric_tier") or sidecar.get("d1_metric_tier") or "opencv_c2w")
    convention = _target_convention(sidecar)
    for field in fields:
        value = sidecar.get(field)
        if not value:
            continue
        if not _is_opencv_c2w(convention):
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": source,
                "target_certification_status": CERTIFIED_TARGET_STATUS,
                "target_coordinate_convention": convention,
                "notes": [f"target convention is not OpenCV C2W: {convention}"],
                "status": "coordinate_mismatch",
            }
        validation = validate_camera_control_target(
            sidecar,
            video_path=path,
            require_artifacts=True,
            require_camera_sidecar=True,
            require_main_profile=require_main_profile,
        )
        metadata = _validation_metadata(sidecar, validation)
        if not validation.ok:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": source,
                **metadata,
                "notes": list(validation.notes),
                "status": validation.status,
            }
        pose_path = validation.target_pose_path or Path(str(value))
        if not pose_path.exists():
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": source,
                **metadata,
                "notes": [f"{field} does not exist"],
                "status": "missing_target_pose",
            }
        try:
            return {
                "poses_c2w": _expand_poses(np.load(pose_path, allow_pickle=False)),
                "metric_tier": metric_tier,
                "target_source": source,
                **metadata,
                "notes": [],
                "status": "ok",
            }
        except ValueError:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": source,
                **metadata,
                "notes": [f"{field} is not a valid pose stack"],
                "status": "invalid_pose",
            }
        except Exception:
            return {
                "poses_c2w": None,
                "metric_tier": metric_tier,
                "target_source": source,
                **metadata,
                "notes": [f"failed to read {field}"],
                "status": "error",
            }
    return None


def build_d1_target(
    row: dict[str, Any],
    *,
    default_frames: int,
    require_main_profile: bool,
) -> dict[str, Any]:
    """Return the D1 target pose payload for one candidate row.

    Certified OpenCV C2W targets win over uncertified legacy row fields. Missing
    non-Wan targets are reported instead of reconstructed from labels, because
    D1 must compare against model-layer targets rather than adapter-surface
    camera names.
    """
    explicit = _load_explicit_target(row, require_main_profile=require_main_profile)
    d1_sidecar = _load_sidecar_target(
        row,
        suffix=D1_TARGET_SIDECAR_SUFFIX,
        source="d1_target_sidecar",
        fields=D1_TARGET_SIDECAR_FIELDS,
        require_main_profile=require_main_profile,
    )
    sidecar = _load_sidecar_target(row, require_main_profile=require_main_profile)
    if _is_loaded_certified_opencv(explicit):
        return explicit
    if _is_loaded_certified_opencv(d1_sidecar):
        return d1_sidecar
    if _is_loaded_certified_opencv(sidecar):
        return sidecar
    if explicit is not None:
        return explicit
    if d1_sidecar is not None:
        return d1_sidecar
    if sidecar is not None:
        return sidecar

    camera = _camera_type(row)
    if camera == "uncontrolled":
        return {
            "poses_c2w": None,
            "metric_tier": "excluded",
            "target_source": "uncontrolled",
            "target_certification_status": "excluded",
            "notes": ["uncontrolled camera rows are excluded from D1 camera-control accuracy"],
            "status": "excluded_uncontrolled",
        }

    return {
        "poses_c2w": None,
        "metric_tier": "opencv_c2w",
        "target_source": "missing",
        "target_certification_status": "missing",
        "notes": ["no explicit model-layer target trajectory was available"],
        "status": "missing_target_pose",
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


def _relative_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses[0]) @ poses


def _yaw_degrees(poses: np.ndarray) -> np.ndarray:
    rel = _relative_poses(poses)
    return np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))


def _translation_relative(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3, 3] - poses[0, :3, 3]


def _peak_signed(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    idx = int(np.argmax(np.abs(values)))
    return float(values[idx])


def _shape_error(target: np.ndarray, predicted: np.ndarray) -> float:
    target_rel = _translation_relative(target)
    pred_rel = _translation_relative(predicted)
    pred_norm_sq = float(np.sum(pred_rel * pred_rel))
    if pred_norm_sq > 1e-12:
        scale = max(float(np.sum(target_rel * pred_rel)) / pred_norm_sq, 0.0)
        pred_rel = pred_rel * scale
    target_scale = math.sqrt(float(np.mean(np.sum(target_rel * target_rel, axis=1))))
    if target_scale <= 1e-12:
        return math.sqrt(float(np.mean(np.sum(pred_rel * pred_rel, axis=1))))
    rmse = math.sqrt(float(np.mean(np.sum((target_rel - pred_rel) ** 2, axis=1))))
    return rmse / target_scale


def _trajectory_shape_correlation(target: np.ndarray, predicted: np.ndarray) -> float:
    target_rel = _translation_relative(target).reshape(-1)
    pred_rel = _translation_relative(predicted).reshape(-1)
    target_norm = float(np.linalg.norm(target_rel))
    pred_norm = float(np.linalg.norm(pred_rel))
    if target_norm <= 1e-12 and pred_norm <= 1e-12:
        yaw_target = _yaw_degrees(target)
        yaw_pred = _yaw_degrees(predicted)
        target_norm = float(np.linalg.norm(yaw_target))
        pred_norm = float(np.linalg.norm(yaw_pred))
        if target_norm <= 1e-12 and pred_norm <= 1e-12:
            return 1.0
        if target_norm <= 1e-12 or pred_norm <= 1e-12:
            return 0.0
        return float(np.dot(yaw_target, yaw_pred) / (target_norm * pred_norm))
    if target_norm <= 1e-12 or pred_norm <= 1e-12:
        return 0.0
    return float(np.dot(target_rel, pred_rel) / (target_norm * pred_norm))


def _movement_strength_rot(poses: np.ndarray) -> float:
    rel = _relative_poses(poses)
    if len(rel) == 0:
        return 0.0
    return max(_rotation_angle_deg(pose[:3, :3]) for pose in rel)


def _movement_strength_trans(poses: np.ndarray) -> float:
    rel = _translation_relative(poses)
    return math.sqrt(float(np.max(np.sum(rel * rel, axis=1)))) if len(rel) else 0.0


def _return_metrics(poses: np.ndarray) -> tuple[float, float]:
    rel_final = np.linalg.inv(poses[0]) @ poses[-1]
    return _rotation_angle_deg(rel_final[:3, :3]), float(np.linalg.norm(rel_final[:3, 3]))


def _same_sign(a: float, b: float, *, eps: float = 1e-8) -> bool:
    return abs(a) > eps and abs(b) > eps and (a > 0.0) == (b > 0.0)


def score_d1_trajectory(
    target_poses_c2w: Any,
    predicted_poses: Any,
    *,
    camera_type: str,
    predicted_pose_type: str,
    predicted_camera_convention: str,
    target_camera_convention: str,
    rot_scale_deg: float,
    trans_scale: float,
    yaw_weak_threshold_deg: float,
    pan_weak_threshold: float,
    static_rot_threshold_deg: float,
    static_trans_threshold: float,
) -> dict[str, Any]:
    target = _expand_poses(target_poses_c2w)
    predicted = _normalize_predicted_poses(
        predicted_poses,
        predicted_pose_type,
        predicted_camera_convention,
        target_camera_convention,
    )
    if len(target) != len(predicted):
        return {"d1_status": "length_mismatch", "d1_flags": ["length_mismatch"]}
    try:
        alignment = target[0] @ np.linalg.inv(predicted[0])
    except np.linalg.LinAlgError:
        return {"d1_status": "invalid_pose", "d1_flags": ["invalid_pose"]}
    predicted = alignment @ predicted
    rot_errors = np.asarray(
        [
            _rotation_angle_deg(target[idx, :3, :3].T @ predicted[idx, :3, :3])
            for idx in range(len(target))
        ],
        dtype=np.float64,
    )

    pose = score_trajectory_pose(
        target,
        predicted,
        rot_scale_deg=rot_scale_deg,
        trans_scale=trans_scale,
        predicted_pose_type="c2w",
        predicted_camera_convention="opencv",
        target_camera_convention=target_camera_convention,
    )
    if pose.get("pose_status") != "ok":
        return {"d1_status": str(pose.get("pose_status", "error")), "d1_flags": [str(pose.get("pose_status", "error"))]}

    camera = _camera_type(camera_type)
    yaw_target = _yaw_degrees(target)
    yaw_pred = _yaw_degrees(predicted)
    yaw_peak_target = _peak_signed(yaw_target)
    yaw_peak_pred = _peak_signed(yaw_pred)
    target_rel = _translation_relative(target)
    pred_rel = _translation_relative(predicted)
    pan_peak_target = _peak_signed(target_rel[:, 0])
    pan_peak_pred = _peak_signed(pred_rel[:, 0])
    rot_strength = _movement_strength_rot(predicted)
    trans_strength = _movement_strength_trans(predicted)
    return_rot, return_trans = _return_metrics(predicted)
    trans_shape_error = float(pose.get("pose_trans_error", _shape_error(target, predicted)))

    flags: list[str] = []
    yaw_sign_ok = False
    pan_sign_ok = False
    static_ok = False
    accuracy = float(pose["pose_reward"])

    if camera in {"yaw_LR", "yaw_RL"}:
        yaw_sign_ok = _same_sign(yaw_peak_target, yaw_peak_pred, eps=yaw_weak_threshold_deg)
        if abs(yaw_peak_target) < yaw_weak_threshold_deg or abs(yaw_peak_pred) < yaw_weak_threshold_deg:
            flags.append("weak_yaw_motion")
            accuracy = 0.0
        elif not yaw_sign_ok:
            flags.append("yaw_direction_mismatch")
            accuracy = 0.0
    elif camera in {"pan_LR", "pan_RL"}:
        pan_sign_ok = _same_sign(pan_peak_target, pan_peak_pred, eps=pan_weak_threshold)
        if abs(pan_peak_target) < pan_weak_threshold or abs(pan_peak_pred) < pan_weak_threshold:
            flags.append("weak_pan_motion")
            accuracy = 0.0
        elif not pan_sign_ok:
            flags.append("pan_direction_mismatch")
            accuracy = 0.0
    elif camera == "static":
        static_ok = (
            rot_strength <= float(static_rot_threshold_deg)
            and trans_strength <= float(static_trans_threshold)
        )
        accuracy = 1.0 if static_ok else 0.0
        if not static_ok:
            flags.append("static_motion_detected")
    else:
        flags.append("unsupported_camera_type")
        accuracy = 0.0

    direction_correct = True
    if camera in {"yaw_LR", "yaw_RL"}:
        direction_correct = bool(yaw_sign_ok)
    elif camera in {"pan_LR", "pan_RL"}:
        direction_correct = bool(pan_sign_ok)
    elif camera == "static":
        direction_correct = bool(static_ok)
    failure_reason = flags[0] if flags else None

    return {
        "d1_status": "ok",
        "d1_camera_accuracy": max(0.0, min(1.0, float(accuracy))),
        "d1_pose_reward": float(pose["pose_reward"]),
        "d1_metric_tier": "opencv_c2w",
        "rot_mae_deg": float(pose["pose_rot_error_deg"]),
        "rot_err_deg_mean": float(np.mean(rot_errors)),
        "rot_err_deg_p50": float(np.percentile(rot_errors, 50)),
        "rot_err_deg_p90": float(np.percentile(rot_errors, 90)),
        "trans_shape_error": float(trans_shape_error),
        "trans_err_mean_scale_aligned": float(trans_shape_error),
        "yaw_sign_ok": bool(yaw_sign_ok),
        "pan_sign_ok": bool(pan_sign_ok),
        "static_ok": bool(static_ok),
        "yaw_peak_target_deg": float(yaw_peak_target),
        "yaw_peak_pred_deg": float(yaw_peak_pred),
        "yaw_peak_estimated_deg": float(yaw_peak_pred),
        "yaw_peak_abs_err": float(abs(abs(yaw_peak_target) - abs(yaw_peak_pred))),
        "direction_correct": bool(direction_correct),
        "trajectory_shape_correlation": _trajectory_shape_correlation(target, predicted),
        "trajectory_valid": not flags,
        "failure_reason": failure_reason,
        "movement_strength_rot_deg": float(rot_strength),
        "movement_strength_trans_norm": float(trans_strength),
        "return_rot_deg": float(return_rot),
        "return_trans_norm": float(return_trans),
        "d1_flags": flags,
    }


def _pose_file_for(row: dict[str, Any], cache_root: str | Path, poses_file: str = "poses.npy") -> Path:
    return Path(cache_root) / "pose" / safe_video_id(row.get("video_id")) / poses_file


def _with_d1_status(row: dict[str, Any], status: str, *, target: dict[str, Any] | None = None, flags: list[str] | None = None) -> dict[str, Any]:
    enriched = dict(row)
    enriched["d1_status"] = status
    enriched["d1_camera_accuracy"] = None
    enriched["d1_pose_reward"] = None
    enriched["d1_metric_tier"] = (target or {}).get("metric_tier", "opencv_c2w")
    enriched["d1_target_source"] = (target or {}).get("target_source", "missing")
    enriched["d1_target_certification_status"] = (target or {}).get("target_certification_status", "missing")
    enriched["d1_target_coordinate_convention"] = (target or {}).get("target_coordinate_convention")
    enriched["d1_evidence_level"] = (target or {}).get("evidence_level")
    enriched["d1_control_family"] = (target or {}).get("control_family")
    enriched["d1_control_direction"] = (target or {}).get("control_direction") or _camera_type(row)
    enriched["d1_control_profile"] = (target or {}).get("control_profile")
    enriched["d1_gt_valid"] = bool((target or {}).get("status") == "ok")
    for field in (
        "rot_mae_deg",
        "trans_shape_error",
        "yaw_sign_ok",
        "pan_sign_ok",
        "static_ok",
        "yaw_peak_target_deg",
        "yaw_peak_pred_deg",
        "yaw_peak_estimated_deg",
        "yaw_peak_abs_err",
        "direction_correct",
        "trajectory_shape_correlation",
        "trajectory_valid",
        "failure_reason",
        "rot_err_deg_mean",
        "rot_err_deg_p50",
        "rot_err_deg_p90",
        "trans_err_mean_scale_aligned",
        "movement_strength_rot_deg",
        "movement_strength_trans_norm",
        "return_rot_deg",
        "return_trans_norm",
    ):
        enriched.setdefault(field, None)
    enriched["d1_flags"] = flags or []
    return enriched


def _require_config(config: dict[str, Any], field: str) -> Any:
    if field not in config:
        raise ValueError(f"D1 config field {field!r} is required")
    value = config[field]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"D1 config field {field!r} is required")
    return value


def _require_config_int(config: dict[str, Any], field: str) -> int:
    value = _require_config(config, field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"D1 config field {field!r} must be an integer") from exc


def _require_config_float(config: dict[str, Any], field: str) -> float:
    value = _require_config(config, field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"D1 config field {field!r} must be a number") from exc


def _require_config_bool(config: dict[str, Any], field: str) -> bool:
    value = _require_config(config, field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ValueError(f"D1 config field {field!r} must be a boolean")


def score_d1_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    default_frames = _require_config_int(cfg, "default_frames")
    require_main_profile = _require_config_bool(cfg, "require_main_profile")
    cache_root = _require_config(cfg, "cache_root")
    poses_file = str(_require_config(cfg, "poses_file"))
    predicted_pose_type = str(_require_config(cfg, "predicted_pose_type"))
    predicted_camera_convention = str(_require_config(cfg, "predicted_camera_convention"))
    target_camera_convention = str(_require_config(cfg, "target_camera_convention"))
    rot_scale_deg = _require_config_float(cfg, "rot_scale_deg")
    trans_scale = _require_config_float(cfg, "trans_scale")
    yaw_weak_threshold_deg = _require_config_float(cfg, "yaw_weak_threshold_deg")
    pan_weak_threshold = _require_config_float(cfg, "pan_weak_threshold")
    static_rot_threshold_deg = _require_config_float(cfg, "static_rot_threshold_deg")
    static_trans_threshold = _require_config_float(cfg, "static_trans_threshold")
    camera = _camera_type(row)
    if camera == "uncontrolled":
        target = build_d1_target(
            row,
            default_frames=default_frames,
            require_main_profile=require_main_profile,
        )
        return _with_d1_status(row, "excluded_uncontrolled", target=target, flags=["excluded_uncontrolled"])
    if camera not in ELIGIBLE_CAMERAS:
        return _with_d1_status(row, "unsupported_camera_type", flags=["unsupported_camera_type"])

    target = build_d1_target(
        row,
        default_frames=default_frames,
        require_main_profile=require_main_profile,
    )
    if target.get("status") != "ok":
        return _with_d1_status(row, str(target.get("status", "error")), target=target, flags=list(target.get("notes") or []))

    pose_file = _pose_file_for(
        row,
        cache_root,
        poses_file,
    )
    if not pose_file.exists():
        return _with_d1_status(row, "missing_output", target=target, flags=["missing_predicted_pose"])

    try:
        predicted = np.load(pose_file, allow_pickle=False)
        score = score_d1_trajectory(
            target["poses_c2w"],
            predicted,
            camera_type=camera,
            predicted_pose_type=predicted_pose_type,
            predicted_camera_convention=predicted_camera_convention,
            target_camera_convention=target_camera_convention,
            rot_scale_deg=rot_scale_deg,
            trans_scale=trans_scale,
            yaw_weak_threshold_deg=yaw_weak_threshold_deg,
            pan_weak_threshold=pan_weak_threshold,
            static_rot_threshold_deg=static_rot_threshold_deg,
            static_trans_threshold=static_trans_threshold,
        )
    except ValueError:
        return _with_d1_status(row, "invalid_pose", target=target, flags=["invalid_predicted_pose"])
    except Exception:
        return _with_d1_status(row, "error", target=target, flags=["d1_scoring_error"])

    enriched = dict(row)
    enriched.update(score)
    enriched["d1_target_source"] = target["target_source"]
    enriched["d1_target_certification_status"] = target.get("target_certification_status", CERTIFIED_TARGET_STATUS)
    enriched["d1_target_coordinate_convention"] = target.get("target_coordinate_convention")
    enriched["d1_evidence_level"] = target.get("evidence_level")
    enriched["d1_control_family"] = target.get("control_family")
    enriched["d1_control_direction"] = target.get("control_direction") or camera
    enriched["d1_control_profile"] = target.get("control_profile")
    enriched["d1_gt_valid"] = True
    enriched["d1_metric_tier"] = score.get("d1_metric_tier") or target["metric_tier"]
    return _jsonable(enriched)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _camera_type(row) == "uncontrolled":
            continue
        grouped[str(row.get("model") or "unknown")].append(row)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "eligible_rows",
                "GT_coverage",
                "pose_status_ok",
                "MegaSAM_success_rate",
                "certification_status_counts",
                "evidence_level_counts",
                "D1_valid_rate",
                "D1_acc_valid",
                "D1_rotation",
                "D1_translation",
                "D1_static",
                "D1_overall_clean_control",
                "failure_counts",
            ]
        )
        for model in sorted(grouped):
            items = grouped[model]
            ok = [row for row in items if row.get("d1_status") == "ok"]
            acc = [float(row["d1_camera_accuracy"]) for row in ok if row.get("d1_camera_accuracy") is not None]
            failures: dict[str, int] = defaultdict(int)
            certification_counts: dict[str, int] = defaultdict(int)
            evidence_counts: dict[str, int] = defaultdict(int)
            for row in items:
                certification_counts[str(row.get("d1_target_certification_status") or "missing")] += 1
                evidence_counts[str(row.get("d1_evidence_level") or "missing")] += 1
                if row.get("d1_status") != "ok":
                    failures[str(row.get("d1_status") or "unknown")] += 1
            family_scores: dict[str, float | str] = {}
            for family in ("rotation", "translation", "static"):
                family_acc = [
                    float(row["d1_camera_accuracy"])
                    for row in ok
                    if row.get("d1_control_family") == family and row.get("d1_camera_accuracy") is not None
                ]
                family_scores[family] = sum(family_acc) / len(family_acc) if family_acc else ""
            available_family_scores = [
                float(value)
                for value in family_scores.values()
                if value != ""
            ]
            overall = sum(available_family_scores) / len(available_family_scores) if available_family_scores else ""
            gt_valid = [row for row in items if row.get("d1_gt_valid")]
            writer.writerow(
                [
                    model,
                    len(items),
                    len(gt_valid) / len(items) if items else 0.0,
                    len(ok),
                    len(ok) / len(items) if items else 0.0,
                    json.dumps(dict(sorted(certification_counts.items())), sort_keys=True),
                    json.dumps(dict(sorted(evidence_counts.items())), sort_keys=True),
                    len(ok) / len(items) if items else 0.0,
                    sum(acc) / len(acc) if acc else "",
                    family_scores["rotation"],
                    family_scores["translation"],
                    family_scores["static"],
                    overall,
                    json.dumps(dict(sorted(failures.items())), sort_keys=True),
                ]
            )


def score_d1_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [score_d1_row(row, config) for row in rows]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score D1 camera movement accuracy from target and cached MegaSAM poses.")
    parser.add_argument("--input_jsonl", "--input-jsonl", "--input", dest="input_jsonl", required=True)
    parser.add_argument("--output_jsonl", "--output-jsonl", "--output", dest="output_jsonl", required=True)
    parser.add_argument("--summary_csv", "--summary-csv", dest="summary_csv", required=True)
    parser.add_argument("--megasam_cache_root", "--megasam-cache-root", "--pose-cache-root", dest="pose_cache_root", required=True)
    parser.add_argument("--pose-backend", required=True)
    parser.add_argument("--poses_file", "--poses-file", dest="poses_file", required=True)
    parser.add_argument("--default_frames", "--default-frames", dest="default_frames", type=int, required=True)
    parser.add_argument("--predicted-pose-type", required=True)
    parser.add_argument("--predicted-camera-convention", required=True)
    parser.add_argument("--target-camera-convention", required=True)
    parser.add_argument("--rot-scale-deg", type=float, required=True)
    parser.add_argument("--trans-scale", type=float, required=True)
    parser.add_argument("--yaw-weak-threshold-deg", type=float, required=True)
    parser.add_argument("--pan-weak-threshold", type=float, required=True)
    parser.add_argument("--static-rot-threshold-deg", type=float, required=True)
    parser.add_argument("--static-trans-threshold", type=float, required=True)
    parser.add_argument(
        "--sidecar-profile-gate",
        choices=("main", "certified_opencv"),
        required=True,
        help=(
            "Target sidecar validation gate. 'main' requires the canonical main-table profile; "
            "'certified_opencv' accepts certified OpenCV C2W sidecars after external QC."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = score_d1_rows(
        read_jsonl(args.input_jsonl),
        {
            "cache_root": args.pose_cache_root,
            "poses_file": args.poses_file,
            "default_frames": args.default_frames,
            "require_main_profile": args.sidecar_profile_gate == "main",
            "predicted_pose_type": args.predicted_pose_type,
            "predicted_camera_convention": args.predicted_camera_convention,
            "target_camera_convention": args.target_camera_convention,
            "rot_scale_deg": args.rot_scale_deg,
            "trans_scale": args.trans_scale,
            "yaw_weak_threshold_deg": args.yaw_weak_threshold_deg,
            "pan_weak_threshold": args.pan_weak_threshold,
            "static_rot_threshold_deg": args.static_rot_threshold_deg,
            "static_trans_threshold": args.static_trans_threshold,
        },
    )
    for row in rows:
        row["d1_pose_backend"] = args.pose_backend
    write_jsonl(rows, args.output_jsonl)
    write_summary_csv(rows, args.summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
