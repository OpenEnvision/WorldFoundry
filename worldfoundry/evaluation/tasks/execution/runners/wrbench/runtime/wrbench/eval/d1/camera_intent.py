"""D1 prompt-camera alignment (CamAlign) intent scoring from recovered poses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCORER_VERSION = "d1_camera_intent_v1"
THRESHOLD_VERSION = "d1_camera_intent_thresholds_v1"
FORMULA_VERSION = "direction_first_magnitude_norm_v1"
POSE_CONVENTION = "opencv_c2w"
SUPPORTED_INTENTS = frozenset({"yaw_LR", "yaw_RL", "static"})

WEAK_YAW_DEG = 2.0
DEFAULT_NOMINAL_YAW_DEG = 30.0
STATIC_ROT_OBVIOUS_DEG = 5.0
STATIC_TRANS_OBVIOUS = 0.10


def _base_result(
    *,
    status: str,
    score: float | None,
    direction: str | None,
    magnitude: float | None,
    metric_scope: str,
    aggregation_scope: str,
) -> dict[str, Any]:
    return {
        "d1_camalign_score": score,
        "d1_camalign_status": status,
        "d1_camalign_direction": direction,
        "d1_camalign_magnitude": magnitude,
        "d1_camalign_metric_scope": metric_scope,
        "d1_camalign_aggregation_scope": aggregation_scope,
        "d1_camalign_scorer_version": SCORER_VERSION,
        "d1_camalign_threshold_version": THRESHOLD_VERSION,
        "d1_camalign_formula_version": FORMULA_VERSION,
        "d1_camalign_pose_convention": POSE_CONVENTION,
    }


def _validate_poses(poses_c2w: Any) -> np.ndarray | None:
    try:
        poses = np.asarray(poses_c2w, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        return None
    if not np.all(np.isfinite(poses)):
        return None
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(poses[:, 3, :], expected_last_row[None], atol=1e-6):
        return None
    return poses


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


def _relative_poses(poses: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.inv(poses[0]) @ poses
    except np.linalg.LinAlgError:
        return None


def _peak_signed(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(values[int(np.argmax(np.abs(values)))])


def _yaw_degrees(relative: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 0, 0]))


def _motion_strength(relative: np.ndarray) -> tuple[float, float]:
    rot_deg = 0.0 if len(relative) == 0 else max(_rotation_angle_deg(pose[:3, :3]) for pose in relative)
    trans = relative[:, :3, 3]
    trans_norm = math.sqrt(float(np.max(np.sum(trans * trans, axis=1)))) if len(trans) else 0.0
    return float(rot_deg), float(trans_norm)


def _observed_yaw_direction(yaw_peak_deg: float, weak_threshold_deg: float) -> str:
    if abs(yaw_peak_deg) < weak_threshold_deg:
        return "static"
    return "yaw_RL" if yaw_peak_deg > 0.0 else "yaw_LR"


def _scopes(intent: str) -> tuple[str, str]:
    if intent in {"yaw_LR", "yaw_RL"}:
        return "D1-CamAlign_common_yaw", "matched_yaw_common_denominator_candidate"
    if intent == "static":
        return "D1-CamAlign_static_hold", "static_hold_candidate"
    return "D1-CamAlign_unsupported", "unsupported"


def score_camera_intent(
    poses_c2w: Any,
    *,
    intent: str,
    yaw_weak_threshold_deg: float = WEAK_YAW_DEG,
    yaw_full_score_deg: float = DEFAULT_NOMINAL_YAW_DEG,
    static_rot_obvious_deg: float = STATIC_ROT_OBVIOUS_DEG,
    static_trans_obvious: float = STATIC_TRANS_OBVIOUS,
) -> dict[str, Any]:
    """Score whether a recovered pose stack expresses the requested camera intent."""
    intent = str(intent or "").strip()
    metric_scope, aggregation_scope = _scopes(intent)
    if intent not in SUPPORTED_INTENTS:
        return _base_result(
            status="unsupported_intent",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    poses = _validate_poses(poses_c2w)
    if poses is None:
        return _base_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    relative = _relative_poses(poses)
    if relative is None:
        return _base_result(
            status="invalid_pose",
            score=None,
            direction=None,
            magnitude=None,
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    yaw_peak_deg = _peak_signed(_yaw_degrees(relative))
    yaw_magnitude_deg = abs(yaw_peak_deg)
    rot_strength_deg, trans_strength = _motion_strength(relative)

    if intent in {"yaw_LR", "yaw_RL"}:
        observed_direction = _observed_yaw_direction(yaw_peak_deg, yaw_weak_threshold_deg)
        if observed_direction == "static":
            return _base_result(
                status="weak_motion",
                score=0.0,
                direction=observed_direction,
                magnitude=float(yaw_magnitude_deg),
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
            )
        if observed_direction != intent:
            return _base_result(
                status="direction_mismatch",
                score=0.0,
                direction=observed_direction,
                magnitude=float(yaw_magnitude_deg),
                metric_scope=metric_scope,
                aggregation_scope=aggregation_scope,
            )
        denominator = max(float(yaw_full_score_deg) - float(yaw_weak_threshold_deg), 1e-6)
        score = (yaw_magnitude_deg - float(yaw_weak_threshold_deg)) / denominator
        return _base_result(
            status="ok",
            score=float(np.clip(score, 0.0, 1.0)),
            direction=observed_direction,
            magnitude=float(yaw_magnitude_deg),
            metric_scope=metric_scope,
            aggregation_scope=aggregation_scope,
        )

    rot_motion = rot_strength_deg / max(float(static_rot_obvious_deg), 1e-6)
    trans_motion = trans_strength / max(float(static_trans_obvious), 1e-12)
    normalized_motion = max(rot_motion, trans_motion)
    static_score = float(np.clip(1.0 - normalized_motion, 0.0, 1.0))
    status = "ok" if static_score >= 1.0 else "motion_detected"
    return _base_result(
        status=status,
        score=static_score,
        direction="static",
        magnitude=float(normalized_motion),
        metric_scope=metric_scope,
        aggregation_scope=aggregation_scope,
    )


def score_camera_intent_row(row: dict[str, Any], poses_c2w: Any) -> dict[str, Any]:
    intent = row.get("camera_type") or row.get("camera") or row.get("intent")
    scored = dict(row)
    scored.update(score_camera_intent(poses_c2w, intent=str(intent or "")))
    return scored
