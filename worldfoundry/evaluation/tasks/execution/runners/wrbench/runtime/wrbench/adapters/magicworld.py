"""MagicWorld camera adapter (action segments -> native trajectory rows)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


def _camera_profile_name(camera_type: str) -> str:
    text = str(camera_type or "").strip()
    if text in {"yaw_LR", "yaw_RL", "pan_LR", "pan_RL", "static"}:
        return text
    if text.startswith("yaw:left:") and ",yaw:right:" in text:
        return "yaw_LR"
    if text.startswith("yaw:right:") and ",yaw:left:" in text:
        return "yaw_RL"
    if text.startswith("pan:left:") and ",pan:right:" in text:
        return "pan_LR"
    if text.startswith("pan:right:") and ",pan:left:" in text:
        return "pan_RL"
    if text.startswith("static"):
        return "static"
    raise ValueError(f"MagicWorld only supports yaw_LR/yaw_RL/pan_LR/pan_RL/static, got {camera_type!r}")


def _action_segments_for(camera_type: str, num_frames: int) -> list[tuple[str, int]]:
    half = int(num_frames) // 2
    rest = int(num_frames) - half
    profile = _camera_profile_name(camera_type)
    if profile == "yaw_LR":
        return [("Yaw Left", half), ("Yaw Right", rest)]
    if profile == "yaw_RL":
        return [("Yaw Right", half), ("Yaw Left", rest)]
    if profile == "pan_LR":
        return [("Pan Left", half), ("Pan Right", rest)]
    if profile == "pan_RL":
        return [("Pan Right", half), ("Pan Left", rest)]
    if profile == "static":
        return [("Static", int(num_frames))]
    raise ValueError(f"MagicWorld only supports yaw_LR/yaw_RL/pan_LR/pan_RL/static, got {camera_type!r}")


def _yaw_peak_abs_deg(trajectory: CameraTrajectory) -> float:
    c2w = trajectory.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    return abs(float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0)


def _native_rows_to_trajectory(rows: np.ndarray, *, camera_type: str, fps: int) -> CameraTrajectory:
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 19:
        raise ValueError(f"MagicWorld native trajectory must have shape (N, 19), got {arr.shape}")
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], len(arr), axis=0)
    w2c[:, :3, :4] = arr[:, 7:].reshape(len(arr), 3, 4)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    intr = np.repeat(np.eye(3, dtype=np.float32)[None], len(arr), axis=0)
    intr[:, 0, 0] = arr[:, 1]
    intr[:, 1, 1] = arr[:, 2]
    intr[:, 0, 2] = arr[:, 3]
    intr[:, 1, 2] = arr[:, 4]
    return CameraTrajectory(
        c2w=c2w,
        intrinsics=intr,
        camera_type=camera_type,
        fps=fps,
        source="magicworld_action_segments_native_rows",
        conversion_mode="magicworld_native_rows_w2c_to_opencv_c2w",
    )


def _trajectory_to_native_rows(trajectory: CameraTrajectory) -> np.ndarray:
    c2w = trajectory.to_c2w()
    w2c = np.linalg.inv(c2w).astype(np.float32)
    intr = np.asarray(trajectory.intrinsics, dtype=np.float32)
    rows = np.zeros((trajectory.frame_count, 19), dtype=np.float32)
    rows[:, 1] = intr[:, 0, 0]
    rows[:, 2] = intr[:, 1, 1]
    rows[:, 3] = intr[:, 0, 2]
    rows[:, 4] = intr[:, 1, 2]
    rows[:, 7:] = w2c[:, :3, :4].reshape(trajectory.frame_count, 12)
    anchor = rows[:1].copy()
    return np.concatenate([anchor, rows], axis=0)


@register("magicworld")
class MagicWorldAdapter:
    name = "magicworld"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        canonical_target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        payload_type = "magicworld_action_segments"
        camera_type = _camera_profile_name(str(canonical_target.camera_type or ""))
        action_segments = _action_segments_for(camera_type, num_frames)
        total_angle_deg = _yaw_peak_abs_deg(canonical_target)
        native_rows = _trajectory_to_native_rows(canonical_target)
        entrypoint = "action_segments + canonical_target_rows"
        sampling_rule = "WRBench deterministic canonical target rows paired with MagicWorld action segment labels"
        if len(native_rows) != int(num_frames) + 1:
            raise ValueError(
                f"MagicWorld native trajectory returned {len(native_rows)} rows; expected {int(num_frames) + 1}"
            )
        target = _native_rows_to_trajectory(native_rows[1 : int(num_frames) + 1], camera_type=camera_type, fps=trajectory.fps)
        precision = adapter_taxonomy_metadata(
            model_name=model_name,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            certification_kind="direct_frame_pose_payload",
            model_payload_summary={
                "action_segments": [
                    {"action": str(action), "frames": int(frames)}
                    for action, frames in action_segments
                ],
                "total_angle_deg": total_angle_deg,
                "native_trajectory_rows": int(len(native_rows)),
            },
            control_sample_kind="action_matrix_or_pose",
            control_sample_count=len(action_segments),
            source_frame_indices=[sum(int(frames) for _action, frames in action_segments[:idx]) for idx in range(len(action_segments))],
            sampling_rule=sampling_rule,
            model_control_extra={
                "control_contract": "exact_model_action_payload",
                "action_segments": [
                    {"action": str(action), "frames": int(frames)}
                    for action, frames in action_segments
                ],
                "total_angle_deg": total_angle_deg,
                "native_trajectory_rows": int(len(native_rows)),
            },
        )
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "action_segments": action_segments,
                "total_angle_deg": total_angle_deg,
                "native_rows": native_rows.tolist(),
            },
            target_trajectory=target,
            official_camera_entrypoint=entrypoint,
            coordinate_notes="Native MagicWorld 19-column rows inverted from W2C to OpenCV C2W for QC",
            calibration_status=amp.calibration_status,
            metadata=precision,
        )
