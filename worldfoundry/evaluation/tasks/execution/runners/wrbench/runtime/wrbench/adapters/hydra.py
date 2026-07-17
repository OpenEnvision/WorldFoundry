"""HyDRA camera adapter: cond_cam/tgt_cam split camera JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory, write_json
from wrbench.adapters.base import register
from wrbench.adapters.operators.hydra_trajectory import hydra_camera_payload_dict, hydra_yaw_target_c2w
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


def _relative_yaw_peak_deg(c2w: np.ndarray) -> float:
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    if len(yaw) == 0:
        return 0.0
    return float(yaw[int(np.argmax(np.abs(yaw)))])


def _is_hydra_yaw_only(camera_type: str) -> bool:
    text = str(camera_type or "")
    if text in {"yaw_LR", "yaw_RL"}:
        return True
    return "yaw:" in text and all(token not in text for token in ("pan:", "dolly:", "crane:", "pitch:", "roll:"))


def hydra_model_target_trajectory(target: CameraTrajectory) -> CameraTrajectory:
    if not _is_hydra_yaw_only(target.camera_type):
        return target
    peak_deg = _relative_yaw_peak_deg(target.to_c2w())
    return CameraTrajectory(
        c2w=hydra_yaw_target_c2w(-peak_deg, target.frame_count),
        intrinsics=target.intrinsics,
        camera_type=target.camera_type,
        fps=target.fps,
        source=target.source,
        conversion_mode="hydra_smoothstep_cond_cam_tgt_cam_json",
    )


@register("hydra")
class HyDRAAdapter:
    name = "hydra"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        target = hydra_model_target_trajectory(target)
        target_c2w = target.to_c2w()
        cond_c2w = np.repeat(target_c2w[0:1], target.frame_count, axis=0)
        payload = {
            "cond_cam": hydra_camera_payload_dict(cond_c2w),
            "tgt_cam": hydra_camera_payload_dict(target_c2w),
        }
        out = write_json(ensure_work_dir(work_dir) / "hydra_split_camera.json", payload)
        payload_type = "hydra_split_camera_json"
        meta = adapter_taxonomy_metadata(
            model_name=model_name,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            model_payload_summary={
                "cond_cam_frame_count": target.frame_count,
                "tgt_cam_frame_count": target.frame_count,
                "embedding_sample_stride": 4,
                "relative_embedding_shape": [20, 12],
            },
        )
        meta.update({
            "cond_cam_source": "repeated_anchor_pose",
            "cond_cam_frame_count": target.frame_count,
            "tgt_cam_frame_count": target.frame_count,
            "embedding_sample_stride": 4,
            "relative_embedding_shape": [20, 12],
            "translation_divisor": 100,
            "payload_coordinate_convention": "hydra_pretransform_c2w",
            "yaw_schedule": "smoothstep_go_return" if _is_hydra_yaw_only(target.camera_type) else "direct_resample",
        })
        return CameraPayload(
            payload_type=payload_type,
            payload={"camera_json": str(out)},
            target_trajectory=target,
            official_camera_entrypoint="cond_cam/tgt_cam split camera JSON",
            coordinate_notes="Unified target trajectory is OpenCV c2w; JSON payload is pretransformed for HyDRA load_condition_json",
            calibration_status=amp.calibration_status,
            metadata=meta,
        )
