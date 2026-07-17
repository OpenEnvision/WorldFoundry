"""Wan2.1/2.2 Fun camera-control adapter (CameraCtrl W2C pose text)."""

from __future__ import annotations

from pathlib import Path

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory, write_cameractrl_txt
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("wan21-fun-14b-cam", "wan21-fun-1p3b-cam", "wan22-fun-a14b-cam", "wan22-fun-5b-cam")
class WanFunAdapter:
    name = "wan_fun"

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        pose_txt = write_cameractrl_txt(
            Path(work_dir or ".") / f"{model_name.replace('-', '_')}_cameractrl.txt", target
        )
        payload_type = "wan_fun_pose_txt"
        return CameraPayload(
            payload_type=payload_type,
            payload={"pose_txt": str(pose_txt), "control_camera_video": None},
            target_trajectory=target,
            official_camera_entrypoint="control_camera_video",
            coordinate_notes=(
                "canonical OpenCV C2W converted to CameraCtrl W2C rows; tensor conversion "
                "happens in upstream VideoX-Fun environment"
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"pose_row_count": target.frame_count},
            ),
        )
