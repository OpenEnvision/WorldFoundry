"""GEN3C camera adapter (W2C matrices + intrinsics for ViPE cache rendering)."""

from __future__ import annotations

from pathlib import Path

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("gen3c")
class GEN3CAdapter:
    name = "gen3c"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        payload_type = "gen3c_w2c_intrinsics"
        return CameraPayload(
            payload_type=payload_type,
            payload={"w2cs": target.to_w2c()[None], "intrinsics": target.intrinsics[None]},
            target_trajectory=target,
            official_camera_entrypoint="w2cs/intrinsics cache rendering",
            coordinate_notes="OpenCV C2W converted to GEN3C W2C matrices",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"w2c_count": target.frame_count},
            ),
        )
