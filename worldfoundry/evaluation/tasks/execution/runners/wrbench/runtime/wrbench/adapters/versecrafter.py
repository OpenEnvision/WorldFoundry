"""VerseCrafter camera adapter (Blender C2W NPZ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


def _opencv_c2w_to_blender_c2w(c2w: np.ndarray) -> np.ndarray:
    arr = np.asarray(c2w, dtype=np.float32).copy()
    basis = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    arr[:, :3, :3] = arr[:, :3, :3] @ basis
    return arr


@register("versecrafter")
class VerseCrafterAdapter:
    name = "versecrafter"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        out = ensure_work_dir(work_dir) / "custom_camera_trajectory.npz"
        np.savez(out, extrinsics=_opencv_c2w_to_blender_c2w(target.to_c2w()), intrinsics=target.intrinsics)
        payload_type = "versecrafter_npz"
        return CameraPayload(
            payload_type=payload_type,
            payload={"trajectory_npz": str(out)},
            target_trajectory=target,
            official_camera_entrypoint="custom_camera_trajectory.npz",
            coordinate_notes="OpenCV C2W converted to VerseCrafter Blender C2W extrinsics",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"pose_count": target.frame_count},
            ),
        )
