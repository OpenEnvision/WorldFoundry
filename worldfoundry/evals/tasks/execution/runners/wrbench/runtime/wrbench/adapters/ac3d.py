"""AC3D camera adapter using its native RealEstate10K pose-row convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


@register("ac3d")
class AC3DCameraAdapter:
    name = "ac3d"

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
        del work_dir, device
        key = canonical_model_key(model_name)
        target, amp = model_target_trajectory(trajectory, key, int(num_frames))
        c2w = target.to_c2w().astype(np.float64)
        w2c = np.linalg.inv(c2w)

        rows = np.zeros((target.frame_count, 19), dtype=np.float64)
        rows[:, 0] = np.arange(target.frame_count, dtype=np.float64)
        rows[:, 1] = target.intrinsics[:, 0, 0] / float(width)
        rows[:, 2] = target.intrinsics[:, 1, 1] / float(height)
        rows[:, 3] = target.intrinsics[:, 0, 2] / float(width)
        rows[:, 4] = target.intrinsics[:, 1, 2] / float(height)
        rows[:, 7:19] = w2c[:, :3, :4].reshape(target.frame_count, 12)

        payload_type = "ac3d_realestate10k_pose_rows"
        metadata = adapter_taxonomy_metadata(
            model_name=key,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            model_payload_summary={
                "pose_row_shape": list(rows.shape),
                "intrinsics_normalized_by": [int(width), int(height)],
            },
            control_sample_kind="realestate10k_absolute_w2c_pose_row",
            control_sample_count=target.frame_count,
            source_frame_indices=list(range(target.frame_count)),
            sampling_rule="one_absolute_w2c_pose_row_per_source_video_frame_at_stride_1",
            model_control_extra={
                "pose_columns": "timestamp fx fy cx cy unused unused w2c_3x4",
                "official_relative_pose": True,
                "zero_translation_at_first_frame": True,
            },
        )
        return CameraPayload(
            payload_type=payload_type,
            payload={"camera_pose_rows": rows},
            target_trajectory=target,
            official_camera_entrypoint="RealEstate10KPoseControlnetDataset(camera pose rows)",
            coordinate_notes=(
                "OpenCV C2W is inverted to row-major W2C 3x4. Intrinsics are normalized by source "
                "width/height; official AC3D reanchors the first camera pose before Plucker encoding."
            ),
            calibration_status=amp.calibration_status,
            metadata=metadata,
        )


__all__ = ["AC3DCameraAdapter"]
