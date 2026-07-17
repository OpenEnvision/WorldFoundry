"""ReCamMaster camera adapter (21-latent relative pose embedding)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("recammaster")
class ReCamMasterAdapter:
    name = "recammaster"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        latent = target.resample(21)
        rel = np.linalg.inv(latent.to_c2w()[0]) @ latent.to_c2w()
        embedding = rel[:, :3, :4].reshape(1, 21, 12).astype(np.float32)
        payload_type = "recammaster_relative_pose_embedding"
        meta = adapter_taxonomy_metadata(
            model_name=model_name,
            amp=amp,
            target=latent,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            model_payload_summary={"latent_pose_count": 21, "embedding_shape": [1, 21, 12]},
            control_sample_kind="latent_pose",
            control_sample_count=21,
            sampling_rule="resample_to_21_relative_poses",
            model_control_extra={
                "embedding_shape": [1, 21, 12],
                "relative_pose_reference": "first_latent_pose",
            },
        )
        meta["latent_pose_count"] = 21
        return CameraPayload(
            payload_type=payload_type,
            payload={"camera_embedding": embedding},
            target_trajectory=latent,
            official_camera_entrypoint="relative pose embedding",
            coordinate_notes="Canonical C2W resampled to 21 latent relative poses and flattened to (1,21,12)",
            calibration_status=amp.calibration_status,
            metadata=meta,
        )
