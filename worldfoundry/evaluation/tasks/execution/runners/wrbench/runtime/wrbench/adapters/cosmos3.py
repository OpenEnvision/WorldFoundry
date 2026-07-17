"""Cosmos3 camera adapter using NVIDIA's official 9D camera-pose action convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


def _backward_framewise_rot6d(c2w: np.ndarray) -> np.ndarray:
    """Match Cosmos Framework pose_abs_to_rel(..., backward_framewise, rot6d)."""

    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) < 2:
        raise ValueError(f"Expected C2W poses shaped [F,4,4] with F>=2, got {c2w.shape}.")
    delta = np.linalg.inv(c2w[:-1]) @ c2w[1:]
    # Cosmos rot6d stores rotation columns 0 and 1, each as xyz.
    rot6d = delta[:, :3, :2].transpose(0, 2, 1).reshape(len(delta), 6)
    return np.concatenate((delta[:, :3, 3], rot6d), axis=1).astype(np.float32)


@register("cosmos3-nano-generator")
class Cosmos3CameraAdapter:
    name = "cosmos3"

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
        actions = _backward_framewise_rot6d(target.to_c2w().astype(np.float64))
        metadata = adapter_taxonomy_metadata(
            model_name=key,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type="cosmos3_camera_pose_actions",
            model_payload_summary={"action_shape": list(actions.shape), "domain_name": "camera_pose"},
            control_sample_kind="framewise_relative_pose_action",
            control_sample_count=len(actions),
            source_frame_indices=list(range(1, target.frame_count)),
            sampling_rule="one_backward_framewise_9d_action_per_visual_transition",
            model_control_extra={
                "rotation_format": "rot6d_column_based",
                "pose_convention": "backward_framewise",
                "translation_scale": 1.0,
            },
        )
        return CameraPayload(
            payload_type="cosmos3_camera_pose_actions",
            payload={
                "action_mode": "forward_dynamics",
                "action_chunk_size": int(len(actions)),
                "domain_name": "camera_pose",
                "raw_actions": actions,
                "resolution_tier": 480,
                "view_point": "ego_view",
                "enable_sound": False,
            },
            target_trajectory=target,
            official_camera_entrypoint="CosmosActionCondition(mode='forward_dynamics', domain_name='camera_pose')",
            coordinate_notes=(
                "OpenCV camera-to-world poses in meters are converted to inv(C2W[t]) @ C2W[t+1], "
                "then encoded as translation(3) plus column-based rot6d(6)."
            ),
            calibration_status=amp.calibration_status,
            metadata=metadata,
        )


__all__ = ["Cosmos3CameraAdapter"]
