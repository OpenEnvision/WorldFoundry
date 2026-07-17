"""LiveWorld camera adapter (geometry NPZ with anchor pose + target poses)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("liveworld")
class LiveWorldAdapter:
    name = "liveworld"
    default_frames_per_iter = 65

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
        template_npz: str | Path | None = None,
    ) -> CameraPayload:
        # Clamp to default_frames_per_iter when num_frames exceeds the model's
        # native window; deployment-time enforcement is handled upstream.
        frames_per_iter = min(int(num_frames), self.default_frames_per_iter)

        target, amp = model_target_trajectory(trajectory, model_name, frames_per_iter)
        out = ensure_work_dir(work_dir) / "liveworld_geometry_unified.npz"
        target_c2w = target.to_c2w()

        if template_npz is not None:
            template_path = Path(template_npz)
            if not template_path.exists():
                raise FileNotFoundError(f"LiveWorld template_npz not found: {template_path}")
            template = dict(np.load(template_path, allow_pickle=False))
            anchor_pose = np.asarray(template["poses_c2w"][0:1], dtype=np.float32)
            composed_poses = np.concatenate([anchor_pose, target_c2w], axis=0)
            depth_key = next((key for key in ("depths", "depth") if key in template), None)
            if depth_key is not None:
                depth_arr = np.asarray(template[depth_key])
                if len(depth_arr) != len(composed_poses):
                    raise ValueError(
                        f"LiveWorld template depth length {len(depth_arr)} "
                        f"!= pose count {len(composed_poses)}"
                    )
            payload_npz = {key: template[key] for key in template}
            payload_npz["poses_c2w"] = composed_poses.astype(np.float32)
        else:
            anchor = np.eye(4, dtype=np.float32)[None]
            composed_poses = np.concatenate([anchor, target_c2w], axis=0)
            payload_npz = {
                "poses_c2w": composed_poses,
                "intrinsics": np.concatenate([target.intrinsics[:1], target.intrinsics], axis=0),
                "height": int(height),
                "width": int(width),
            }

        np.savez(out, **payload_npz)
        payload_type = "liveworld_geometry_npz"
        meta = adapter_taxonomy_metadata(
            model_name=model_name,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            model_payload_summary={
                "geometry_pose_count": int(composed_poses.shape[0]),
                "target_pose_count": int(target.frame_count),
                "template_npz": str(template_npz) if template_npz is not None else None,
            },
            target_frame_indices=list(range(int(target.frame_count))),
            control_sample_kind="geometry_pose",
            control_sample_count=int(target.frame_count) + 1,
            source_frame_indices=[0] + list(range(int(target.frame_count))),
            sampling_rule="anchor_pose_plus_target_pose_sequence",
            model_control_extra={
                "anchor_frame_index": 0,
                "target_pose_start_index": 1,
            },
        )
        meta.update(
            {
                "anchor_frame_index": 0,
                "target_pose_start_index": 1,
                "num_generated_frames": target.frame_count,
                "used_pose_indices": list(range(target.frame_count + 1)),
                "unused_pose_indices": [],
                "frames_per_iter": target.frame_count,
                "pose_frame_count": target.frame_count + 1,
                "depth_frame_count": len(payload_npz.get("depths", [])) if "depths" in payload_npz else None,
            }
        )
        return CameraPayload(
            payload_type=payload_type,
            payload={"geometry_npz": str(out), "template_npz": str(template_npz) if template_npz else None},
            target_trajectory=target,
            official_camera_entrypoint="trajectory-template geometry.npz",
            coordinate_notes="Pose index 0 is anchor; target generated poses start at index 1; sidecar target_c2w is target-only",
            calibration_status=amp.calibration_status,
            metadata=meta,
        )
