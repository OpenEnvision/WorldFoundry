"""Spatia and InSpatio-World camera adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.actions import parse_camera_script
from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.adapters.operators.trajectory_utils import write_json_w2c_file
from wrbench.payload import CameraPayload
from wrbench.trajectory import CameraTrajectory


@register("spatia")
class SpatiaAdapter:
    name = "spatia"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        out = ensure_work_dir(work_dir) / "spatia_w2c.jsonl"
        write_json_w2c_file(out, target.to_w2c())
        payload_type = "spatia_w2c_jsonl"
        return CameraPayload(
            payload_type=payload_type,
            payload={"w2c_trajectory_file": str(out), "intrinsics": target.intrinsics.tolist()},
            target_trajectory=target,
            official_camera_entrypoint="W2C trajectory JSON-lines file",
            coordinate_notes="OpenCV C2W converted to one W2C matrix JSON row per generated frame",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=model_name,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"w2c_row_count": target.frame_count},
            ),
        )


@register("inspatio-world")
class InSpatioAdapter:
    name = "inspatio"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        target, amp = model_target_trajectory(trajectory, model_name, num_frames)
        c2w = target.to_c2w()
        pitch = np.degrees(np.arctan2(c2w[:, 2, 1], c2w[:, 1, 1])).astype(np.float32)
        yaw = np.degrees(np.arctan2(c2w[:, 0, 2], c2w[:, 0, 0])).astype(np.float32)
        translation = c2w[:, :3, 3]
        dominant_axis = int(np.argmax(np.max(np.abs(translation), axis=0)))
        displacement = translation[:, dominant_axis].astype(np.float32)
        if trajectory.camera_type:
            script = parse_camera_script(trajectory.camera_type)
            allowed = {"static", "yaw", "pitch", "pan", "dolly"}
            unsupported = sorted({a.kind for a in script.actions if a.kind not in allowed})
            if unsupported:
                raise ValueError(f"InSpatio official payload does not support actions: {unsupported}")
        out = ensure_work_dir(work_dir) / "inspatio_per_frame_trajectory.txt"
        out.write_text(
            " ".join(f"{v:.8f}" for v in pitch) + "\n"
            + " ".join(f"{v:.8f}" for v in yaw) + "\n"
            + " ".join(f"{v:.8f}" for v in displacement) + "\n",
            encoding="utf-8",
        )
        metadata = adapter_taxonomy_metadata(
            model_name=model_name,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type="inspatio_per_frame_action_txt",
            model_payload_summary={"per_frame_txt_rows": 3, "frame_count": target.frame_count},
            control_sample_kind="numeric_per_frame",
            sampling_rule="one_pitch_yaw_displacement_triplet_per_target_frame",
            model_control_extra={"numeric_rows": ["pitch_x_up", "yaw_y_left", "displacement"]},
        )
        metadata["trajectory_txt_rows"] = ["pitch_x_up", "yaw_y_left", "displacement"]
        return CameraPayload(
            payload_type="inspatio_per_frame_action_txt",
            payload={"trajectory_txt": str(out), "rotation_only": bool(np.max(np.abs(yaw)) > 0 and np.max(np.abs(displacement)) == 0), "translation_only": bool(np.max(np.abs(displacement)) > 0 and np.max(np.abs(yaw)) == 0)},
            target_trajectory=target,
            official_camera_entrypoint="3-line pitch/yaw/displacement trajectory txt",
            coordinate_notes="Official InSpatio keyframe txt written with one value per generated target frame",
            calibration_status=amp.calibration_status,
            metadata=metadata,
        )
