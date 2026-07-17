"""LingBot camera adapter (pose arrays; optional WASD action for act-mode)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.actions import parse_camera_script
from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


@register("lingbot-world", "lingbot-world-act")
class LingBotAdapter:
    name = "lingbot"

    def _wasd_action(self, trajectory: CameraTrajectory) -> np.ndarray:
        actions = np.zeros((trajectory.frame_count, 4), dtype=np.float32)
        try:
            script = parse_camera_script(trajectory.camera_type)
        except ValueError:
            return actions
        idx = 0
        for action in script.actions:
            frames = int(action.frames or 0)
            if action.kind == "dolly":
                if action.direction == "forward":
                    actions[idx : idx + frames, 0] = float(action.amount or 1.0)
                elif action.direction in {"back", "backward"}:
                    actions[idx : idx + frames, 2] = float(action.amount or 1.0)
            elif action.kind == "pan":
                if action.direction == "left":
                    actions[idx : idx + frames, 1] = float(action.amount or 1.0)
                elif action.direction == "right":
                    actions[idx : idx + frames, 3] = float(action.amount or 1.0)
            elif action.kind in {"crane", "roll"}:
                raise ValueError(f"LingBot-Act WASD path does not support {action.kind} actions")
            idx += frames
        return actions

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        key = canonical_model_key(model_name)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        ks4 = np.stack(
            [target.intrinsics[:, 0, 0], target.intrinsics[:, 1, 1], target.intrinsics[:, 0, 2], target.intrinsics[:, 1, 2]],
            axis=1,
        ).astype(np.float32)
        payload_type = "lingbot_pose_arrays"
        metadata = adapter_taxonomy_metadata(
            model_name=key,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type=payload_type,
            model_payload_summary={"pose_count": target.frame_count, "wasd_source": None},
        )
        action_matrix = None
        if key == "lingbot-world-act":
            action_matrix = self._wasd_action(target)
            wasd_source = "camera_script" if bool(np.max(np.abs(action_matrix)) > 0) else "zero_for_rotation"
            metadata.update({"control_type": "act", "wasd_source": wasd_source})
            metadata["model_control_timeline"] = adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={"pose_count": target.frame_count, "wasd_source": wasd_source},
                control_sample_kind="action_matrix_or_pose",
                sampling_rule="wasd_action_matrix_when_script_maps_to_translation_else_pose_arrays",
                model_control_extra={"wasd_source": wasd_source, "wasd_shape": list(action_matrix.shape)},
            )["model_control_timeline"]
            metadata["model_payload_summary"] = {
                "pose_count": target.frame_count,
                "wasd_source": wasd_source,
            }
        return CameraPayload(
            payload_type=payload_type,
            payload={"poses_c2ws": target.to_c2w(), "poses_Ks": ks4, "action_matrix": action_matrix},
            target_trajectory=target,
            official_camera_entrypoint="poses_c2ws/poses_Ks",
            coordinate_notes="LingBot consumes OpenCV-style C2W pose arrays plus fx/fy/cx/cy intrinsics",
            calibration_status=amp.calibration_status,
            metadata=metadata,
        )
