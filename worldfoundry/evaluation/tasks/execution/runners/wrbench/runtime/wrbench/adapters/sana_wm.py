from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory, write_json
from wrbench.adapters.base import register
from wrbench.contracts import (
    build_command_template,
    require_execution_contract,
    require_int,
    require_mapping,
    require_str,
)
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


@register("sana-wm")
class SanaWMAdapter:
    name = "sana_wm"

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
        key = canonical_model_key(model_name)
        execution = require_execution_contract(key)
        benchmark_profile = require_mapping(execution, "wrbench_benchmark_profile")
        benchmark_frames = require_int(benchmark_profile, "num_frames")
        benchmark_fps = require_int(benchmark_profile, "fps")
        entrypoint = require_str(execution, "entrypoint")
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        if int(target.frame_count) != benchmark_frames:
            raise ValueError(f"{key} requires contract frame count {benchmark_frames}, got {target.frame_count}")
        if int(target.fps) != benchmark_fps:
            raise ValueError(f"{key} requires contract fps {benchmark_fps}, got {target.fps}")

        camera_npy = out_dir / "sana_wm_camera_c2w.npy"
        intrinsics_npy = out_dir / "sana_wm_intrinsics.npy"
        np.save(camera_npy, target.to_c2w().astype(np.float32))
        np.save(intrinsics_npy, target.intrinsics.astype(np.float32))

        command_template = build_command_template(
            execution,
            values={
                "image": "<first_frame_image>",
                "prompt": "<prompt_txt>",
                "output_dir": "<output_dir>",
                "name": "<stem>",
                "camera": str(camera_npy),
                "intrinsics": str(intrinsics_npy),
                "num_frames": int(target.frame_count),
                "fps": int(target.fps),
            },
        )
        request_json = write_json(
            out_dir / "sana_wm_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "runtime_parameters": require_mapping(execution, "runtime_parameters"),
                "wrbench_benchmark_profile": benchmark_profile,
                "official_quality_profile": require_mapping(execution, "official_quality_profile"),
                "official_runtime": require_mapping(execution, "official_runtime"),
                "official_config_constraints": require_mapping(execution, "official_config_constraints"),
                "forbidden_workarounds": require_mapping(execution, "forbidden_workarounds"),
                "wrbench_policy": require_mapping(execution, "wrbench_policy"),
            },
        )

        payload_type = "sana_wm_camera_npy"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "camera_npy": str(camera_npy),
                "intrinsics_npy": str(intrinsics_npy),
                "request_json": str(request_json),
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="--camera OpenCV C2W .npy plus --intrinsics .npy",
            coordinate_notes="SANA-WM accepts camera-to-world matrices with shape (F, 4, 4); intrinsics are supplied explicitly to avoid Pi3X downloads.",
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="direct_frame_pose_payload",
                model_payload_summary={
                    "camera_shape": [int(target.frame_count), 4, 4],
                    "intrinsics_shape": [int(target.frame_count), 3, 3],
                    "entrypoint": entrypoint,
                    "prompt_mode": "file_path_required",
                    "model_path": require_str(require_mapping(execution, "runtime_parameters"), "model_path"),
                    "benchmark_frame_count": benchmark_frames,
                    "benchmark_fps": benchmark_fps,
                    "official_quality_frame_count": require_int(require_mapping(execution, "official_quality_profile"), "num_frames"),
                    "official_softmax_every_n": require_int(require_mapping(execution, "official_config_constraints"), "model.softmax_every_n"),
                    "official_runtime": require_mapping(execution, "official_runtime"),
                },
            ),
        )
