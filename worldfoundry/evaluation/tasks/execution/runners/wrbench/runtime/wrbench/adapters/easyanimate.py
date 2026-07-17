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


_OFFICIAL_CAMERACTRL_INTRINSICS = (0.532139961, 0.946026558, 0.5, 0.5)
_OFFICIAL_CAMERACTRL_POSE_FPS = 24


def _official_pose_stride(fps: int) -> int:
    if int(fps) <= 0:
        raise ValueError("EasyAnimate fps must be positive")
    if _OFFICIAL_CAMERACTRL_POSE_FPS % int(fps) != 0:
        raise ValueError(
            f"EasyAnimate official control_camera_txt decimation requires "
            f"{_OFFICIAL_CAMERACTRL_POSE_FPS} % fps == 0, got fps={fps}"
        )
    return _OFFICIAL_CAMERACTRL_POSE_FPS // int(fps)


def easyanimate_camera_rows(trajectory: CameraTrajectory, *, fps: int) -> np.ndarray:
    """Build EasyAnimate 19-column CameraCtrl rows.

    The official script treats the text file as a 24 fps pose stream and then
    samples it with ``[:: int(24 // fps)]``.  We therefore write a dense 24 fps
    pose stream so the official decimation lands on one control per output
    frame.
    """

    stride = _official_pose_stride(fps)
    pose_frame_count = (int(trajectory.frame_count) - 1) * stride + 1
    dense = trajectory.resample(pose_frame_count)
    w2c = dense.to_w2c()
    rows = np.zeros((dense.frame_count, 19), dtype=np.float32)
    rows[:, 1] = _OFFICIAL_CAMERACTRL_INTRINSICS[0]
    rows[:, 2] = _OFFICIAL_CAMERACTRL_INTRINSICS[1]
    rows[:, 3] = _OFFICIAL_CAMERACTRL_INTRINSICS[2]
    rows[:, 4] = _OFFICIAL_CAMERACTRL_INTRINSICS[3]
    rows[:, 7:] = w2c[:, :3, :4].reshape(dense.frame_count, 12)
    return rows


def write_easyanimate_camera_txt(path: str | Path, trajectory: CameraTrajectory, *, fps: int) -> tuple[Path, int]:
    rows = easyanimate_camera_rows(trajectory, fps=fps)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # First row is skipped by easyanimate.data.dataset_image_video.process_pose_file.
    np.savetxt(out, np.concatenate([rows[:1], rows], axis=0), fmt="%.9f")
    return out, int(rows.shape[0])


@register("easyanimate-v51-camera")
class EasyAnimateAdapter:
    name = "easyanimate_v51_camera"

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
        runtime_parameters = require_mapping(execution, "runtime_parameters")
        benchmark_profile = require_mapping(execution, "wrbench_benchmark_profile")
        official_profile = require_mapping(execution, "official_inference_profile")
        benchmark_frames = require_int(benchmark_profile, "video_length")
        benchmark_fps = require_int(benchmark_profile, "fps")
        entrypoint = require_str(execution, "entrypoint")
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        if int(target.frame_count) != benchmark_frames:
            raise ValueError(f"{key} requires contract video_length {benchmark_frames}, got {target.frame_count}")
        if int(target.fps) != benchmark_fps:
            raise ValueError(f"{key} requires contract fps {benchmark_fps}, got {target.fps}")

        sample_size = list(benchmark_profile.get("sample_size") or [])
        if len(sample_size) != 2:
            raise ValueError(f"{key} wrbench_benchmark_profile.sample_size must be [height, width]")
        sample_height, sample_width = [int(value) for value in sample_size]
        if sample_height != int(height) or sample_width != int(width):
            raise ValueError(
                f"{key} requires sample_size [height,width]=[{sample_height},{sample_width}], "
                f"got height={height} width={width}"
            )
        stride = _official_pose_stride(benchmark_fps)
        camera_txt, pose_row_count = write_easyanimate_camera_txt(
            out_dir / "easyanimate_control_camera.txt",
            target,
            fps=benchmark_fps,
        )
        command_template = build_command_template(execution, values={})
        request_json = write_json(
            out_dir / "easyanimate_v51_camera_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "control_camera_txt": str(camera_txt),
                    "control_txt_total_rows": pose_row_count + 1,
                    "control_pose_rows": pose_row_count,
                    "official_pose_fps": _OFFICIAL_CAMERACTRL_POSE_FPS,
                    "official_fps_downsample_stride": stride,
                    "effective_control_frames_after_downsample": int(target.frame_count),
                    "sample_size": [sample_height, sample_width],
                    "sample_size_order": "height_width",
                },
                "runtime_parameters": runtime_parameters,
                "official_runtime": require_mapping(execution, "official_runtime"),
                "official_script_defaults": require_mapping(execution, "official_script_defaults"),
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
                "wrbench_policy": require_mapping(execution, "wrbench_policy"),
            },
        )

        payload_type = "easyanimate_camera_txt"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "control_camera_txt": str(camera_txt),
                "request_json": str(request_json),
                "sample_size": [sample_height, sample_width],
                "video_length": int(target.frame_count),
                "fps": int(target.fps),
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="control_camera_txt -> process_pose_file(...)[:: int(24 // fps)]",
            coordinate_notes=(
                "EasyAnimate V5.1 consumes CameraCtrl 19-column W2C rows with normalized official intrinsics. "
                "WRBench writes a 24 fps dense pose stream so the official fps decimation yields one control per video frame."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="direct_frame_pose_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "control_camera_txt": str(camera_txt),
                    "control_txt_total_rows": pose_row_count + 1,
                    "control_pose_rows": pose_row_count,
                    "effective_control_frames_after_downsample": int(target.frame_count),
                    "official_pose_fps": _OFFICIAL_CAMERACTRL_POSE_FPS,
                    "official_fps_downsample_stride": stride,
                    "official_intrinsics_normalized": list(_OFFICIAL_CAMERACTRL_INTRINSICS),
                    "sample_size": [sample_height, sample_width],
                    "sample_size_order": "height_width",
                    "video_length": int(target.frame_count),
                    "fps": int(target.fps),
                    "num_inference_steps": require_int(official_profile, "num_inference_steps"),
                },
                control_sample_kind="dense_pose",
                control_sample_count=int(target.frame_count),
                sampling_rule="24fps_control_camera_txt_rows_decimated_by_official_fps_rule",
                model_control_extra={
                    "control_txt_total_rows": pose_row_count + 1,
                    "control_pose_rows": pose_row_count,
                    "official_pose_fps": _OFFICIAL_CAMERACTRL_POSE_FPS,
                    "official_fps_downsample_stride": stride,
                    "effective_control_frames_after_downsample": int(target.frame_count),
                    "sample_size": [sample_height, sample_width],
                },
            ),
        )
