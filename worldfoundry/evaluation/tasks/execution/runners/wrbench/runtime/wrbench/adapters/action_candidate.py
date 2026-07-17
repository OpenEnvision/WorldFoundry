"""Contract-driven action-condition camera adapters (Matrix-Game, Voyager, HY-World)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory, write_json
from wrbench.adapters.base import register
from wrbench.contracts import (
    build_command_template,
    require_execution_contract,
    require_float,
    require_int,
    require_mapping,
    require_str,
)
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


_CAMERA_KEYS = {"static", "yaw_LR", "yaw_RL", "pan_LR", "pan_RL"}


def _profile_from_trajectory(target: CameraTrajectory) -> str:
    camera_type = str(target.camera_type or "")
    if camera_type in _CAMERA_KEYS:
        return camera_type

    c2w = target.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    yaw_peak = float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0
    trans = rel[:, :3, 3]
    trans_peak = 0.0
    if trans.size:
        norms = np.linalg.norm(trans, axis=1)
        peak_vec = trans[int(np.argmax(norms))]
        trans_peak = float(peak_vec[int(np.argmax(np.abs(peak_vec)))])

    if abs(yaw_peak) > max(abs(trans_peak), 1e-4):
        return "yaw_LR" if yaw_peak < 0 else "yaw_RL"
    if abs(trans_peak) > 1e-4:
        return "pan_LR" if trans_peak < 0 else "pan_RL"
    return "static"


def _split_counts(frame_count: int) -> tuple[int, int]:
    first = int(frame_count) // 2
    return first, int(frame_count) - first


def _go_return_values(frame_count: int, *, sign: float, peak: float) -> np.ndarray:
    first, second = _split_counts(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if first == 0:
        return np.zeros((frame_count,), dtype=np.float32)
    up = np.linspace(0.0, float(sign) * float(peak), first, dtype=np.float32)
    down = np.linspace(float(sign) * float(peak), 0.0, second, dtype=np.float32)
    return np.concatenate([up, down], axis=0)


def _profile_segments(profile: str, frame_count: int) -> list[dict[str, object]]:
    first, second = _split_counts(frame_count)
    if profile == "static":
        return [{"action": "static", "frames": int(frame_count)}]
    if profile == "yaw_LR":
        return [{"action": "turn_left", "frames": first}, {"action": "turn_right", "frames": second}]
    if profile == "yaw_RL":
        return [{"action": "turn_right", "frames": first}, {"action": "turn_left", "frames": second}]
    if profile == "pan_LR":
        return [{"action": "left", "frames": first}, {"action": "right", "frames": second}]
    if profile == "pan_RL":
        return [{"action": "right", "frames": first}, {"action": "left", "frames": second}]
    raise ValueError(f"Unsupported camera profile: {profile}")


def _matrix_game3_action_values(profile: str, *, cam_value: float) -> tuple[list[float], list[float], str]:
    keyboard = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mouse = [0.0, 0.0]
    if profile == "static":
        return keyboard, mouse, "static"
    if profile == "yaw_LR":
        mouse[1] = -float(cam_value)
        return keyboard, mouse, "turn_left"
    if profile == "yaw_RL":
        mouse[1] = float(cam_value)
        return keyboard, mouse, "turn_right"
    if profile == "pan_LR":
        keyboard[2] = 1.0
        return keyboard, mouse, "left"
    if profile == "pan_RL":
        keyboard[3] = 1.0
        return keyboard, mouse, "right"
    raise ValueError(f"Unsupported Matrix-Game-3 segment profile: {profile}")


def _matrix_game3_return_profile(profile: str) -> str:
    if profile == "yaw_LR":
        return "yaw_RL"
    if profile == "yaw_RL":
        return "yaw_LR"
    if profile == "pan_LR":
        return "pan_RL"
    if profile == "pan_RL":
        return "pan_LR"
    return profile


def _matrix_game3_segment_frames(num_iterations: int) -> list[int]:
    if int(num_iterations) <= 0:
        raise ValueError("Matrix-Game-3 num_iterations must be positive")
    return [57, *([40] * (int(num_iterations) - 1))]


def _matrix_game3_clip_segments(profile: str, *, num_iterations: int, cam_value: float) -> list[dict[str, object]]:
    segment_frames = _matrix_game3_segment_frames(num_iterations)
    total = sum(segment_frames)
    midpoint = total / 2.0
    segments: list[dict[str, object]] = []
    start = 0
    for index, frames in enumerate(segment_frames):
        center = start + frames / 2.0
        segment_profile = profile if center < midpoint else _matrix_game3_return_profile(profile)
        keyboard, mouse, action = _matrix_game3_action_values(segment_profile, cam_value=cam_value)
        segments.append(
            {
                "segment_index": index,
                "action": action,
                "frames": int(frames),
                "keyboard": keyboard,
                "mouse": mouse,
            }
        )
        start += frames
    return segments


def _expanded_actions(segments: list[dict[str, object]]) -> list[str]:
    out: list[str] = []
    for segment in segments:
        out.extend([str(segment["action"])] * int(segment["frames"]))
    return out


def _contract_profiles(key: str) -> tuple[dict, dict, dict, str]:
    execution = require_execution_contract(key)
    benchmark_profile = require_mapping(execution, "wrbench_benchmark_profile")
    official_profile = require_mapping(execution, "official_inference_profile")
    runtime_parameters = require_mapping(execution, "runtime_parameters")
    entrypoint = require_str(execution, "entrypoint")
    return benchmark_profile, official_profile, runtime_parameters, entrypoint


def _validate_profile(target: CameraTrajectory, *, key: str, width: int, height: int, benchmark_profile: dict) -> None:
    frames = require_int(benchmark_profile, "video_length")
    fps = require_int(benchmark_profile, "fps")
    if int(target.frame_count) != frames:
        raise ValueError(f"{key} requires contract video_length {frames}, got {target.frame_count}")
    if int(target.fps) != fps:
        raise ValueError(f"{key} requires contract fps {fps}, got {target.fps}")
    dimensions = benchmark_profile.get("dimensions")
    if isinstance(dimensions, list) and len(dimensions) == 2:
        expected_h, expected_w = int(dimensions[0]), int(dimensions[1])
        if (int(height), int(width)) != (expected_h, expected_w):
            raise ValueError(
                f"{key} requires dimensions [height,width]=[{expected_h},{expected_w}], "
                f"got height={height} width={width}"
            )


def _manual_yaml_row(path: Path, *, image: str, text: str, track: str) -> Path:
    def q(value: str) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"- image: {q(image)}",
                f"  text: {q(text)}",
                f"  track: {q(track)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@register("matrix-game-3")
class MatrixGame3Adapter:
    name = "matrix_game_3"

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
        benchmark_profile, official_profile, runtime_parameters, entrypoint = _contract_profiles(key)
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        _validate_profile(target, key=key, width=width, height=height, benchmark_profile=benchmark_profile)

        profile = _profile_from_trajectory(target)
        cam_value = require_float(runtime_parameters, "camera_value")
        num_iterations = require_int(official_profile, "num_iterations")
        segments = _matrix_game3_clip_segments(profile, num_iterations=num_iterations, cam_value=cam_value)
        if sum(int(segment["frames"]) for segment in segments) != target.frame_count:
            raise ValueError(f"{key} segment frame total does not match target frame count")

        keyboard = np.concatenate(
            [
                np.repeat(np.asarray(segment["keyboard"], dtype=np.float32)[None, :], int(segment["frames"]), axis=0)
                for segment in segments
            ],
            axis=0,
        )
        mouse = np.concatenate(
            [
                np.repeat(np.asarray(segment["mouse"], dtype=np.float32)[None, :], int(segment["frames"]), axis=0)
                for segment in segments
            ],
            axis=0,
        )
        keyboard_npy = out_dir / "matrix_game3_keyboard_condition.npy"
        mouse_npy = out_dir / "matrix_game3_mouse_condition.npy"
        np.save(keyboard_npy, keyboard)
        np.save(mouse_npy, mouse)
        sequence_json = write_json(
            out_dir / "matrix_game3_action_sequence.json",
            {
                "schema_version": 1,
                "profile": profile,
                "num_iterations": num_iterations,
                "clip_segment_frames": _matrix_game3_segment_frames(num_iterations),
                "segments": segments,
                "keyboard_shape": list(keyboard.shape),
                "mouse_shape": list(mouse.shape),
                "official_action_interface": "interactive get_current_action() returns one action per generated clip segment",
            },
        )
        command_template = build_command_template(
            require_execution_contract(key),
            values={
                "image": "<first_frame_image>",
                "output_dir": "<output_dir>",
                "prompt": "<prompt>",
                "save_name": "<save_name>",
            },
        )
        request_json = write_json(
            out_dir / "matrix_game3_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "action_sequence_json": str(sequence_json),
                    "keyboard_condition_npy": str(keyboard_npy),
                    "mouse_condition_npy": str(mouse_npy),
                    "segments": segments,
                    "keyboard_shape": list(keyboard.shape),
                    "mouse_shape": list(mouse.shape),
                    "segment_control_note": "Matrix-Game-3 official interactive code samples one action per 57/40-frame clip segment.",
                },
                "runtime_parameters": runtime_parameters,
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
            },
        )
        payload_type = "matrix_game3_action_conditions"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "action_sequence_json": str(sequence_json),
                "keyboard_condition_npy": str(keyboard_npy),
                "mouse_condition_npy": str(mouse_npy),
                "request_json": str(request_json),
                "segments": segments,
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="interactive get_current_action() keyboard/mouse tensors",
            coordinate_notes=(
                "Matrix-Game-3 consumes keyboard and mouse action tensors through its official interactive path. "
                "WRBench injects one action per official 57/40-frame rollout segment; target C2W is desired motion."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_action_condition_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "interactive_segments": len(segments),
                    "segment_frames": [int(segment["frames"]) for segment in segments],
                    "keyboard_shape": list(keyboard.shape),
                    "mouse_shape": list(mouse.shape),
                    "camera_value": cam_value,
                    "num_inference_steps": require_int(official_profile, "num_inference_steps"),
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=len(segments),
                sampling_rule="one_matrix_game3_keyboard_mouse_action_per_official_rollout_segment",
                model_control_extra={
                    "action_sequence_json": str(sequence_json),
                    "keyboard_condition_npy": str(keyboard_npy),
                    "mouse_condition_npy": str(mouse_npy),
                    "target_c2w_is_desired_wrbench_motion": True,
                    "control_granularity": "57 frames for first segment, then 40 frames per subsequent segment",
                },
            ),
        )


@register("matrix-game-2")
class MatrixGame2Adapter:
    name = "matrix_game_2"

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
        benchmark_profile, official_profile, runtime_parameters, entrypoint = _contract_profiles(key)
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        _validate_profile(target, key=key, width=width, height=height, benchmark_profile=benchmark_profile)

        profile = _profile_from_trajectory(target)
        keyboard = np.zeros((target.frame_count, 4), dtype=np.float32)
        mouse = np.zeros((target.frame_count, 2), dtype=np.float32)
        cam_value = require_float(runtime_parameters, "camera_value")
        mode = require_str(runtime_parameters, "mode")
        if profile in {"yaw_LR", "yaw_RL"}:
            sign = -1.0 if profile == "yaw_LR" else 1.0
            mouse[:, 1] = _go_return_values(target.frame_count, sign=sign, peak=cam_value)
        elif profile in {"pan_LR", "pan_RL"}:
            first, second = _split_counts(target.frame_count)
            first_col, second_col = (2, 3) if profile == "pan_LR" else (3, 2)
            keyboard[:first, first_col] = 1.0
            keyboard[first : first + second, second_col] = 1.0

        keyboard_npy = out_dir / "matrix_game2_keyboard_condition.npy"
        mouse_npy = out_dir / "matrix_game2_mouse_condition.npy"
        np.save(keyboard_npy, keyboard)
        np.save(mouse_npy, mouse)
        segments = _profile_segments(profile, target.frame_count)
        command_template = build_command_template(
            require_execution_contract(key),
            values={
                "img_path": "<first_frame_image>",
                "output_folder": "<output_dir>",
                "num_output_frames": require_int(official_profile, "num_output_frames"),
            },
        )
        request_json = write_json(
            out_dir / "matrix_game2_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "keyboard_condition_npy": str(keyboard_npy),
                    "mouse_condition_npy": str(mouse_npy),
                    "mode": mode,
                    "keyboard_shape": list(keyboard.shape),
                    "mouse_shape": list(mouse.shape),
                    "segments": segments,
                },
                "runtime_parameters": runtime_parameters,
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
            },
        )
        payload_type = "matrix_game2_action_conditions"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "keyboard_condition_npy": str(keyboard_npy),
                "mouse_condition_npy": str(mouse_npy),
                "request_json": str(request_json),
                "segments": segments,
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="conditional_dict['keyboard_cond']/['mouse_cond']",
            coordinate_notes=(
                "Matrix-Game-2 consumes per-frame keyboard and mouse action tensors. "
                "WRBench maps yaw to mouse x-axis camera actions and pan to left/right keyboard actions."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_action_condition_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "mode": mode,
                    "keyboard_shape": list(keyboard.shape),
                    "mouse_shape": list(mouse.shape),
                    "segments": segments,
                    "camera_value": cam_value,
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=int(target.frame_count),
                sampling_rule="wrbench_camera_family_to_matrix_game2_per_frame_action_conditions",
                model_control_extra={
                    "keyboard_condition_npy": str(keyboard_npy),
                    "mouse_condition_npy": str(mouse_npy),
                    "target_c2w_is_desired_wrbench_motion": True,
                },
            ),
        )


class ATIWan21Adapter:
    """Deferred: not registered in WRBench."""

    name = "ati_wan21_14b"

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
        benchmark_profile, official_profile, runtime_parameters, entrypoint = _contract_profiles(key)
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        _validate_profile(target, key=key, width=width, height=height, benchmark_profile=benchmark_profile)

        raw_track_frames = require_int(runtime_parameters, "raw_track_frames")
        point_grid = require_int(runtime_parameters, "point_grid")
        profile = _profile_from_trajectory(target)
        xs = np.linspace(width * 0.25, width * 0.75, point_grid, dtype=np.float32)
        ys = np.linspace(height * 0.25, height * 0.75, point_grid, dtype=np.float32)
        base = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        peak = require_float(runtime_parameters, "track_peak_fraction") * float(width)
        sign = -1.0 if profile in {"yaw_LR", "pan_LR"} else 1.0
        offsets = np.zeros((raw_track_frames,), dtype=np.float32)
        if profile != "static":
            offsets = _go_return_values(raw_track_frames, sign=sign, peak=peak)

        tracks = np.zeros((raw_track_frames, base.shape[0], 1, 3), dtype=np.float32)
        tracks[..., 0] = (base[None, :, None, 0] + offsets[:, None, None]) * 8.0
        tracks[..., 1] = base[None, :, None, 1] * 8.0
        tracks[..., 2] = 8.0
        npz_bytes = io.BytesIO()
        np.savez_compressed(npz_bytes, array=tracks)
        track_npz = out_dir / "ati_track_array.npz"
        track_npz.write_bytes(npz_bytes.getvalue())
        official_track_pth = out_dir / "ati_track_official.pth"
        prompt_yaml = _manual_yaml_row(
            out_dir / "ati_prompt.yaml",
            image="<first_frame_image>",
            text="<prompt>",
            track=str(official_track_pth),
        )
        command_template = build_command_template(
            require_execution_contract(key),
            values={
                "prompt_yaml": str(prompt_yaml),
                "output_dir": "<output_dir>",
                "save_file": "<output_dir>/outputs/%03d.mp4",
                "frame_num": int(target.frame_count),
                "size": f"{int(width)}*{int(height)}",
            },
        )
        request_json = write_json(
            out_dir / "ati_wan21_14b_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "track_array_npz": str(track_npz),
                    "official_torch_track_path": str(official_track_pth),
                    "prompt_yaml": str(prompt_yaml),
                    "track_shape": list(tracks.shape),
                    "track_format": "np.savez_compressed bytes under key=array; remote materializer torch.save(bytes)",
                },
                "runtime_parameters": runtime_parameters,
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
            },
        )
        payload_type = "ati_wan21_point_track"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "track_array_npz": str(track_npz),
                "official_track_pth": str(official_track_pth),
                "prompt_yaml": str(prompt_yaml),
                "request_json": str(request_json),
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="generate.py --track point trajectory pth",
            coordinate_notes=(
                "ATI consumes visible point tracks, not C2W matrices. WRBench maps camera action families "
                "to dense synthetic point-track motion; target C2W remains the desired benchmark motion."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_point_track_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "track_array_shape": list(tracks.shape),
                    "raw_track_frames": raw_track_frames,
                    "processed_model_frames": int(target.frame_count),
                    "sample_steps": require_int(official_profile, "sample_steps"),
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=raw_track_frames,
                sampling_rule="121_track_rows_downsampled_by_ATI_process_tracks_to_81_model_frames",
                model_control_extra={
                    "track_array_npz": str(track_npz),
                    "official_track_pth": str(official_track_pth),
                    "target_c2w_is_desired_wrbench_motion": True,
                },
            ),
        )


@register("hunyuanworld-voyager")
class HunyuanWorldVoyagerAdapter:
    name = "hunyuanworld_voyager"

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
        benchmark_profile, official_profile, runtime_parameters, entrypoint = _contract_profiles(key)
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        _validate_profile(target, key=key, width=width, height=height, benchmark_profile=benchmark_profile)
        profile = _profile_from_trajectory(target)
        if profile == "static":
            raise ValueError("HunyuanWorld-Voyager official data_engine has no static camera action")
        segments = _profile_segments(profile, target.frame_count)
        action_types = _expanded_actions(segments)
        voyager_types = [
            "turn_left" if item == "turn_left" else "turn_right" if item == "turn_right" else item
            for item in action_types
        ]
        sequence_json = write_json(
            out_dir / "voyager_camera_sequence.json",
            {
                "schema_version": 1,
                "action_types": voyager_types,
                "segments": segments,
                "official_type_choices": ["forward", "backward", "left", "right", "turn_left", "turn_right"],
            },
        )
        wrbench_camera_json = target.write_json(out_dir / "voyager_wrbench_camera.json")
        command_template = build_command_template(
            require_execution_contract(key),
            values={
                "input_path": "<voyager_input_path>",
                "save_path": "<output_dir>",
                "prompt": "<prompt>",
            },
        )
        request_json = write_json(
            out_dir / "voyager_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "camera_sequence_json": str(sequence_json),
                    "wrbench_camera_json": str(wrbench_camera_json),
                    "first_frame_image": "<first_frame_image>",
                    "data_engine_render_dir": "<voyager_input_path>",
                    "segments": segments,
                    "action_types": voyager_types,
                    "materialization": "WRBench target C2W rendered through Voyager data_engine functions into official partial_cond files",
                },
                "runtime_parameters": runtime_parameters,
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
            },
        )
        payload_type = "voyager_wrbench_rendered_camera_condition"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "camera_sequence_json": str(sequence_json),
                "wrbench_camera_json": str(wrbench_camera_json),
                "request_json": str(request_json),
                "segments": segments,
                "action_types": voyager_types,
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint=(
                "WRBench trajectory -> Voyager data_engine render_from_cameras_videos/create_video_input "
                "-> sample_image2video.py partial_cond"
            ),
            coordinate_notes=(
                "Voyager diffusion consumes rendered RGB/depth/mask partial conditions, not C2W tensors directly. "
                "WRBench renders those official partial-condition files from the target C2W sequence and keeps "
                "the nearest official action labels as audit metadata."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="rendered_geometric_condition_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "segments": segments,
                    "action_type_count": len(voyager_types),
                    "wrbench_camera_json": str(wrbench_camera_json),
                    "video_length": require_int(official_profile, "video_length"),
                },
                target_c2w_is_model_effective=True,
                control_sample_kind="rendered_rgbd_mask_from_dense_pose",
                control_sample_count=int(target.frame_count),
                sampling_rule="wrbench_target_c2w_rendered_to_official_voyager_partial_condition_frames",
                model_control_extra={
                    "camera_sequence_json": str(sequence_json),
                    "wrbench_camera_json": str(wrbench_camera_json),
                    "official_action_labels_are_audit_metadata": True,
                    "model_conditioning_files": ["render_XXXX.png", "depth_XXXX.exr", "mask_XXXX.png"],
                    "target_c2w_effective_via_rendered_partial_conditions": True,
                },
            ),
        )


@register("hyworld-worldgen")
class HYWorldWorldGenAdapter:
    name = "hyworld_worldgen"

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
        benchmark_profile, official_profile, runtime_parameters, entrypoint = _contract_profiles(key)
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        _validate_profile(target, key=key, width=width, height=height, benchmark_profile=benchmark_profile)
        camera_json = write_json(
            out_dir / "hyworld_camera.json",
            {
                "intrinsic": target.intrinsics.astype(float).tolist(),
                "extrinsic": target.to_w2c().astype(float).tolist(),
            },
        )
        command_template = build_command_template(
            require_execution_contract(key),
            values={"target_path": "<hyworld_scene_path>"},
        )
        request_json = write_json(
            out_dir / "hyworld_worldgen_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "input_contract": {
                    "camera_json": str(camera_json),
                    "scene_path": "<hyworld_scene_path>",
                    "scene_materialization": "HYWORLD_SCENE_PATH or shared HYWORLD_SCENE_ROOT scene; optional HYWORLD_PANORAMA_PATH; optional HYWORLD_SCENE_TYPE for non-VLM scene type; optional HYWORLD_PANOGEN_* official CLI overrides",
                    "wrbench_traj": "render_results/view0/traj_wrbench/camera.json",
                    "render_output": "render.mp4 under the selected trajectory directory",
                },
                "runtime_parameters": runtime_parameters,
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
            },
        )
        payload_type = "hyworld_worldgen_camera_json"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "camera_json": str(camera_json),
                "request_json": str(request_json),
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="hyworld2/worldgen/traj_render.py camera.json intrinsic/extrinsic",
            coordinate_notes="HY-World worldgen renders from per-frame W2C extrinsics and intrinsics stored in camera.json.",
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
                    "camera_json": str(camera_json),
                    "pose_count": int(target.frame_count),
                    "fps": require_int(official_profile, "fps"),
                },
                control_sample_kind="dense_pose",
                control_sample_count=int(target.frame_count),
                sampling_rule="one_camera_json_pose_per_hyworld_render_frame",
                model_control_extra={"camera_json": str(camera_json)},
            ),
        )
