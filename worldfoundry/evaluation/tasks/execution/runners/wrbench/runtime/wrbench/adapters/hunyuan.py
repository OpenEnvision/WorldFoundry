"""HunyuanVideo camera adapter: GameCraft (input_pose txt) and WorldPlay (pose dict)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


_GAMECRAFT_INPUT_POSE_HEADER = "# gamecraft_camera_ctrl_v1"
_WORLDPLAY_OFFICIAL_FX_NORM = 969.6969696969696 / (960.0 * 2.0)
_WORLDPLAY_OFFICIAL_FY_NORM = 969.6969696969696 / (540.0 * 2.0)
_GAMECRAFT_OFFICIAL_FX_NORM = 0.50505
_GAMECRAFT_OFFICIAL_FY_NORM = 0.8979


def _relative_yaw_peak_deg(c2w: np.ndarray) -> float:
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    return float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0


def _with_official_normalized_intrinsics(
    trajectory: CameraTrajectory,
    *,
    fx_norm: float,
    fy_norm: float,
) -> CameraTrajectory:
    k = np.asarray(
        [
            [float(fx_norm), 0.0, 0.5],
            [0.0, float(fy_norm), 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.repeat(k[None], trajectory.frame_count, axis=0)
    return CameraTrajectory(
        c2w=trajectory.to_c2w(),
        intrinsics=intrinsics,
        camera_type=trajectory.camera_type,
        fps=trajectory.fps,
        source=trajectory.source,
        conversion_mode=trajectory.conversion_mode,
        coordinate_convention=trajectory.coordinate_convention,
    )


def _write_gamecraft_input_pose_txt(path: str | Path, trajectory: CameraTrajectory) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [_GAMECRAFT_INPUT_POSE_HEADER]
    for i, (w2c, intr) in enumerate(zip(trajectory.to_w2c(), trajectory.intrinsics)):
        row = [
            float(i),
            float(intr[0, 0]),
            float(intr[1, 1]),
            float(intr[0, 2]),
            float(intr[1, 2]),
            0.0,
            0.0,
            *[float(x) for x in w2c[:3, :4].reshape(-1)],
        ]
        rows.append(" ".join(f"{x:.8f}" for x in row))
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def _gamecraft_model_input_trajectory(target: CameraTrajectory) -> tuple[CameraTrajectory, bool]:
    if abs(_relative_yaw_peak_deg(target.to_c2w())) <= 1e-6:
        return target, False

    c2w = target.to_c2w()
    base = c2w[0]
    base_inv = np.linalg.inv(base)
    out = np.empty_like(c2w)
    for idx, pose in enumerate(c2w):
        rel = base_inv @ pose
        out[idx] = base @ np.linalg.inv(rel)
    return (
        CameraTrajectory(
            c2w=out.astype(np.float32),
            intrinsics=target.intrinsics,
            camera_type=target.camera_type,
            fps=target.fps,
            source=target.source,
            conversion_mode=f"{target.conversion_mode}_gamecraft_input_pose_yaw_sign_flipped",
            coordinate_convention=target.coordinate_convention,
        ),
        True,
    )


@register("hunyuan-game-craft", "hunyuan-worldplay")
class HunyuanAdapter:
    name = "hunyuan"

    def compile(self, trajectory: CameraTrajectory, *, model_name: str, width: int, height: int, num_frames: int, work_dir: str | Path | None = None, device: str | None = None) -> CameraPayload:
        key = canonical_model_key(model_name)
        if key == "hunyuan-game-craft":
            target, amp = model_target_trajectory(trajectory, key, num_frames)
            target = _with_official_normalized_intrinsics(
                target,
                fx_norm=_GAMECRAFT_OFFICIAL_FX_NORM,
                fy_norm=_GAMECRAFT_OFFICIAL_FY_NORM,
            )
            model_input, yaw_sign_flipped = _gamecraft_model_input_trajectory(target)
            pose_txt = _write_gamecraft_input_pose_txt(Path(work_dir or ".") / "hunyuan_game_craft_input_pose.txt", model_input)
            target_yaw_peak = _relative_yaw_peak_deg(target.to_c2w())
            input_yaw_peak = _relative_yaw_peak_deg(model_input.to_c2w())
            payload_type = "hunyuan_gamecraft_input_pose_txt"
            return CameraPayload(
                payload_type=payload_type,
                payload={"input_pose": str(pose_txt)},
                target_trajectory=target,
                official_camera_entrypoint="input_pose",
                coordinate_notes=(
                    "OpenCV C2W converted to W2C CameraCtrl rows for GetPoseEmbedsFromTxt "
                    "with official normalized GameCraft intrinsics; yaw payload rows are "
                    "sign-calibrated to match GameCraft's observed input_pose response"
                ),
                calibration_status=amp.calibration_status,
                metadata=adapter_taxonomy_metadata(
                    model_name=key,
                    amp=amp,
                    target=target,
                    requested_frames=int(num_frames),
                    payload_type=payload_type,
                    certification_kind="exact_model_action_payload",
                    target_c2w_is_model_effective=not yaw_sign_flipped,
                    model_payload_summary={
                        "pose_row_count": target.frame_count,
                        "gamecraft_input_pose_yaw_sign_flipped": yaw_sign_flipped,
                        "target_yaw_peak_signed_deg": target_yaw_peak,
                        "input_pose_yaw_peak_signed_deg": input_yaw_peak,
                        "official_normalized_intrinsics": [
                            _GAMECRAFT_OFFICIAL_FX_NORM,
                            _GAMECRAFT_OFFICIAL_FY_NORM,
                            0.5,
                            0.5,
                        ],
                    },
                ),
            )
        pose_frames = (int(num_frames) - 1) // 4 + 1
        target, amp = model_target_trajectory(trajectory, key, pose_frames)
        target = _with_official_normalized_intrinsics(
            target,
            fx_norm=_WORLDPLAY_OFFICIAL_FX_NORM,
            fy_norm=_WORLDPLAY_OFFICIAL_FY_NORM,
        )
        payload_type = "hunyuan_worldplay_pose_dict"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "poses": {
                    str(i): {"extrinsic": c.tolist(), "K": k.tolist()}
                    for i, (c, k) in enumerate(zip(target.to_c2w(), target.intrinsics))
                }
            },
            target_trajectory=target,
            official_camera_entrypoint="pose dict/json",
            coordinate_notes=(
                "OpenCV C2W extrinsics plus official normalized K in WorldPlay "
                "frame-indexed pose dict format"
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                model_payload_summary={
                    "pose_count": target.frame_count,
                    "official_normalized_intrinsics": [
                        _WORLDPLAY_OFFICIAL_FX_NORM,
                        _WORLDPLAY_OFFICIAL_FY_NORM,
                        0.5,
                        0.5,
                    ],
                },
                control_sample_kind="latent_pose",
                sampling_rule="video_frames_to_worldplay_pose_frames_stride4",
                model_control_extra={"video_frame_formula": "(num_frames - 1) // 4 + 1"},
            ),
        )
