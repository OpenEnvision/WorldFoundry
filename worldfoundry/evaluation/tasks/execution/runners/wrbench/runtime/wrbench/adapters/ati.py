"""ATI adapter: project a camera trajectory into official dense point tracks."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory
from wrbench.adapters.base import register
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


def _camera_tracks(trajectory: CameraTrajectory, width: int, height: int) -> np.ndarray:
    """Project a canonical fronto-parallel point grid through OpenCV C2W poses."""

    if trajectory.frame_count != 121:
        raise ValueError(f"ATI's official track codec requires 121 samples, got {trajectory.frame_count}.")
    xs = np.linspace(0.08 * width, 0.92 * width, 12, dtype=np.float64)
    ys = np.linspace(0.08 * height, 0.92 * height, 8, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    pixels = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)

    k0 = trajectory.intrinsics[0].astype(np.float64)
    depth = 2.0
    camera_points = np.stack(
        (
            (pixels[:, 0] - k0[0, 2]) / k0[0, 0] * depth,
            (pixels[:, 1] - k0[1, 2]) / k0[1, 1] * depth,
            np.full(len(pixels), depth, dtype=np.float64),
            np.ones(len(pixels), dtype=np.float64),
        ),
        axis=-1,
    )
    world_points = (trajectory.to_c2w()[0].astype(np.float64) @ camera_points.T).T
    tracks = np.zeros((len(pixels), trajectory.frame_count, 1, 3), dtype=np.float32)
    for frame_index, (c2w, intrinsics) in enumerate(zip(trajectory.to_c2w(), trajectory.intrinsics)):
        camera = (np.linalg.inv(c2w.astype(np.float64)) @ world_points.T).T[:, :3]
        z = camera[:, 2]
        safe_z = np.where(np.abs(z) > 1e-8, z, 1.0)
        u = intrinsics[0, 0] * camera[:, 0] / safe_z + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / safe_z + intrinsics[1, 2]
        visible = (z > 1e-6) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        tracks[:, frame_index, 0, 0] = np.where(visible, u, 0.0)
        tracks[:, frame_index, 0, 1] = np.where(visible, v, 0.0)
        tracks[:, frame_index, 0, 2] = visible.astype(np.float32)
    return tracks


def _write_official_track_file(path: Path, tracks: np.ndarray) -> Path:
    """Use ATI's official NPZ-bytes-inside-torch-save representation."""

    import torch

    buffer = io.BytesIO()
    np.savez_compressed(buffer, array=(tracks * 8.0).astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer.getvalue(), path)
    return path


@register("ati-wan21-14b")
class ATIAdapter:
    name = "ati"

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
        del device
        key = canonical_model_key(model_name)
        if int(num_frames) != 81:
            raise ValueError("ATI-Wan2.1 requires 81 generated frames.")
        target, amp = model_target_trajectory(trajectory, key, int(num_frames))
        track_trajectory = target.resample(121)
        tracks = _camera_tracks(track_trajectory, int(width), int(height))
        track_path = _write_official_track_file(ensure_work_dir(work_dir) / "ati_camera_tracks.pth", tracks)
        metadata = adapter_taxonomy_metadata(
            model_name=key,
            amp=amp,
            target=target,
            requested_frames=int(num_frames),
            payload_type="ati_dense_point_tracks",
            model_payload_summary={
                "track_count": int(tracks.shape[0]),
                "raw_track_samples": 121,
                "model_track_samples": 81,
            },
            control_sample_kind="dense_projected_point_track",
            control_sample_count=121,
            sampling_rule="121_camera_samples_projected_to_tracks_then_official_121_to_81_resampling",
            model_control_extra={"projection_depth": 2.0, "grid": [12, 8]},
        )
        return CameraPayload(
            payload_type="ati_dense_point_tracks",
            payload={
                "track_path": str(track_path),
                "track_width": int(width),
                "track_height": int(height),
            },
            target_trajectory=target,
            official_camera_entrypoint="WanATI.generate(..., tracks=process_tracks(...))",
            coordinate_notes=(
                "OpenCV C2W poses project a fixed canonical point grid into ATI's official "
                "[N,121,1,(x,y,visibility)] track codec."
            ),
            calibration_status=amp.calibration_status,
            metadata=metadata,
        )


__all__ = ["ATIAdapter"]
