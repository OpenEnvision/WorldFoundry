"""Run VGGT-Omega pose inference for one video and export OpenCV C2W poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def safe_scene_name(scene_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_name.strip())
    return safe.strip("._") or "scene"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _expand_extrinsics(extrinsics: Any) -> np.ndarray:
    arr = _as_numpy(extrinsics).astype(np.float64)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"expected one batch of VGGT-Omega extrinsics, got {arr.shape}")
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[-2:] == (3, 4):
        bottom = np.zeros((*arr.shape[:-2], 1, 4), dtype=arr.dtype)
        bottom[..., 0, 3] = 1.0
        arr = np.concatenate([arr, bottom], axis=-2)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        raise ValueError(f"expected extrinsics shape T x 3 x 4 or T x 4 x 4, got {arr.shape}")
    if len(arr) == 0:
        raise ValueError("extrinsics stack is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("extrinsics contain non-finite values")
    return arr


def validate_cam_c2w(poses: Any) -> np.ndarray:
    arr = _as_numpy(poses).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4) or len(arr) == 0:
        raise ValueError(f"expected non-empty T x 4 x 4 pose stack, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose stack contains non-finite values")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=arr.dtype)
    if not np.allclose(arr[:, 3, :], expected_bottom, atol=1e-5):
        raise ValueError("pose stack is not homogeneous 4x4 c2w")
    det = np.linalg.det(arr[:, :3, :3])
    if not np.all(np.isfinite(det)) or np.any(np.abs(det) <= 1e-8):
        raise ValueError("pose rotation block is singular")
    return arr


def vggt_extrinsics_to_c2w(extrinsics_camera_from_world: Any) -> np.ndarray:
    """Convert VGGT-Omega OpenCV camera-from-world extrinsics to OpenCV C2W."""

    w2c = _expand_extrinsics(extrinsics_camera_from_world)
    return validate_cam_c2w(np.linalg.inv(w2c))


def _squeeze_batch(value: Any) -> np.ndarray:
    arr = _as_numpy(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def write_pose_outputs(
    extrinsics_camera_from_world: Any,
    *,
    output_dir: str | Path,
    expected_frames: int | None = None,
    intrinsics: Any | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_w2c = _expand_extrinsics(extrinsics_camera_from_world).astype(np.float32)
    np.save(output / "extrinsics_opencv_w2c.npy", raw_w2c)
    poses = vggt_extrinsics_to_c2w(extrinsics_camera_from_world)
    if expected_frames is not None and len(poses) != int(expected_frames):
        raise RuntimeError(f"VGGT-Omega pose length mismatch: poses={len(poses)} expected_frames={expected_frames}")
    pose_path = output / "poses.npy"
    np.save(pose_path, poses)
    if intrinsics is not None:
        np.save(output / "intrinsics.npy", _squeeze_batch(intrinsics).astype(np.float32))
    return pose_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def extract_frames(video_path: Path, frames_dir: Path, *, max_frames: int | None = None) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required to extract video frames") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    paths: list[Path] = []
    try:
        while max_frames is None or len(paths) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frames_dir / f"{len(paths) + 1:06d}.png"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"failed to write frame: {out_path}")
            paths.append(out_path)
    finally:
        cap.release()
    if not paths:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return paths


def _load_state_dict(torch_module: Any, checkpoint_path: Path) -> dict[str, Any]:
    state = torch_module.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                return candidate
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint did not contain a state dict: {checkpoint_path}")
    return state


def run_pipeline(args: argparse.Namespace) -> Path:
    video_path = Path(args.video_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    vggt_repo = Path(args.vggt_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    safe_scene = safe_scene_name(args.scene_name)

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not vggt_repo.is_dir():
        raise FileNotFoundError(f"VGGT-Omega repo not found: {vggt_repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")

    if str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))

    import torch  # type: ignore
    from vggt_omega.models import VGGTOmega  # type: ignore
    from vggt_omega.utils.load_fn import load_and_preprocess_images  # type: ignore
    from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore

    frames_dir = output_dir / "frames" / safe_scene
    frame_paths = extract_frames(video_path, frames_dir, max_frames=args.max_frames)

    model = VGGTOmega().to(args.device).eval()
    load_ret = model.load_state_dict(_load_state_dict(torch, checkpoint), strict=True)
    images = load_and_preprocess_images(
        [str(path) for path in frame_paths],
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
    ).to(args.device)
    with torch.inference_mode():
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

    pose_path = write_pose_outputs(
        extrinsics,
        output_dir=output_dir,
        expected_frames=len(frame_paths),
        intrinsics=intrinsics,
    )
    summary = {
        "video_path": str(video_path),
        "scene_name": safe_scene,
        "checkpoint": str(checkpoint),
        "vggt_repo": str(vggt_repo),
        "frame_count": len(frame_paths),
        "image_resolution": int(args.image_resolution),
        "preprocess_mode": str(args.preprocess_mode),
        "pose_path": str(pose_path),
        "raw_extrinsics_path": str(output_dir / "extrinsics_opencv_w2c.npy"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "vggt_source_commit": git_commit(vggt_repo),
        "checkpoint_strict_load_missing_keys": list(getattr(load_ret, "missing_keys", [])),
        "checkpoint_strict_load_unexpected_keys": list(getattr(load_ret, "unexpected_keys", [])),
        "pose_convention": "opencv_c2w",
        "raw_extrinsics_convention": "opencv_w2c",
        "source_extrinsics_convention": "opencv_camera_from_world_w2c",
        "d1_cache_convention": "opencv_c2w",
    }
    (output_dir / "vggt_omega_pose_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pose_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega and export predicted OpenCV C2W poses.npy.")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vggt_repo", "--vggt-repo", dest="vggt_repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image-resolution", type=int, required=True)
    parser.add_argument("--preprocess-mode", choices=("balanced", "max_size"), required=True)
    parser.add_argument("--max_frames", "--max-frames", dest="max_frames", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pose_path = run_pipeline(args)
    print(f"Wrote VGGT-Omega poses: {pose_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
