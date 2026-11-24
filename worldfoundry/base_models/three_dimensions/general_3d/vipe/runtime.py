"""Inference-only WorldFoundry runtime for pinned NVIDIA ViPE."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from worldfoundry.base_models.three_dimensions.general_3d.vipe.assets import (
    prepare_pose_assets,
    required_pose_assets,
)
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.build import (
    load_native_extension,
    native_extension_status,
)


@dataclass(frozen=True, slots=True)
class PoseInferenceResult:
    """Output contract consumed by iWorld-Bench trajectory metrics."""

    input_video: str
    pose_path: str
    frame_count: int
    first_frame_index: int
    frame_skip: int
    runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def preflight(*, build_native: bool = False, require_assets: bool = True, verbose: bool = False) -> dict[str, object]:
    """Return native, checkpoint, and CUDA readiness without a mock fallback."""
    native = native_extension_status(build_if_missing=build_native, verbose=verbose)
    assets = required_pose_assets()
    assets_ready = all(asset.available for asset in assets)
    reasons: list[str] = []
    if native.reason:
        reasons.append(native.reason)
    if require_assets and not assets_ready:
        missing = ", ".join(f"{asset.name} -> {asset.path}" for asset in assets if not asset.available)
        reasons.append(f"missing checkpoints: {missing}")
    return {
        "ready": native.ready and (assets_ready or not require_assets),
        "native": native.to_dict(),
        "assets": [asset.to_dict() for asset in assets],
        "reason": "; ".join(reasons) or None,
    }


def _validate_video(path: Path, *, frame_start: int, frame_end: int, frame_skip: int) -> int:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"ViPE input video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = capture.read()
    finally:
        capture.release()
    if not ok or total < 2 or width <= 0 or height <= 0:
        raise ValueError(f"ViPE requires a decodable video with at least two frames: {path}")
    stop = total if frame_end < 0 else min(frame_end, total)
    selected = len(range(frame_start, stop, frame_skip))
    if selected < 2:
        raise ValueError(
            f"Selected ViPE range contains {selected} frame(s); choose at least two "
            f"(total={total}, start={frame_start}, end={frame_end}, skip={frame_skip})."
        )
    return stop


def infer_poses(
    input_videos: Sequence[str | Path],
    output_dir: str | Path,
    *,
    frame_start: int = 0,
    frame_end: int = -1,
    frame_skip: int = 1,
    hydra_overrides: Sequence[str] = (),
) -> list[PoseInferenceResult]:
    """Estimate poses for multiple videos while reusing immutable model weights."""
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if isinstance(input_videos, (str, Path)):
        raise TypeError("input_videos must be a sequence of paths, not one path string")
    if not input_videos:
        raise ValueError("input_videos must contain at least one video")

    # Keep the caller-visible filename for the NPZ contract. Resolving a
    # benchmark symlink would replace its semantic sample id with the target's
    # unrelated filename.
    video_paths = [Path(path).expanduser().absolute() for path in input_videos]
    stems = [path.stem for path in video_paths]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise ValueError(f"ViPE output names would collide for duplicate video stems: {duplicate_stems}")

    output_root = Path(output_dir).expanduser().resolve()
    resolved_ends = [
        _validate_video(
            video_path,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
        )
        for video_path in video_paths
    ]
    prepare_pose_assets(download=False)
    load_native_extension(build_if_missing=False)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ViPE pose inference requires a visible CUDA GPU.")

    from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import parse_typed_config
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.pipeline import make_pipeline
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.slam.interface import SLAMOutput
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.base import ProcessedVideoStream
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.raw_mp4_stream import RawMp4Stream

    overrides = [
        "pipeline=default",
        f"streams.base_path={json.dumps(str(video_paths[0]))}",
        "pipeline.init.instance=null",
        "pipeline.slam.keyframe_depth=null",
        "pipeline.post.depth_align_model=null",
        f"pipeline.output.path={json.dumps(str(output_root))}",
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_viz=false",
        "pipeline.output.save_slam_map=false",
        "pipeline.slam.visualize=false",
        "pipeline.slam.ba.fused=false",
        *hydra_overrides,
    ]
    config = parse_typed_config("default", hydra_args=overrides)
    pipeline = make_pipeline(config.pipeline)
    pipeline.return_payload = True

    results: list[PoseInferenceResult] = []
    for video_path, resolved_end in zip(video_paths, resolved_ends, strict=True):
        stream = ProcessedVideoStream(
            RawMp4Stream(
                video_path,
                seek_range=range(frame_start, resolved_end, frame_skip),
            ),
            [],
        ).cache(desc=f"Reading ViPE input: {video_path.name}")

        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipeline.run(stream)
        torch.cuda.synchronize()
        runtime_seconds = time.perf_counter() - started
        if not isinstance(result.payload, SLAMOutput):
            raise RuntimeError(f"ViPE did not return an SLAMOutput payload: {type(result.payload).__name__}")

        poses = result.payload.get_view_trajectory(0).matrix().detach().cpu().numpy()
        del result
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
            raise RuntimeError(f"ViPE returned an invalid pose tensor shape: {poses.shape}")
        if not np.isfinite(poses).all():
            raise RuntimeError("ViPE returned non-finite camera poses.")
        rotations = poses[:, :3, :3]
        determinants = np.linalg.det(rotations)
        if not all(math.isfinite(float(value)) and abs(float(value) - 1.0) < 5e-2 for value in determinants):
            raise RuntimeError(f"ViPE returned invalid rotation determinants: {determinants.tolist()}")

        frame_indices = frame_start + np.arange(poses.shape[0], dtype=np.int64) * frame_skip
        pose_path = output_root / "pose" / f"{video_path.stem}.npz"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            pose_path,
            data=poses,
            inds=frame_indices,
            runtime=np.asarray(runtime_seconds, dtype=np.float64),
        )
        with np.load(pose_path) as saved:
            if saved["data"].shape != poses.shape or saved["inds"].shape != frame_indices.shape:
                raise RuntimeError(f"ViPE pose artifact failed round-trip validation: {pose_path}")

        results.append(
            PoseInferenceResult(
                input_video=str(video_path),
                pose_path=str(pose_path),
                frame_count=int(poses.shape[0]),
                first_frame_index=frame_start,
                frame_skip=frame_skip,
                runtime_seconds=runtime_seconds,
            )
        )
    return results


def infer_pose(
    input_video: str | Path,
    output_dir: str | Path,
    *,
    frame_start: int = 0,
    frame_end: int = -1,
    frame_skip: int = 1,
    hydra_overrides: Sequence[str] = (),
) -> PoseInferenceResult:
    """Convenience wrapper for one video; batch callers should use ``infer_poses``."""
    return infer_poses(
        [input_video],
        output_dir,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        hydra_overrides=hydra_overrides,
    )[0]


__all__ = ["PoseInferenceResult", "infer_pose", "infer_poses", "preflight"]
