# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process Lyra-2 video-to-Gaussian reconstruction.

Ported from ``lyra_2/_src/inference/vipe_da3_gs_recon.py`` of nv-tlabs/lyra
(87f79a52b81b366d1d4aa3a526aa12e54207c998). ViPE estimates per-frame poses,
Depth Anything 3 reconstructs 3D Gaussians conditioned on those poses, and the
Gaussians are rendered along the input trajectory. ViPE and DA3 are the
canonical WorldFoundry base models.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
_LOG_PREFIX = "[lyra2_recon]"
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Lyra2ReconstructionOutputs:
    """Artifacts written by :func:`reconstruct_video_gaussians`."""

    output_dir: Path
    reconstructed_scene_path: Path
    render_video_path: Path
    camera_path: Path
    vipe_prediction_path: Path | None


@dataclass(slots=True)
class _VipePoses:
    intrinsics: torch.Tensor
    extrinsics_c2w: torch.Tensor
    frame_ids: torch.Tensor


def _output_paths(output_dir: Path, *, no_vipe: bool) -> Lyra2ReconstructionOutputs:
    """Return the artifact paths of one reconstruction run.

    Args:
        output_dir: Resolved reconstruction output directory.
        no_vipe: Whether ViPE was skipped, in which case no ViPE prediction file exists.
    """
    return Lyra2ReconstructionOutputs(
        output_dir=output_dir,
        reconstructed_scene_path=output_dir / "reconstructed_scene.ply",
        render_video_path=output_dir / "gs_trajectory.mp4",
        camera_path=output_dir / "cameras.npz",
        vipe_prediction_path=None if no_vipe else output_dir / "vipe_predictions.npz",
    )


def _read_video_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    """Decode RGB frames from the start of a video.

    Args:
        video_path: Input video file.
        max_frames: Maximum number of frames to read; ``0`` reads the whole video.

    Returns:
        RGB uint8 frames and the container frame rate (30 when unknown).
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while max_frames <= 0 or len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames, fps if fps > 1e-6 else 30.0


def _uniform_subsample_indices(num_frames: int, max_frames: int) -> list[int]:
    """Pick at most ``max_frames`` indices spread uniformly over ``num_frames``.

    Args:
        num_frames: Number of available frames.
        max_frames: Maximum number of indices; ``0`` keeps every frame.
    """
    if num_frames <= 0:
        return []
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    return np.floor(np.linspace(0, num_frames - 1, num=max_frames)).astype(np.int64).tolist()


def _intrinsics_vec_to_k33(intrinsics_vec: torch.Tensor) -> torch.Tensor:
    """Convert ViPE ``(T, 4+D)`` ``[fx, fy, cx, cy, ...]`` intrinsics to ``(T, 3, 3)`` matrices.

    Args:
        intrinsics_vec: Per-frame ViPE intrinsics vectors.
    """
    import torch

    if intrinsics_vec.ndim != 2 or intrinsics_vec.shape[1] < 4:
        raise ValueError(f"Expected intrinsics shape (T,>=4), got {tuple(intrinsics_vec.shape)}")
    k = torch.zeros((intrinsics_vec.shape[0], 3, 3), dtype=intrinsics_vec.dtype, device=intrinsics_vec.device)
    k[:, 0, 0] = intrinsics_vec[:, 0]
    k[:, 1, 1] = intrinsics_vec[:, 1]
    k[:, 0, 2] = intrinsics_vec[:, 2]
    k[:, 1, 2] = intrinsics_vec[:, 3]
    k[:, 2, 2] = 1.0
    return k


def _pad_to_44(mat: np.ndarray) -> np.ndarray:
    """Pad ``(..., 3, 4)`` extrinsics to homogeneous ``(..., 4, 4)`` matrices.

    Args:
        mat: Extrinsics with ``3x4`` or ``4x4`` trailing dimensions.
    """
    if mat.shape[-2:] == (4, 4):
        return mat
    padded = np.zeros((*mat.shape[:-2], 4, 4), dtype=mat.dtype)
    padded[..., :3, :4] = mat[..., :3, :4]
    padded[..., 3, 3] = 1.0
    return padded


def _vipe_default_overrides(output_path: Path) -> list[str]:
    """Return the Lyra-2 ViPE Hydra overrides: pose-only, fixed intrinsics, no artifacts.

    Args:
        output_path: Scratch directory ViPE may write to.
    """
    return [
        "pipeline=default",
        "pipeline.slam.optimize_intrinsics=false",
        "pipeline.post.depth_align_model=null",
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_viz=false",
        f"pipeline.output.path={output_path}",
    ]


# Fast mode drops the instance segmentation stage and every on-disk ViPE output.
_VIPE_FAST_MODE_OVERRIDES = (
    "pipeline.init.instance=null",
    "pipeline.output.save_viz=false",
    "pipeline.output.save_artifacts=false",
    "pipeline.output.save_slam_map=false",
)


def _run_vipe(
    frames_rgb: Sequence[np.ndarray],
    *,
    fps: float,
    name: str,
    overrides: Sequence[str],
    fast_mode: bool,
    device: torch.device,
) -> _VipePoses:
    """Estimate per-frame camera poses and intrinsics with the canonical ViPE pipeline.

    Args:
        frames_rgb: RGB uint8 frames ``(H, W, 3)``.
        fps: Frame rate reported to ViPE.
        name: Stream name used by ViPE for logging and output naming.
        overrides: Hydra overrides applied to the ``default`` ViPE config.
        fast_mode: Skip instance segmentation and every ViPE artifact output.
        device: Device on which frames are handed to ViPE.
    """
    import torch

    from worldfoundry.base_models.three_dimensions.general_3d.vipe.assets import prepare_pose_assets
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.config import parse_typed_config
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.build import load_native_extension
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.pipeline import make_pipeline
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.streams.base import VideoFrame, VideoStream

    class InMemoryVideoStream(VideoStream):
        def __init__(self, frames: torch.Tensor) -> None:
            super().__init__()
            self.frames = frames

        def frame_size(self) -> tuple[int, int]:
            return int(self.frames.shape[1]), int(self.frames.shape[2])

        def name(self) -> str:
            return name

        def fps(self) -> float:
            return float(fps)

        def __len__(self) -> int:
            return int(self.frames.shape[0])

        def __iter__(self) -> Iterator[VideoFrame]:
            for idx in range(len(self)):
                rgb = self.frames[idx].to(device, non_blocking=True).float().div_(255.0)
                yield VideoFrame(raw_frame_idx=idx, rgb=rgb)

    prepare_pose_assets(download=False)
    load_native_extension(build_if_missing=False)
    hydra_args = [*overrides, *(_VIPE_FAST_MODE_OVERRIDES if fast_mode else ())]
    config = parse_typed_config("default", hydra_args=hydra_args)
    pipeline = make_pipeline(config.pipeline)
    pipeline.return_output_streams = True

    stream = InMemoryVideoStream(torch.from_numpy(np.stack(frames_rgb, axis=0)))
    result = pipeline.run(stream)
    if result.output_streams is None or len(result.output_streams) != 1:
        raise RuntimeError("ViPE did not return exactly one output stream.")

    intrinsics, c2w, frame_ids = [], [], []
    for frame in result.output_streams[0]:
        frame = frame.cpu()
        if frame.intrinsics is None or frame.pose is None:
            raise RuntimeError(f"ViPE returned no pose for frame {frame.raw_frame_idx}.")
        frame_ids.append(frame.raw_frame_idx)
        intrinsics.append(frame.intrinsics.float())
        c2w.append(frame.pose.matrix().float())
    return _VipePoses(
        intrinsics=torch.stack(intrinsics, dim=0),
        extrinsics_c2w=torch.stack(c2w, dim=0),
        frame_ids=torch.tensor(frame_ids, dtype=torch.int64),
    )


def _compute_aligned_pred_w2c(pred_extr: np.ndarray, input_w2c: np.ndarray) -> np.ndarray:
    """Map DA3 predicted extrinsics into the input (ViPE) world frame.

    Args:
        pred_extr: DA3 predicted world-to-camera extrinsics ``(N, 3|4, 4)``.
        input_w2c: Input world-to-camera extrinsics ``(N, 3|4, 4)``.

    Returns:
        Aligned world-to-camera extrinsics ``(N, 4, 4)``.
    """
    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.geometry import (
        affine_inverse_np,
    )
    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.pose_align import (
        align_poses_umeyama,
    )

    pred_44 = _pad_to_44(pred_extr.copy())
    r, t, s = align_poses_umeyama(pred_44, _pad_to_44(input_w2c.copy()), retry_degenerate=True)
    r_inv = r.T
    pred_c2w = affine_inverse_np(pred_44)

    aligned_c2w = np.zeros_like(pred_c2w)
    aligned_c2w[:, :3, :3] = np.einsum("ij,njk->nik", r_inv, pred_c2w[:, :3, :3])
    aligned_c2w[:, :3, 3] = np.einsum("ij,nj->ni", r_inv, pred_c2w[:, :3, 3] - t[None, :]) / s
    aligned_c2w[:, 3, 3] = 1.0
    return affine_inverse_np(aligned_c2w).astype(np.float32)


def _interpolate_w2c(w2c_keyframes: np.ndarray, key_indices: Sequence[int], n_total: int) -> np.ndarray:
    """Interpolate keyframe extrinsics to every frame (SLERP rotations, linear translations).

    Args:
        w2c_keyframes: Keyframe world-to-camera extrinsics ``(K, 3|4, 4)``.
        key_indices: Frame index of each keyframe, increasing.
        n_total: Number of output frames; frames outside the keyframe range are clamped.
    """
    w2c_keyframes = _pad_to_44(w2c_keyframes)
    if len(key_indices) == 1:
        return np.repeat(w2c_keyframes[:1], n_total, axis=0).astype(np.float32)

    from scipy.spatial.transform import Rotation, Slerp

    c2w = np.linalg.inv(w2c_keyframes)
    times_key = np.asarray(key_indices, dtype=np.float64)
    translations = c2w[:, :3, 3].astype(np.float64)
    times = np.clip(np.arange(n_total, dtype=np.float64), times_key[0], times_key[-1])

    c2w_dense = np.zeros((n_total, 4, 4), dtype=np.float64)
    c2w_dense[:, :3, :3] = Slerp(times_key, Rotation.from_matrix(c2w[:, :3, :3]))(times).as_matrix()
    c2w_dense[:, :3, 3] = np.column_stack([np.interp(times, times_key, translations[:, dim]) for dim in range(3)])
    c2w_dense[:, 3, 3] = 1.0
    return np.linalg.inv(c2w_dense).astype(np.float32)


def _load_gaussian_ply(ply_path: Path, device: torch.device) -> Any:
    """Load a 3DGS PLY written by DA3 ``save_gaussian_ply`` back into DA3 ``Gaussians``.

    Args:
        ply_path: Gaussian PLY file.
        device: Device for the returned tensors.
    """
    import torch
    from plyfile import PlyData

    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import Gaussians

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"No 'vertex' element in PLY: {ply_path}")
    vertices = ply["vertex"].data
    names = list(vertices.dtype.names or [])

    def _column(key: str) -> np.ndarray:
        if key not in names:
            raise ValueError(f"Missing '{key}' in PLY: {ply_path}")
        return vertices[key].astype(np.float32, copy=False)

    def _stack(prefix: str, count: int) -> np.ndarray:
        return np.stack([_column(f"{prefix}{idx}") for idx in range(count)], axis=1)

    means = np.stack([_column("x"), _column("y"), _column("z")], axis=1)
    scales = np.exp(_stack("scale_", 3))
    rotations = _stack("rot_", 4)
    opacities = 1.0 / (1.0 + np.exp(-_column("opacity")))
    f_dc = _stack("f_dc_", 3)
    f_rest_keys = sorted((key for key in names if key.startswith("f_rest_")), key=lambda key: int(key.split("_")[-1]))
    if f_rest_keys:
        f_rest = np.stack([_column(key) for key in f_rest_keys], axis=1)
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected f_rest size {f_rest.shape[1]} in PLY: {ply_path}")
        f_rest = f_rest.reshape(f_rest.shape[0], 3, f_rest.shape[1] // 3)
        harmonics = np.concatenate([f_dc[:, :, None], f_rest], axis=2)
    else:
        harmonics = f_dc[:, :, None]

    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=device).unsqueeze(0)

    return Gaussians(
        means=_tensor(means),
        scales=_tensor(scales),
        rotations=_tensor(rotations),
        harmonics=_tensor(harmonics),
        opacities=_tensor(opacities),
    )


def _empty_cuda_cache() -> None:
    """Release cached CUDA memory between reconstruction stages."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reconstruct_video_gaussians(
    input_video_path: str | Path,
    output_dir: str | Path,
    *,
    da3_model_path_custom: str | Path | None,
    da3_model_name: str = DEFAULT_DA3_MODEL_NAME,
    device: str | None = None,
    force: bool = False,
    no_vipe: bool = False,
    vipe_overrides: Sequence[str] | None = None,
    vipe_full_mode: bool = False,
    max_frames: int = 0,
    da3_max_frames: int = 128,
    da3_process_res: int | None = None,
    da3_process_method: Literal["upper_bound_resize", "lower_bound_resize"] = "upper_bound_resize",
    max_resolution: int = 0,
    gs_down_ratio: int = 2,
    gs_scale_extra_multiplier: float = 1.0,
    gs_ply_prune_opacity_percentile: float | None = None,
    gs_ds_feature_mode: bool = True,
    use_da3_render_pose: bool = True,
    render_fps: float | None = None,
    render_chunk_size: int = 1,
) -> Lyra2ReconstructionOutputs:
    """Reconstruct 3D Gaussians from one video and render them along its camera path.

    ViPE estimates poses for every frame; DA3 reconstructs Gaussians from at most
    ``da3_max_frames`` uniformly sampled views conditioned on those poses. With
    ``no_vipe`` a first DA3 pass predicts the poses that condition the second,
    Gaussian pass. Outputs are ``reconstructed_scene.ply``, ``gs_trajectory.mp4``,
    ``cameras.npz`` and, with ViPE, ``vipe_predictions.npz``; a ``.done`` marker
    makes later calls return the existing artifacts unless ``force`` is set.

    Args:
        input_video_path: Input video file.
        output_dir: Directory receiving the artifacts.
        da3_model_path_custom: Lyra-2 reconstruction DA3 checkpoint; ``None`` loads ``da3_model_name`` weights.
        da3_model_name: DA3 architecture and pretrained weights identifier.
        device: Torch device; defaults to CUDA when available.
        force: Recompute even when the ``.done`` marker exists.
        no_vipe: Skip ViPE and render with DA3 predicted poses.
        vipe_overrides: Hydra overrides replacing the default Lyra-2 ViPE overrides.
        vipe_full_mode: Keep ViPE instance segmentation and artifact outputs enabled.
        max_frames: Maximum number of video frames to read; ``0`` reads all.
        da3_max_frames: Maximum number of views passed to DA3.
        da3_process_res: DA3 processing resolution; overrides ``max_resolution``.
        da3_process_method: DA3 resize method used with ``da3_process_res``.
        max_resolution: When positive, DA3 short-side resolution (``lower_bound_resize``).
        gs_down_ratio: Spatial downsampling factor of the DA3 Gaussian head.
        gs_scale_extra_multiplier: Multiplier applied to the maximum Gaussian scale.
        gs_ply_prune_opacity_percentile: Drop this fraction of lowest-opacity Gaussians from the PLY.
        gs_ds_feature_mode: Average-pool full-resolution GS image features when downsampling.
        use_da3_render_pose: Render with DA3 poses aligned to ViPE instead of raw ViPE poses.
        render_fps: Output video frame rate; defaults to the rounded input frame rate.
        render_chunk_size: Number of views rendered per chunk.
    """
    import torch

    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.model.utils.gs_renderer import (
        run_renderer_in_chunk_w_trj_mode,
    )
    from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.gsply_helpers import (
        save_gaussian_ply,
    )
    from worldfoundry.core.media.codecs.video import save_video_h264
    from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference.depth_utils import load_da3_model

    input_video = Path(input_video_path).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(output_root, no_vipe=no_vipe)

    done_marker = output_root / ".done"
    if done_marker.is_file() and not force:
        log.info(f"{_LOG_PREFIX} Skipping {input_video.name}: {done_marker} exists.")
        return outputs

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if da3_model_path_custom is not None:
        da3_model_path_custom = Path(da3_model_path_custom).expanduser().resolve()
        if not da3_model_path_custom.is_file():
            raise FileNotFoundError(f"DA3 checkpoint not found: {da3_model_path_custom}")
    log.info(f"{_LOG_PREFIX} input={input_video} output={output_root} device={torch_device}")

    da3_model = load_da3_model(
        da3_model_name=da3_model_name,
        da3_model_path_custom=None if da3_model_path_custom is None else str(da3_model_path_custom),
        device=str(torch_device),
    )
    da3_model.eval()

    images_all, fps = _read_video_frames(input_video, max_frames)
    if not images_all:
        raise RuntimeError(f"No frames read from video: {input_video}")
    indices_all = list(range(len(images_all)))
    indices_da3 = _uniform_subsample_indices(len(images_all), da3_max_frames)
    images_da3 = [images_all[idx] for idx in indices_da3]
    log.info(f"{_LOG_PREFIX} fps={fps:.4g} no_vipe={no_vipe} frames_all={len(images_all)} frames_da3={len(images_da3)}")

    w2c_vipe = k_vipe = w2c_da3 = k_da3 = None
    if not no_vipe:
        with tempfile.TemporaryDirectory(prefix="lyra2_vipe_") as vipe_tmp:
            poses = _run_vipe(
                images_all,
                fps=fps,
                name=input_video.stem,
                overrides=vipe_overrides or _vipe_default_overrides(Path(vipe_tmp)),
                fast_mode=not vipe_full_mode,
                device=torch_device,
            )
        w2c_vipe = torch.linalg.inv(poses.extrinsics_c2w.float()).cpu().numpy().astype(np.float32)
        k_vipe = _intrinsics_vec_to_k33(poses.intrinsics.float()).cpu().numpy().astype(np.float32)
        w2c_da3 = w2c_vipe[indices_da3]
        k_da3 = k_vipe[indices_da3]
        np.savez(
            outputs.vipe_prediction_path,
            frame_ids=poses.frame_ids.cpu().numpy().astype(np.int64),
            w2c_vipe=w2c_vipe,
            intrinsics_vipe=k_vipe,
            w2c_da3=w2c_da3,
            intrinsics_da3=k_da3,
            indices_vipe=np.asarray(indices_all, dtype=np.int64),
            indices_da3=np.asarray(indices_da3, dtype=np.int64),
            fps=np.asarray([fps], dtype=np.float32),
            input_video_path=np.asarray([str(input_video)]),
        )

    if da3_process_res is not None:
        process_res, process_method = int(da3_process_res), str(da3_process_method)
    elif max_resolution > 0:
        process_res, process_method = int(max_resolution), "lower_bound_resize"
    else:
        process_res, process_method = int(max(images_da3[0].shape[:2])), "upper_bound_resize"

    da3_pred_w2c = da3_pred_k = None
    if no_vipe:
        log.info(f"{_LOG_PREFIX} DA3 pose pass: views={len(images_da3)} process_res={process_res}")
        pose_pred = da3_model.inference(
            image=images_da3,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            align_to_input_pose=False,
            infer_gs=False,
            process_res=process_res,
            process_res_method=process_method,
        )
        if pose_pred.extrinsics is None or pose_pred.intrinsics is None:
            raise RuntimeError("DA3 pose pass did not return predicted poses.")
        w2c_da3 = _pad_to_44(np.asarray(pose_pred.extrinsics, dtype=np.float32))
        k_da3 = np.asarray(pose_pred.intrinsics, dtype=np.float32)
        da3_pred_w2c, da3_pred_k = w2c_da3.copy(), k_da3.copy()
        del pose_pred
        _empty_cuda_cache()

    log.info(f"{_LOG_PREFIX} DA3 Gaussian pass: views={len(images_da3)} process_res={process_res}")
    pred = da3_model.inference(
        image=images_da3,
        extrinsics=w2c_da3,
        intrinsics=k_da3,
        align_to_input_ext_scale=False,
        align_to_input_pose=False,
        infer_gs=True,
        process_res=process_res,
        process_res_method=process_method,
        use_aligned_pred_cam=True,
        gs_down_ratio=gs_down_ratio,
        gs_scale_extra_multiplier=gs_scale_extra_multiplier,
        gs_ds_feature_mode=gs_ds_feature_mode,
    )

    aligned_w2c_da3 = None
    if not no_vipe and use_da3_render_pose and pred.extrinsics is not None:
        aligned_w2c_da3 = _compute_aligned_pred_w2c(np.asarray(pred.extrinsics, dtype=np.float32), w2c_da3)

    prune_by_opacity = gs_ply_prune_opacity_percentile is not None and gs_ply_prune_opacity_percentile > 0
    save_gaussian_ply(
        pred.gaussians,
        str(outputs.reconstructed_scene_path),
        ctx_depth=torch.from_numpy(np.asarray(pred.depth, dtype=np.float32)).float().unsqueeze(-1),
        prune_by_opacity_percentile=gs_ply_prune_opacity_percentile,
        prune_border_gs=not prune_by_opacity,
    )
    log.info(f"{_LOG_PREFIX} Saved PLY to {outputs.reconstructed_scene_path}")

    if no_vipe:
        w2c_render = _interpolate_w2c(da3_pred_w2c, indices_da3, len(images_all))
        k_render = np.repeat(da3_pred_k[0:1], len(images_all), axis=0).astype(np.float32)
        log.info(f"{_LOG_PREFIX} Rendering with DA3 predicted poses.")
    elif aligned_w2c_da3 is not None:
        w2c_render = _interpolate_w2c(aligned_w2c_da3, indices_da3, len(images_all))
        k_render = k_vipe
        log.info(f"{_LOG_PREFIX} Rendering with DA3-aligned poses.")
    else:
        w2c_render = w2c_vipe
        k_render = k_vipe
        if use_da3_render_pose:
            log.warning(f"{_LOG_PREFIX} DA3 poses unavailable, rendering with ViPE poses.")

    cameras: dict[str, np.ndarray] = {
        "w2c_render": w2c_render,
        "indices_da3": np.asarray(indices_da3, dtype=np.int64),
        "fps": np.asarray([fps], dtype=np.float32),
        "no_vipe": np.asarray([int(no_vipe)], dtype=np.int32),
    }
    if no_vipe:
        cameras.update(w2c_da3_pred=da3_pred_w2c, intrinsics_da3_pred=da3_pred_k, intrinsics_render=k_render)
    else:
        cameras.update(
            w2c_vipe=w2c_vipe,
            intrinsics_vipe=k_vipe,
            w2c_da3=w2c_da3,
            intrinsics_da3=k_da3,
            indices_vipe=np.asarray(indices_all, dtype=np.int64),
            use_da3_render_pose=np.asarray([int(use_da3_render_pose)], dtype=np.int32),
        )
    np.savez(outputs.camera_path, **cameras)

    del pred
    _empty_cuda_cache()

    gs_device = next(da3_model.model.parameters()).device
    gaussians = _load_gaussian_ply(outputs.reconstructed_scene_path, device=gs_device)
    dtype = gaussians.means.dtype
    render_extr = torch.from_numpy(_pad_to_44(w2c_render)).to(device=gs_device, dtype=dtype)[None]
    render_intr = torch.from_numpy(k_render).to(device=gs_device, dtype=dtype)[None]
    render_h, render_w = images_all[0].shape[:2]
    output_fps = float(render_fps) if render_fps is not None else float(max(1, round(fps)))
    log.info(f"{_LOG_PREFIX} Rendering {render_extr.shape[1]} frames at {render_h}x{render_w} fps={output_fps:.2f}")
    color, _ = run_renderer_in_chunk_w_trj_mode(
        gaussians=gaussians,
        extrinsics=render_extr,
        intrinsics=render_intr,
        image_shape=(render_h, render_w),
        chunk_size=int(render_chunk_size),
        trj_mode="original",
        use_sh=True,
        color_mode="RGB+ED",
        enable_tqdm=True,
    )
    frames = color[0].clamp(0.0, 1.0).mul(255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    save_video_h264(frames, outputs.render_video_path, fps=max(output_fps, 1.0), crf=18, preset="slow")
    log.info(f"{_LOG_PREFIX} Saved Gaussian render to {outputs.render_video_path}")

    del gaussians, render_extr, render_intr, color, frames
    _empty_cuda_cache()
    done_marker.write_text("done\n")
    return outputs


__all__ = ["DEFAULT_DA3_MODEL_NAME", "Lyra2ReconstructionOutputs", "reconstruct_video_gaussians"]
