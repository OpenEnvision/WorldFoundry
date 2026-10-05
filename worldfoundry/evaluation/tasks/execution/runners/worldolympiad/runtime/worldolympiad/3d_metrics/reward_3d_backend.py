#!/usr/bin/env python3
"""
3D reward backend built on top of Depth Anything 3.
"""

from __future__ import annotations

from _bootstrap import setup_paths

setup_paths()

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import difflib
from typing import List, Tuple, Optional
import os
import sys
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.specs import Gaussians


def _default_reconstruction_model() -> str:
    local_model_dir = Path(__file__).resolve().parents[1] / "weights" / "da3"
    if local_model_dir.is_dir():
        return str(local_model_dir)
    return "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


DEFAULT_RECONSTRUCTION_MODEL = _default_reconstruction_model()


def _suggest_existing_path(candidate: Path) -> str | None:
    """Suggest a nearby existing path when the requested local path is missing."""
    existing_ancestor = candidate
    missing_parts = []

    while not existing_ancestor.exists() and existing_ancestor.parent != existing_ancestor:
        missing_parts.append(existing_ancestor.name)
        existing_ancestor = existing_ancestor.parent

    if not existing_ancestor.exists() or not missing_parts or not existing_ancestor.is_dir():
        return None

    next_missing = missing_parts[-1]
    siblings = [child.name for child in existing_ancestor.iterdir() if child.is_dir()]
    matches = difflib.get_close_matches(next_missing, siblings, n=1, cutoff=0.6)
    if not matches:
        return None

    suggestion = existing_ancestor / matches[0]
    for part in reversed(missing_parts[:-1]):
        suggestion = suggestion / part
    return str(suggestion)


def resolve_model_source(model_name: str | os.PathLike[str]) -> str:
    """Resolve local model directories before falling back to Hugging Face ids."""
    model_name = os.fspath(model_name)
    candidate = Path(model_name).expanduser()

    # Treat explicit filesystem-like inputs as local paths.
    looks_like_path = (
        candidate.is_absolute()
        or model_name.startswith(".")
        or model_name.startswith("~")
        or candidate.exists()
    )
    if looks_like_path:
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if not candidate.exists():
            suggestion = _suggest_existing_path(candidate)
            suggestion_text = f" Did you mean: {suggestion} ?" if suggestion else ""
            raise FileNotFoundError(
                f"Local DA3 model path does not exist: {candidate}. "
                f"Use a valid local directory or a Hugging Face model id.{suggestion_text}"
            )
        if not candidate.is_dir():
            raise NotADirectoryError(f"DA3 model path must be a directory: {candidate}")

        required_files = ("config.json", "model.safetensors")
        missing_files = [name for name in required_files if not (candidate / name).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Local DA3 model directory is missing required files {missing_files}: {candidate}"
            )
        return str(candidate)

    return model_name

class Reward3DBackend:
    """Generate 3D reward artifacts from a video rollout."""

    def __init__(self, device='cuda', model_name=DEFAULT_RECONSTRUCTION_MODEL):
        """
        Initialize the 3D reward backend.

        Args:
            device: Device to run on (cuda/cpu)
            model_name: Reconstruction model name to load
        """
        self.device = device
        self.model_name = resolve_model_source(model_name)

        print(f"Loading 3D reconstruction model: {self.model_name}")
        self.model = DepthAnything3.from_pretrained(self.model_name).to(device)
        self.model.eval()
        print("3D reconstruction model loaded successfully")

    def process_video_frames(
        self,
        frames: List[np.ndarray],
        dynamic_masks: Optional[List[np.ndarray]] = None,
        process_res: int = 504,
        gs_render_size: Tuple[int, int] = (512, 512),
        trj_mode: str = "extend",
        chunk_size: int = 4,
        camera_trajectory=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[np.ndarray]]:
        """
        Process video frames and generate 3D reward renderings.

        Args:
            frames: List of RGB frames as numpy arrays (H, W, 3) in range [0, 255]
            dynamic_masks: Optional binary masks where True marks dynamic regions to remove from GS
            process_res: Processing resolution for reconstruction
            gs_render_size: Output size for GS rendering (H, W)
            trj_mode: Trajectory mode for GS rendering ('extend', 'interpolate', etc.)
            chunk_size: Chunk size for rendering
            camera_trajectory: Target camera trajectory for motion similarity scoring

        Returns:
            gs_video: Rendered GS video frames (T, 3, H, W) in range [0, 1]
            meta_view: Representative depth visualization image (3, H, W) in range [0, 1]
            camera_motion_score: Similarity score for camera movement
            trajectory_comparison_image: HWC uint8 comparison image if GT trajectory exists
        """
        print(f"Running 3D reconstruction inference on {len(frames)} frames...")
        prediction = self.model.inference(
            image=frames,
            extrinsics=None,
            intrinsics=None,
            process_res=process_res,
            align_to_input_ext_scale=True,
            infer_gs=True,  # Enable Gaussian Splatting
            export_dir=None,  # Don't export to disk
        )

        self._filter_gaussians_by_dynamic_masks(prediction, dynamic_masks)

        # Generate GS video
        print("Generating GS video...")
        gs_video = self._generate_gs_video(
            prediction,
            render_size=gs_render_size,
            trj_mode=trj_mode,
            chunk_size=chunk_size,
        )

        # Generate meta view (single GS-rendered image from farthest camera)
        print("Generating meta view...")
        meta_view = self._generate_meta_view(prediction, render_size=gs_render_size)

        # Compute camera motion score
        print("Computing camera motion score...")
        camera_motion_score, pred_pose_rel, target_pose_rel = self._compute_camera_motion_score(
            prediction, camera_trajectory
        )
        trajectory_comparison_image = self._render_trajectory_comparison(
            pred_pose_rel, target_pose_rel
        )

        return gs_video, meta_view, camera_motion_score, trajectory_comparison_image

    def _filter_gaussians_by_dynamic_masks(self, prediction, dynamic_masks: Optional[List[np.ndarray]]) -> None:
        if dynamic_masks is None or prediction.gaussians is None:
            return

        gaussians = prediction.gaussians
        if gaussians.means.ndim != 3 or gaussians.means.shape[0] != 1:
            print(f"[WARN] Unexpected Gaussian shape; skip dynamic filtering: {tuple(gaussians.means.shape)}")
            return

        total_gaussians = int(gaussians.means.shape[1])
        frame_count = len(dynamic_masks)
        if frame_count == 0 or total_gaussians % frame_count != 0:
            print(
                "[WARN] Dynamic mask count does not match Gaussian layout; "
                f"masks={frame_count}, gaussians={total_gaussians}. Skipping filter."
            )
            return

        if prediction.depth is not None and len(prediction.depth) == frame_count:
            mask_height, mask_width = prediction.depth.shape[1:3]
            if mask_height * mask_width != total_gaussians // frame_count:
                print(
                    "[WARN] Prediction depth resolution does not match Gaussian layout; "
                    f"depth={(mask_height, mask_width)}, pixels_per_frame={total_gaussians // frame_count}. Skipping filter."
                )
                return
        else:
            print("[WARN] Prediction depth unavailable for dynamic mask alignment; skipping filter.")
            return

        pixels_per_frame = total_gaussians // frame_count
        mask_flattened = []
        for mask in dynamic_masks:
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask_bool = mask.astype(bool)
            if mask_bool.size != pixels_per_frame:
                import cv2

                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (mask_width, mask_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            mask_flattened.append(mask_bool.reshape(-1))

        dynamic_flat = np.concatenate(mask_flattened, axis=0)
        if dynamic_flat.size != total_gaussians:
            print(
                "[WARN] Dynamic mask flattened size does not match Gaussian count; "
                f"mask={dynamic_flat.size}, gaussians={total_gaussians}. Skipping filter."
            )
            return

        keep = torch.from_numpy(~dynamic_flat).to(device=gaussians.means.device, dtype=torch.bool)
        keep_count = int(keep.sum().item())
        if keep_count == 0:
            print("[WARN] Dynamic masks would remove all Gaussians; skipping filter.")
            return

        removed_count = total_gaussians - keep_count
        prediction.gaussians = Gaussians(
            means=gaussians.means[:, keep, :],
            scales=gaussians.scales[:, keep, :],
            rotations=gaussians.rotations[:, keep, :],
            harmonics=gaussians.harmonics[:, keep, :, :],
            opacities=gaussians.opacities[:, keep, ...],
        )
        print(f"Filtered dynamic Gaussians: removed={removed_count}, kept={keep_count}")

    def _generate_gs_video(
        self,
        prediction,
        render_size: Tuple[int, int],
        trj_mode: str,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Generate GS rendering video from prediction

        Args:
            prediction: Reconstruction prediction object
            render_size: Output size (H, W)
            trj_mode: Trajectory mode
            chunk_size: Chunk size for rendering

        Returns:
            gs_video: (T, 3, H, W) tensor in range [0, 1]
        """
        gaussians = prediction.gaussians

        # Prepare extrinsics and intrinsics
        extrinsics = torch.from_numpy(prediction.extrinsics).unsqueeze(0).to(gaussians.means)
        intrinsics = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(gaussians.means)

        # Adjust extrinsics scale if metric
        if prediction.is_metric and prediction.scale_factor is not None:
            extrinsics[:, :, :3, 3] /= prediction.scale_factor

        H, W = render_size

        # Render GS video
        color, _ = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(H, W),
            chunk_size=chunk_size,
            trj_mode=trj_mode,
            use_sh=True,
            color_mode="RGB+D",
            enable_tqdm=True,
        )

        # color shape: (batch, num_views, 3, H, W)
        # We only have one batch, so squeeze it
        gs_video = color.squeeze(0)  # (T, 3, H, W)

        # Clamp to [0, 1]
        gs_video = gs_video.clamp(0, 1)

        return gs_video

    def _generate_meta_view(self, prediction, render_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Generate meta view by rendering GS from the farthest camera position

        Meta view is a single GS-rendered image from the camera position that is
        farthest from the origin, providing the best overview of the 3D reconstruction.

        Args:
            prediction: Reconstruction prediction object
            render_size: Output image size (H, W)

        Returns:
            meta_view: (3, H, W) tensor in range [0, 1] - single rendered image
        """
        from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous
        from depth_anything_3.model.utils.gs_renderer import render_3dgs

        gaussians = prediction.gaussians
        H, W = render_size

        # Get camera extrinsics (world2cam) - may be (N, 3, 4) or (N, 4, 4)
        # prediction.extrinsics: numpy array (N, 3, 4) or (N, 4, 4)
        # prediction.intrinsics: numpy array (N, 3, 3)
        extrinsics = torch.from_numpy(prediction.extrinsics).to(gaussians.means)
        intrinsics = torch.from_numpy(prediction.intrinsics).to(gaussians.means)

        # Convert to homogeneous if needed (N, 3, 4) -> (N, 4, 4)
        extrinsics = as_homogeneous(extrinsics)

        # Adjust scale if metric
        if prediction.is_metric and prediction.scale_factor is not None:
            extrinsics[:, :3, 3] /= prediction.scale_factor

        # Convert to camera2world to get camera positions
        cam2world = affine_inverse(extrinsics)  # (N, 4, 4)
        camera_positions = cam2world[:, :3, 3]  # (N, 3)

        # Find the camera farthest from origin
        distances = torch.norm(camera_positions, dim=1)  # (N,)
        farthest_idx = torch.argmax(distances).item()

        print(f"Selected camera {farthest_idx}/{len(distances)} (distance: {distances[farthest_idx]:.3f})")

        # Extract the farthest camera's parameters
        # render_3dgs expects:
        #   extrinsics: (batch_views, 4, 4)
        #   intrinsics: (batch_views, 3, 3) - normalized
        meta_extrinsics = extrinsics[farthest_idx:farthest_idx+1]  # (1, 4, 4)
        meta_intrinsics = intrinsics[farthest_idx:farthest_idx+1].clone()  # (1, 3, 3)

        # Normalize intrinsics by dividing by image dimensions
        meta_intrinsics[:, 0, :] /= W  # Normalize x coordinates
        meta_intrinsics[:, 1, :] /= H  # Normalize y coordinates

        # Render GS directly using render_3dgs
        color, _ = render_3dgs(
            extrinsics=meta_extrinsics,
            intrinsics=meta_intrinsics,
            image_shape=(H, W),
            gaussian=gaussians,
            use_sh=True,
            num_view=1,
            color_mode="RGB+D",
        )

        # color shape: (1, 3, H, W)
        # Extract single image
        meta_view = color[0]  # (3, H, W)

        # Clamp to [0, 1]
        meta_view = meta_view.clamp(0, 1)

        return meta_view

    def _parse_camera_matrix(self, matrix_str: str) -> np.ndarray:
        cols_str = matrix_str.strip().split('] [')
        cols = []
        for col_str in cols_str:
            col_str = col_str.replace('[', '').replace(']', '').strip()
            if not col_str:
                continue
            values = [float(x) for x in col_str.split()]
            cols.append(values)
        return np.array(cols, dtype=np.float32).T

    def _ensure_homogeneous_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape == (4, 4):
            return matrix
        if matrix.shape == (3, 4):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :] = matrix
            return pose
        if matrix.shape == (3,):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = matrix
            return pose
        raise ValueError(f"Unsupported trajectory matrix shape: {matrix.shape}")

    def _trajectory_frame_keys(self, camera_trajectory) -> List[str]:
        try:
            return sorted(
                camera_trajectory.keys(),
                key=lambda x: int(str(x).replace("frame", "")),
            )
        except ValueError:
            return list(camera_trajectory.keys())

    def _select_frame_indices(self, total_frames: int, target_len: int) -> List[int]:
        if total_frames <= 0 or target_len <= 0:
            return []
        if target_len >= total_frames:
            return list(range(total_frames)) + [total_frames - 1] * (target_len - total_frames)
        if target_len == 1:
            return [0]
        return [
            int(round(i * (total_frames - 1) / (target_len - 1)))
            for i in range(target_len)
        ]

    def _extract_target_camera_poses(self, camera_trajectory, target_len: int) -> np.ndarray | None:
        if camera_trajectory is None:
            return None

        if isinstance(camera_trajectory, str):
            try:
                camera_trajectory = json.loads(camera_trajectory)
            except json.JSONDecodeError:
                return None

        pose_sequence = []
        if isinstance(camera_trajectory, dict):
            for key in self._trajectory_frame_keys(camera_trajectory):
                value = camera_trajectory[key]
                matrix = self._parse_camera_matrix(value) if isinstance(value, str) else value
                pose_sequence.append(self._ensure_homogeneous_matrix(matrix))
        elif isinstance(camera_trajectory, (list, tuple, np.ndarray)):
            array = np.asarray(camera_trajectory, dtype=np.float32)
            if array.ndim == 3:
                for matrix in array:
                    pose_sequence.append(self._ensure_homogeneous_matrix(matrix))
            elif array.ndim == 2 and array.shape[1] >= 3:
                for point in array:
                    pose_sequence.append(self._ensure_homogeneous_matrix(point[:3]))
            else:
                return None
        else:
            return None

        if not pose_sequence:
            return None

        pose_sequence = np.stack(pose_sequence, axis=0)
        indices = self._select_frame_indices(len(pose_sequence), target_len)
        if not indices:
            return None
        return pose_sequence[indices]

    def _resample_pose_sequence(self, poses: np.ndarray, target_len: int) -> np.ndarray:
        if poses.shape[0] == target_len:
            return poses
        indices = self._select_frame_indices(poses.shape[0], target_len)
        return poses[indices]

    def _align_pose_sequence_to_first(self, poses: np.ndarray) -> np.ndarray:
        first_inv = np.linalg.inv(poses[0])
        return np.stack([first_inv @ pose for pose in poses], axis=0)

    def _rotation_geodesic_deg(self, rot_a: np.ndarray, rot_b: np.ndarray) -> float:
        rel = rot_a.T @ rot_b
        trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace)))

    def _total_rotation_deg(self, poses: np.ndarray) -> float:
        if len(poses) < 2:
            return 0.0
        total = 0.0
        for idx in range(1, len(poses)):
            total += self._rotation_geodesic_deg(
                poses[idx - 1, :3, :3],
                poses[idx, :3, :3],
            )
        return total

    def _total_translation(self, centers: np.ndarray) -> float:
        if len(centers) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())

    def _flatten_cos_score(self, source: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        source_norm = float(np.linalg.norm(source_flat))
        target_norm = float(np.linalg.norm(target_flat))
        if source_norm < eps and target_norm < eps:
            return 1.0
        if source_norm < eps or target_norm < eps:
            return 0.0
        cos_sim = float(np.dot(source_flat, target_flat) / (source_norm * target_norm))
        return 0.5 * (np.clip(cos_sim, -1.0, 1.0) + 1.0)

    def _compute_motion_profile_scores(
        self,
        pred_poses: np.ndarray,
        target_poses: np.ndarray,
    ) -> float:
        pred_centers = pred_poses[:, :3, 3]
        target_centers = target_poses[:, :3, 3]

        pred_trans_total = self._total_translation(pred_centers)
        target_trans_total = self._total_translation(target_centers)
        pred_rot_total = self._total_rotation_deg(pred_poses)
        target_rot_total = self._total_rotation_deg(target_poses)
        step_count = max(len(target_poses) - 1, 1)
        target_trans_per_step = target_trans_total / step_count
        target_rot_per_step = target_rot_total / step_count

        trans_eps = 1e-4
        rot_eps = 1e-3
        eps = 1e-6
        static_trans_per_step_eps = 0.01
        static_rot_per_step_eps = 0.5

        # DA3 trajectories extracted from mostly static-camera videos can contain
        # small frame-to-frame jitter. Treat such trajectories as near-static so
        # the score reflects consistency instead of matching noise.
        target_is_near_static = (
            target_trans_per_step < static_trans_per_step_eps
            and target_rot_per_step < static_rot_per_step_eps
        )

        if target_trans_total < trans_eps or target_is_near_static:
            trans_score = float(np.exp(-pred_trans_total / 3.0))
        else:
            mean_translation_error = float(np.linalg.norm(pred_centers - target_centers, axis=1).mean())
            translation_scale = max(target_trans_total / max(len(target_centers) - 1, 1), 1.0)
            mean_error_score = float(np.exp(-mean_translation_error / (2.0 * translation_scale)))
            path_score = self._flatten_cos_score(pred_centers, target_centers)
            extent_score = float(
                np.exp(-abs(np.log((pred_trans_total + eps) / (target_trans_total + eps))))
            )
            trans_score = 0.4 * path_score + 0.3 * extent_score + 0.3 * mean_error_score

        rotation_errors = [
            self._rotation_geodesic_deg(target_pose[:3, :3], pred_pose[:3, :3])
            for pred_pose, target_pose in zip(pred_poses, target_poses)
        ]
        mean_rotation_error = float(np.mean(rotation_errors))
        final_rotation_error = float(rotation_errors[-1]) if rotation_errors else 180.0

        if target_rot_total < rot_eps or target_is_near_static:
            rot_score = float(np.exp(-pred_rot_total / 5.0))
        else:
            mean_error_score = float(np.exp(-mean_rotation_error / 6.0))
            final_error_score = float(np.exp(-final_rotation_error / 6.0))
            extent_score = float(
                np.exp(-abs(np.log((pred_rot_total + eps) / (target_rot_total + eps))))
            )
            rot_score = 0.5 * mean_error_score + 0.3 * final_error_score + 0.2 * extent_score

        if target_is_near_static or (target_trans_total < trans_eps and target_rot_total < rot_eps):
            final_score = trans_score * rot_score
        elif target_trans_total < trans_eps:
            final_score = 0.2 * trans_score + 0.8 * rot_score
        elif target_rot_total < rot_eps:
            final_score = 0.8 * trans_score + 0.2 * rot_score
        else:
            final_score = 0.5 * (trans_score + rot_score)

        final_score = float(np.clip(final_score, 0.0, 1.0))
        print(
            "Camera motion profile:"
            f" pred_trans={pred_trans_total:.4f}, gt_trans={target_trans_total:.4f},"
            f" pred_rot={pred_rot_total:.4f}, gt_rot={target_rot_total:.4f},"
            f" gt_trans_step={target_trans_per_step:.4f}, gt_rot_step={target_rot_per_step:.4f},"
            f" near_static={target_is_near_static},"
            f" trans_score={trans_score:.4f}, rot_score={rot_score:.4f}, final={final_score:.4f}"
        )
        return final_score

    def _camera_vertices(
        self,
        c2w: np.ndarray,
        hw_ratio: float = 9 / 16,
        base_xval: float = 0.08,
        zval: float = 0.15,
    ) -> list[np.ndarray]:
        vertex_std = np.array(
            [
                [0, 0, 0, 1],
                [base_xval, -base_xval * hw_ratio, zval, 1],
                [base_xval, base_xval * hw_ratio, zval, 1],
                [-base_xval, base_xval * hw_ratio, zval, 1],
                [-base_xval, -base_xval * hw_ratio, zval, 1],
            ],
            dtype=np.float32,
        )
        vertex_transformed = vertex_std @ c2w.T
        return [vertex_transformed[i, :-1] for i in range(5)]

    def _compute_bounds(self, *pose_sets: Optional[np.ndarray]) -> tuple[list[float], list[float], list[float]]:
        valid_sets = [poses for poses in pose_sets if poses is not None and len(poses) > 0]
        if not valid_sets:
            return [-1, 1], [-1, 1], [-1, 1]
        pts = np.concatenate([poses[:, :3, 3] for poses in valid_sets], axis=0)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        span = np.maximum(maxs - mins, 0.2)
        pad = np.maximum(span * 0.2, 0.1)
        return (
            [float(mins[0] - pad[0]), float(maxs[0] + pad[0])],
            [float(mins[1] - pad[1]), float(maxs[1] + pad[1])],
            [float(mins[2] - pad[2]), float(maxs[2] + pad[2])],
        )

    def _render_trajectory_comparison(
        self,
        pred_pose_rel: Optional[np.ndarray],
        target_pose_rel: Optional[np.ndarray],
        frustum_step: int = 3,
    ) -> Optional[np.ndarray]:
        if pred_pose_rel is None or target_pose_rel is None:
            return None

        xlim, ylim, zlim = self._compute_bounds(pred_pose_rel, target_pose_rel)
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(projection="3d")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("GT vs Predicted Camera Trajectory")

        def add_path(poses: np.ndarray, color: str, label: str):
            centers = poses[:, :3, 3]
            ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, linewidth=2.0, label=label)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color=color, s=18)
            for idx in range(0, len(poses), max(1, frustum_step)):
                vertices = self._camera_vertices(poses[idx])
                meshes = [
                    [vertices[0], vertices[1], vertices[2]],
                    [vertices[0], vertices[2], vertices[3]],
                    [vertices[0], vertices[3], vertices[4]],
                    [vertices[0], vertices[4], vertices[1]],
                    [vertices[1], vertices[2], vertices[3], vertices[4]],
                ]
                ax.add_collection3d(
                    Poly3DCollection(
                        meshes,
                        facecolors=color,
                        linewidths=0.3,
                        edgecolors=color,
                        alpha=0.2,
                    )
                )

        add_path(target_pose_rel, "#1f77b4", "GT")
        add_path(pred_pose_rel, "#d62728", "Pred")
        ax.legend(
            handles=[
                Line2D([0], [0], color="#1f77b4", lw=2, label="GT"),
                Line2D([0], [0], color="#d62728", lw=2, label="Pred"),
            ],
            loc="upper left",
        )
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
        plt.close(fig)
        return image

    def _compute_camera_motion_score(self, prediction, camera_trajectory=None) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute camera motion similarity between predicted and target trajectories.

        Args:
            prediction: Reconstruction prediction object with extrinsics
            camera_trajectory: Target camera trajectory (dict or list)

        Returns:
            score: Camera motion similarity score in range [0, 1]
        """
        from depth_anything_3.utils.geometry import as_homogeneous, affine_inverse_np
        from depth_anything_3.utils.pose_align import align_poses_umeyama

        pred_extrinsics = as_homogeneous(np.asarray(prediction.extrinsics, dtype=np.float32))
        if pred_extrinsics.shape[0] < 2:
            return 0.0, None, None

        target_poses = self._extract_target_camera_poses(
            camera_trajectory,
            target_len=pred_extrinsics.shape[0],
        )
        if target_poses is None:
            print("Camera trajectory missing or invalid; motion score set to 0.0")
            return 0.0, None, None

        target_extrinsics = np.stack(
            [np.linalg.inv(pose) for pose in target_poses],
            axis=0,
        ).astype(np.float32)

        common_len = min(pred_extrinsics.shape[0], target_extrinsics.shape[0])
        if common_len < 2:
            return 0.0, None, None

        pred_extrinsics = self._resample_pose_sequence(pred_extrinsics, common_len)
        target_extrinsics = self._resample_pose_sequence(target_extrinsics, common_len)

        target_pose_rel = self._align_pose_sequence_to_first(affine_inverse_np(target_extrinsics))
        target_trans_total = self._total_translation(target_pose_rel[:, :3, 3])

        pred_extrinsics_aligned = pred_extrinsics
        if target_trans_total >= 1e-4:
            try:
                _, _, _, pred_extrinsics_aligned = align_poses_umeyama(
                    target_extrinsics,
                    pred_extrinsics,
                    return_aligned=True,
                    ransac=common_len >= 6,
                )
            except Exception as exc:
                print(f"Pose alignment failed, falling back to first-frame normalization: {exc}")

        pred_pose_rel = self._align_pose_sequence_to_first(affine_inverse_np(pred_extrinsics_aligned))
        score = self._compute_motion_profile_scores(pred_pose_rel, target_pose_rel)
        return score, pred_pose_rel, target_pose_rel

    @torch.no_grad()
    def __call__(
        self,
        video_tensor: torch.Tensor,
        process_res: int = 504,
        gs_render_size: Tuple[int, int] = (512, 512),
        camera_trajectory=None,
        dynamic_masks: Optional[List[np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[np.ndarray]]:
        """
        Process video tensor and return gs_video and meta_view

        Args:
            video_tensor: Video tensor (1, T, C, H, W) or (T, C, H, W) in range [0, 1]
            process_res: Processing resolution for reconstruction
            gs_render_size: Output size for GS rendering
            camera_trajectory: Target camera trajectory for motion similarity scoring
            dynamic_masks: Optional binary masks where True marks dynamic regions to remove from GS

        Returns:
            gs_video: (T, 3, H, W) tensor in range [0, 1] - video sequence
            meta_view: (3, H, W) tensor in range [0, 1] - single image
            camera_motion_score: Similarity score for camera movement
            trajectory_comparison_image: Optional GT-vs-pred trajectory visualization
        """
        # Handle batch dimension
        if video_tensor.dim() == 5:
            video_tensor = video_tensor.squeeze(0)  # (T, C, H, W)

        # Convert to list of numpy arrays
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t]  # (C, H, W)
            # Convert to HWC format and to uint8
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame_np)

        # Process through the 3D reward backend.
        gs_video, meta_view, camera_motion_score, trajectory_comparison_image = self.process_video_frames(
            frames,
            dynamic_masks=dynamic_masks,
            process_res=process_res,
            gs_render_size=gs_render_size,
            camera_trajectory=camera_trajectory,
        )

        return gs_video, meta_view, camera_motion_score, trajectory_comparison_image
