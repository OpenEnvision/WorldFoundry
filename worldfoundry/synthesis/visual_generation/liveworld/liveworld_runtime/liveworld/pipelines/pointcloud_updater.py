"""Abstract base class for 3D reconstruction handlers.

Each handler manages:
1. First-frame 3D reconstruction (replaces depth estimation)
2. Incremental point cloud updates (streaming or batched)
3. Scene projection rendering (point cloud → VAE latent)
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.geometry_utils import (
    _limit_points_by_density,
    _safe_frame_index,
    render_projection,
    scale_intrinsics,
)


@dataclass
class ReconstructionResult:
    """Result of first-frame 3D reconstruction."""

    points_world: np.ndarray  # (M, 3) float32
    colors: np.ndarray  # (M, 3) uint8
    intrinsics: np.ndarray  # (3, 3) float32
    intrinsics_size: tuple[int, int]  # (H, W) of the model's output resolution
    alignment_transform: np.ndarray


class PointCloudUpdater(ABC):
    """Abstract base for 3D reconstruction backends (Stream3R, MapAnything).

    Each backend:
    - Reconstructs initial point cloud from first frame
    - Handles incremental updates with internal ICP alignment
    - Renders scene projections from its internal point cloud
    - Owns the authoritative point cloud, poses, and intrinsics
    """

    # -- Lifecycle -------------------------------------------------------

    @abstractmethod
    def init_model(self, device: torch.device) -> None:
        """Load the reconstruction model onto ``device``."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release model / session resources and free GPU memory."""

    def to(self, device: torch.device) -> PointCloudUpdater:
        """Move model to *device* (optional, default no-op)."""
        return self

    # -- First-frame reconstruction --------------------------------------

    @abstractmethod
    def reconstruct_first_frame(
        self,
        frame: np.ndarray,
        geometry_poses_c2w: np.ndarray,
        dynamic_mask: np.ndarray | None,
        options: BackboneInferenceOptions,  # noqa: F821 - Name is undefined in the upstream source.
    ) -> ReconstructionResult:
        """Reconstruct initial point cloud from a single first frame.

        Args:
            frame: First video frame ``(H, W, 3)`` uint8.
            geometry_poses_c2w: Camera poses from geometry.npz ``(N, 4, 4)`` float32.
            dynamic_mask: Optional boolean mask ``(H, W)`` marking dynamic/sky pixels
                          to exclude from the point cloud.
            options: Full inference options.

        Returns:
            :class:`ReconstructionResult` with initial point cloud and metadata.
            Also updates internal state (points_world, colors, poses, intrinsics).
        """

    # -- Incremental update ----------------------------------------------

    @abstractmethod
    def update(
        self,
        iter_idx: int,
        frames: np.ndarray,
        frame_indices: list[int] | None,
        state_points: np.ndarray | None,
        state_colors: np.ndarray | None,
        options: BackboneInferenceOptions,  # noqa: F821 - Name is undefined in the upstream source.
        debug_dir: str | None = None,
        rgb_frames: np.ndarray | None = None,
        extra_entity_prompts: list[str] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process new frames and return aligned (new_points, new_colors).

        Also updates internal point cloud state.

        Args:
            iter_idx: Current iteration index.
            frames: New video frames, shape ``(N, H, W, 3)`` uint8.
                Used for depth estimation (fed to Stream3R).
            frame_indices: Optional global frame indices (len == N). Used for pose lookup.
            state_points: Existing global point cloud ``(M, 3)`` float32 or None.
            state_colors: Existing global colours ``(M, 3)`` uint8 or None.
            options: Full inference options (backend reads its own params).
            debug_dir: Optional directory to save debug visualizations
                       (2D coverage masks, confidence maps, etc.).
            rgb_frames: Optional alternative frames for point cloud RGB colours,
                shape ``(N, H, W, 3)`` uint8.  When provided, pixel colours
                are taken from these frames instead of ``frames``.  Useful for
                assigning refined/cleaned colours while keeping depth estimation
                on the original generated frames.

        Returns:
            ``(new_points, new_colors)`` in world coordinates (float32, uint8),
            or ``(None, None)`` if no valid points were produced.
        """

    # -- Scene projection rendering --------------------------------------

    @abstractmethod
    def render_scene_projection(
        self,
        target_frame_indices: list[int],
        output_size: tuple[int, int],
        vae,
        device: torch.device,
        dtype: torch.dtype,
        density_max_pixels: int | None = None,
        density_rng: np.random.Generator | None = None,
        density_blue_noise: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Render scene projections from internal point cloud and VAE-encode.

        Args:
            target_frame_indices: Frame indices to render projections for.
            output_size: ``(height, width)`` of output images.
            vae: VAE wrapper for encoding.
            device: Torch device for VAE.
            dtype: Torch dtype for VAE.
            density_max_pixels: Optional max projected pixels for density limiting.
            density_rng: RNG for density limiting.
            density_blue_noise: Blue noise texture for density limiting.

        Returns:
            VAE-encoded scene projection latent ``(T, C, H_lat, W_lat)``.
        """

    # -- Internal state access -------------------------------------------

    @property
    @abstractmethod
    def points_world(self) -> np.ndarray | None:
        """Current global point cloud ``(M, 3)`` float32."""

    @property
    @abstractmethod
    def colors(self) -> np.ndarray | None:
        """Current point cloud colors ``(M, 3)`` uint8."""

    @property
    @abstractmethod
    def poses_c2w(self) -> np.ndarray | None:
        """Camera-to-world poses ``(N, 4, 4)`` float32."""

    @property
    @abstractmethod
    def intrinsics(self) -> np.ndarray | None:
        """Camera intrinsics ``(3, 3)`` or ``(N, 3, 3)`` float32."""

    @property
    @abstractmethod
    def intrinsics_size(self) -> tuple[int, int] | None:
        """Resolution ``(H, W)`` at which intrinsics are defined."""


# ============================================================================
# Stream3R Updater
# ============================================================================

"""STream3R-based 3D reconstruction handler.

Supports:
- First-frame reconstruction via batch mode (mode="full")
- Incremental updates via streaming (StreamSession)
- Scene projection rendering from internal point cloud
"""

import gc  # noqa: E402 - Follows a module-level setup call, as upstream.

import open3d as o3d  # noqa: E402 - Follows a module-level setup call, as upstream.
import torchvision.transforms.functional as TF  # noqa: E402 - Follows a module-level setup call, as upstream.
from PIL import Image  # noqa: E402 - Follows a module-level setup call, as upstream.


def _depth_based_upsample(
    depth: np.ndarray,
    conf: np.ndarray,
    img: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Upsample Stream3R outputs via depth map interpolation and unprojection.

    Uses Stream3R's native depth prediction to upsample and unproject to
    world coordinates at the target resolution.

    Args:
        depth: (H, W) or (H, W, 1) depth map from Stream3R predictions
        conf: (H, W) confidence map
        img: (H, W, 3) color image
        intrinsics: (3, 3) camera intrinsics at processing resolution
        c2w: (4, 4) camera-to-world transform
        out_h, out_w: target output resolution

    Returns:
        (wp_up, conf_up, img_up, K_scaled) at output resolution
    """
    if depth.ndim == 3:
        # Could be (H, W, 1) or (1, H, W)
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
    depth = depth.astype(np.float32)
    proc_h, proc_w = depth.shape[:2]

    # Upsample depth, conf, img to output resolution
    depth_up = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    conf_up = cv2.resize(conf, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    img_up = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Scale intrinsics to output resolution
    K_scaled = intrinsics.copy()
    K_scaled[0, 0] *= out_w / proc_w
    K_scaled[1, 1] *= out_h / proc_h
    K_scaled[0, 2] *= out_w / proc_w
    K_scaled[1, 2] *= out_h / proc_h

    # Unproject: pixel (u, v) + depth -> camera space -> world space
    v_coords, u_coords = np.meshgrid(
        np.arange(out_h, dtype=np.float32),
        np.arange(out_w, dtype=np.float32),
        indexing="ij",
    )
    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]

    x_cam = (u_coords - cx) / fx * depth_up
    y_cam = (v_coords - cy) / fy * depth_up
    z_cam = depth_up

    pts_cam_up = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (out_h, out_w, 3)

    # Transform to world space
    R_c2w, t_c2w = c2w[:3, :3], c2w[:3, 3]
    wp_up = (R_c2w @ pts_cam_up.reshape(-1, 3).T).T + t_c2w
    wp_up = wp_up.reshape(out_h, out_w, 3).astype(np.float32)

    return wp_up, conf_up, img_up, K_scaled


# Ensure stream3r is importable, then import at module level
from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.models.stream3r import (  # noqa: E402 - Follows module-level definitions, as upstream.
    STream3R,
)
from worldfoundry.base_models.three_dimensions.point_clouds.stream3r.stream_session import (  # noqa: E402 - Follows module-level definitions, as upstream.
    StreamSession,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import (  # noqa: E402 - Follows module-level definitions, as upstream.
    pose_encoding_to_extri_intri,
)


def _create_stream_session(model, options) -> StreamSession:
    """Create a StreamSession with the configured mode and window size."""
    window_size = getattr(options, "stream3r_window_size", None)
    if window_size is not None:
        return StreamSession(model, mode="window", window_size=int(window_size))
    return StreamSession(model, mode="causal")


def _preprocess_frames(
    frames: np.ndarray,
    mode: str = "crop",
    target_size: int = 518,
) -> torch.Tensor:
    """Preprocess in-memory frames to STream3R input tensor."""
    if mode not in {"crop", "pad"}:
        raise ValueError(f"stream3r_preprocess_mode must be 'crop' or 'pad', got: {mode}")
    if frames is None or len(frames) == 0:
        raise ValueError("At least 1 frame is required for STream3R preprocessing")

    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

    images = []
    shapes = set()

    for frame in frames:
        frame_np = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)

        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        img = Image.fromarray(frame_np)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), resampling)
        img_tensor = TF.to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_tensor = torch.nn.functional.pad(
                    img_tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img_tensor.shape[1], img_tensor.shape[2]))
        images.append(img_tensor)

    if len(shapes) > 1:
        max_height = max(s[0] for s in shapes)
        max_width = max(s[1] for s in shapes)
        padded = []
        for img_tensor in images:
            h_padding = max_height - img_tensor.shape[1]
            w_padding = max_width - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_tensor = torch.nn.functional.pad(
                    img_tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded.append(img_tensor)
        images = padded

    images = torch.stack(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    return images


def _project_global_to_frame(global_pts, cam_pose, intrinsics, H, W):
    """Project global 3D points into a camera frame, return a 2D coverage mask and depth map."""
    w2c = np.linalg.inv(cam_pose)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (R @ global_pts.T).T + t

    z = pts_cam[:, 2]
    in_front = z > 0.01
    pts_cam = pts_cam[in_front]

    if len(pts_cam) == 0:
        return np.zeros((H, W), dtype=bool), np.full((H, W), np.inf, dtype=np.float32)

    pts_2d = (intrinsics @ pts_cam.T).T
    u = (pts_2d[:, 0] / pts_2d[:, 2]).astype(int)
    v = (pts_2d[:, 1] / pts_2d[:, 2]).astype(int)
    depths = pts_cam[:, 2]

    coverage = np.zeros((H, W), dtype=bool)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    coverage[v[in_bounds], u[in_bounds]] = True
    # Keep the minimum depth at each pixel (closest point)
    np.minimum.at(depth_map, (v[in_bounds], u[in_bounds]), depths[in_bounds].astype(np.float32))

    return coverage, depth_map


def _check_multiview_depth_consistency(
    new_points: np.ndarray,
    depth_history: dict,
    rel_threshold: float = 0.05,
) -> np.ndarray:
    """Check depth consistency of new 3D points against all historical depth maps.

    For each new point, project it to all previously processed camera views and
    compare the projected depth with the stored depth map. A point is considered
    consistent if its projected depth matches the historical depth within threshold
    across all views where it's visible.

    Args:
        new_points: (N, 3) world-space 3D points to check
        depth_history: dict mapping pose_idx -> (depth_map, c2w, intrinsics, (H, W))
        rel_threshold: relative depth tolerance (e.g., 0.05 = 5% difference allowed)

    Returns:
        (N,) bool mask - True if the point is consistent across all historical views
    """
    n_points = len(new_points)
    if n_points == 0:
        return np.array([], dtype=bool)

    if not depth_history:
        return np.ones(n_points, dtype=bool)

    # Initialize: all points start as consistent
    consistent_mask = np.ones(n_points, dtype=bool)
    # Track how many views each point was checked against
    view_counts = np.zeros(n_points, dtype=np.int32)

    for _pose_idx, (hist_depth, hist_c2w, hist_K, (H, W)) in depth_history.items():
        # Project new points to this historical camera
        w2c = np.linalg.inv(hist_c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        pts_cam = (R @ new_points.T).T + t

        # Filter points in front of camera
        z = pts_cam[:, 2]
        in_front = z > 0.01

        if not np.any(in_front):
            continue

        # Project to 2D
        pts_2d = (hist_K @ pts_cam[in_front].T).T
        u = pts_2d[:, 0] / pts_2d[:, 2]
        v = pts_2d[:, 1] / pts_2d[:, 2]
        projected_depth = pts_cam[in_front, 2]

        # Check bounds
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        if not np.any(in_bounds):
            continue

        # Get historical depth at projected positions
        valid_u = u_int[in_bounds]
        valid_v = v_int[in_bounds]
        valid_proj_depth = projected_depth[in_bounds]
        hist_depth_at_pts = hist_depth[valid_v, valid_u]

        # Only check where historical depth is valid (not inf).
        valid_hist = hist_depth_at_pts < np.inf
        if not np.any(valid_hist):
            continue

        # Hole-filling-friendly rule:
        # Reject only points that are significantly *in front of* historical depth.
        # Points farther than history can be valid newly revealed background.
        inconsistent_local = valid_proj_depth[valid_hist] < hist_depth_at_pts[valid_hist] * (1.0 - rel_threshold)

        # Map back to original point indices
        in_front_indices = np.where(in_front)[0]
        in_bounds_indices = in_front_indices[in_bounds]
        valid_hist_indices = in_bounds_indices[valid_hist]
        inconsistent_indices = valid_hist_indices[inconsistent_local]

        # Mark inconsistent points
        consistent_mask[inconsistent_indices] = False
        # Update view counts for all points that were checked
        view_counts[valid_hist_indices] += 1

    return consistent_mask


def _filter_points_preserve_historical_projection(
    new_points: np.ndarray,
    depth_history: dict,
    rel_threshold: float = 0.05,
) -> np.ndarray:
    """Filter new points so historical pose projections remain frozen.

    Rule against each historical pose:
    - If projected pixel has no historical depth (inf), reject new point
      (prevents old projection from becoming "more full").
    - If projected pixel has historical depth, allow only points that are
      sufficiently behind history by rel_threshold.

    This keeps old pose appearance fixed while still allowing points hidden
    behind already reconstructed surfaces.
    """
    n_points = len(new_points)
    if n_points == 0:
        return np.array([], dtype=bool)
    if not depth_history:
        return np.ones(n_points, dtype=bool)

    keep_mask = np.ones(n_points, dtype=bool)

    for _, (hist_depth, hist_c2w, hist_K, (H, W)) in depth_history.items():
        if not np.any(keep_mask):
            break

        candidate_idx = np.where(keep_mask)[0]
        pts = new_points[candidate_idx]

        w2c = np.linalg.inv(hist_c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        pts_cam = (R @ pts.T).T + t

        z = pts_cam[:, 2]
        in_front = z > 0.01
        if not np.any(in_front):
            continue

        pts_2d = (hist_K @ pts_cam[in_front].T).T
        u = pts_2d[:, 0] / pts_2d[:, 2]
        v = pts_2d[:, 1] / pts_2d[:, 2]
        projected_depth = pts_cam[in_front, 2]

        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        if not np.any(in_bounds):
            continue

        valid_u = u_int[in_bounds]
        valid_v = v_int[in_bounds]
        valid_proj_depth = projected_depth[in_bounds]
        hist_depth_at_pts = hist_depth[valid_v, valid_u]

        hist_unknown = np.isinf(hist_depth_at_pts)
        hist_known = ~hist_unknown
        # Keep only points safely behind known historical depth.
        too_close_or_in_front = np.zeros_like(hist_unknown, dtype=bool)
        if np.any(hist_known):
            too_close_or_in_front[hist_known] = valid_proj_depth[hist_known] <= hist_depth_at_pts[hist_known] * (
                1.0 + rel_threshold
            )
        reject_local = hist_unknown | too_close_or_in_front

        in_front_idx = np.where(in_front)[0]
        in_bounds_idx = in_front_idx[in_bounds]
        reject_idx = candidate_idx[in_bounds_idx[reject_local]]
        keep_mask[reject_idx] = False

    return keep_mask


def _normalize_stream3r_update_mode(mode_value: object) -> str:
    """Normalize Stream3R update mode to one of {"complete", "freeze"}."""
    mode = str(mode_value).strip().lower() if mode_value is not None else "complete"
    if mode == "freeze":
        return "freeze"
    return "complete"


def _project_points_to_depth_map(
    points: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Project 3D points to camera and create a depth map.

    Args:
        points: (N, 3) world-space 3D points
        c2w: (4, 4) camera-to-world transform
        intrinsics: (3, 3) camera intrinsics
        H, W: output depth map size

    Returns:
        (H, W) depth map (inf where no points projected)
    """
    if len(points) == 0:
        return np.full((H, W), np.inf, dtype=np.float32)

    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (R @ points.T).T + t

    # Filter points in front of camera
    z = pts_cam[:, 2]
    in_front = z > 0.01
    pts_cam = pts_cam[in_front]

    if len(pts_cam) == 0:
        return np.full((H, W), np.inf, dtype=np.float32)

    # Project to 2D
    pts_2d = (intrinsics @ pts_cam.T).T
    u = (pts_2d[:, 0] / pts_2d[:, 2]).astype(int)
    v = (pts_2d[:, 1] / pts_2d[:, 2]).astype(int)
    depths = pts_cam[:, 2]

    # Create depth map (keep minimum depth at each pixel)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    np.minimum.at(depth_map, (v[in_bounds], u[in_bounds]), depths[in_bounds].astype(np.float32))

    return depth_map


def _save_update_debug(
    debug_dir: str,
    iter_idx: int,
    frame_idx: int,
    frame_img: np.ndarray | None,
    confidence: np.ndarray,
    coverage_mask: np.ndarray | None,
    dynamic_mask: np.ndarray | None,
    H: int,
    W: int,
):
    """Save per-frame debug visualizations during point cloud update.

    Saves:
    - confidence heatmap (viridis colormap)
    - 2D coverage mask (white=covered by existing points, black=new)
    - dynamic object mask (white=dynamic, black=static)
    - dynamic mask overlaid on input frame (red tint on dynamic regions)
    - input frame for reference
    """
    sub = os.path.join(debug_dir, f"iter_{iter_idx + 1:02d}")
    os.makedirs(sub, exist_ok=True)
    prefix = f"frame_{frame_idx:03d}"

    # Save input frame
    if frame_img is not None:
        img_u8 = frame_img
        if img_u8.dtype != np.uint8:
            img_u8 = np.clip(img_u8 * 255 if img_u8.max() <= 1.0 else img_u8, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sub, f"{prefix}_input.png"), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

    # Save confidence heatmap
    conf_2d = confidence
    if conf_2d.size > 0:
        conf_norm = conf_2d - conf_2d.min()
        denom = conf_norm.max()
        if denom > 0:
            conf_norm = conf_norm / denom
        conf_vis = cv2.applyColorMap((conf_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(sub, f"{prefix}_confidence.png"), conf_vis)

    # Save 2D coverage mask
    if coverage_mask is not None:
        cov_vis = coverage_mask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(sub, f"{prefix}_coverage_mask.png"), cov_vis)

    # Save dynamic object mask
    if dynamic_mask is not None:
        dm_vis = dynamic_mask.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(sub, f"{prefix}_dynamic_mask.png"), dm_vis)

        # Overlay dynamic mask on input frame (red tint on dynamic regions)
        if frame_img is not None:
            overlay = img_u8.copy()
            img_h, img_w = overlay.shape[:2]
            # Resize dynamic mask to match frame_img resolution if needed
            if dynamic_mask.shape[:2] != (img_h, img_w):
                dm_resized = cv2.resize(
                    dynamic_mask.astype(np.uint8),
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                dm_resized = dynamic_mask
            overlay[dm_resized] = (
                overlay[dm_resized].astype(np.float32) * 0.5 + np.array([255, 0, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(sub, f"{prefix}_dynamic_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )


def _save_keep_mask_debug(
    debug_dir: str,
    iter_idx: int,
    frame_idx: int,
    conf_mask: np.ndarray,
    keep_mask: np.ndarray,
    coverage_mask: np.ndarray | None,
    dynamic_mask: np.ndarray | None,
    H: int,
    W: int,
):
    """Save the final keep mask visualization.

    Produces an RGB image (BGR for cv2):
    - Green  = kept (passes all filters)
    - Red    = removed by 2D coverage (confidence OK but already covered)
    - Blue   = removed by confidence (low confidence)
    - Yellow = removed by dynamic mask (dynamic object / sky)
    """
    sub = os.path.join(debug_dir, f"iter_{iter_idx + 1:02d}")
    os.makedirs(sub, exist_ok=True)
    prefix = f"frame_{frame_idx:03d}"

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[keep_mask] = [0, 255, 0]  # green = kept

    # Dynamic mask removal (yellow) — check before coverage so it takes priority
    if dynamic_mask is not None:
        removed_by_dynamic = conf_mask & dynamic_mask & (~keep_mask)
        vis[removed_by_dynamic] = [0, 255, 255]  # yellow (BGR) = removed by dynamic mask

    if coverage_mask is not None:
        # Removed by coverage but NOT by dynamic
        removed_by_coverage = conf_mask & (~keep_mask)
        if dynamic_mask is not None:
            removed_by_coverage = removed_by_coverage & (~dynamic_mask)
        vis[removed_by_coverage] = [0, 0, 255]  # red (BGR) = removed by coverage

    removed_by_conf = ~conf_mask
    vis[removed_by_conf] = [255, 0, 0]  # blue (BGR) = removed by confidence
    cv2.imwrite(os.path.join(sub, f"{prefix}_keep_mask.png"), vis)


class Stream3RUpdater(PointCloudUpdater):
    """3D reconstruction handler using STream3R."""

    def __init__(self) -> None:
        self._model = None
        self._session = None
        self._frames_fed: int = 0
        self._device: torch.device | None = None

        # Internal state
        self._points_world: np.ndarray | None = None
        self._colors: np.ndarray | None = None
        self._poses_c2w: np.ndarray | None = None
        self._intrinsics: np.ndarray | None = None
        self._intrinsics_size: tuple[int, int] | None = None

        # Multi-view depth consistency: store depth maps for each processed pose
        # Key: pose_idx, Value: (depth_map, c2w, intrinsics, (H, W))
        self._depth_history: dict = {}
        self._consistency_threshold: float = 0.05  # relative depth tolerance

        # Dynamic object detection models (lazy-loaded)
        self._sam3_segmenter = None
        self._qwen_extractor = None
        self._sam3_model_path: str | None = None
        self._qwen_model_path: str | None = None

        # Last dynamic masks from update() — exposed for callers to read.
        # List of (H, W) bool masks (one per frame), or None.
        self.last_dynamic_masks: list[np.ndarray] | None = None

        # Pre-filtering per-frame world points from update() — includes dynamic pixels.
        # List of (H, W, 3) float32 arrays (one per frame), or None.
        # Used by intermediate event detection for fg back-projection.
        self.last_per_frame_world_points: list[np.ndarray] | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def points_world(self) -> np.ndarray | None:
        return self._points_world

    @property
    def colors(self) -> np.ndarray | None:
        return self._colors

    @property
    def poses_c2w(self) -> np.ndarray | None:
        return self._poses_c2w

    @property
    def intrinsics(self) -> np.ndarray | None:
        return self._intrinsics

    @property
    def intrinsics_size(self) -> tuple[int, int] | None:
        return self._intrinsics_size

    def set_dynamic_models(
        self,
        qwen_model_path: str,
        sam3_model_path: str,
        qwen_extractor=None,
        sam3_segmenter=None,
        cpu_offload_qwen: bool = False,
        cpu_offload_sam3: bool = False,
        scene_detect_prompt: str = "",
    ) -> None:
        """Store model paths / instances for dynamic object detection.

        If pre-loaded model instances are provided they are reused directly,
        avoiding a redundant second copy on GPU.
        """
        self._qwen_model_path = qwen_model_path
        self._sam3_model_path = sam3_model_path
        self._cpu_offload_qwen = cpu_offload_qwen
        self._cpu_offload_sam3 = cpu_offload_sam3
        self._scene_detect_prompt = scene_detect_prompt
        if qwen_extractor is not None:
            self._qwen_extractor = qwen_extractor
        if sam3_segmenter is not None:
            self._sam3_segmenter = sam3_segmenter

    def _detect_dynamic_masks(
        self,
        frames: np.ndarray,
        options,
        extra_entity_prompts: list[str] | None = None,
    ) -> list[np.ndarray] | None:
        """Detect dynamic objects in generated frames using Qwen + SAM3.

        Uses the shared Qwen/SAM3 instances set via ``set_dynamic_models()``.
        CPU offload is applied when the corresponding flags were passed.

        Args:
            frames: (T, H, W, 3) uint8 RGB frames
            options: inference options (must have stream3r_use_dynamic_mask_in_update)
            extra_entity_prompts: Additional entity names (e.g. from previously
                active events) to always include in SAM3 segmentation, even if
                Qwen does not re-detect them in the current frames.

        Returns:
            List of (H, W) bool masks per frame, or None if no dynamic objects.
        """

        if not getattr(options, "stream3r_use_dynamic_mask_in_update", False):
            return None
        if self._qwen_extractor is None or self._sam3_segmenter is None:
            return None

        device_str = str(self._device) if self._device else "cuda:0"
        cpu_offload_qwen = getattr(self, "_cpu_offload_qwen", False)
        _cpu_offload_sam3 = getattr(self, "_cpu_offload_sam3", False)

        # Step 1: Save frames as temp video for Qwen video-mode detection
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115 - Keeps the upstream file-handle lifetime.
        tmp_video_path = tmp_video.name
        tmp_video.close()
        try:
            h, w = frames.shape[1], frames.shape[2]
            writer = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()

            # CPU offload: move Qwen to GPU before inference, back after.
            if cpu_offload_qwen and hasattr(self._qwen_extractor, "model"):
                self._qwen_extractor.model.to(device_str)
            try:
                detect_prompt = getattr(self, "_scene_detect_prompt", "")
                dynamic_prompts, raw_text = self._qwen_extractor.extract(
                    tmp_video_path,
                    prompt=detect_prompt,
                )
            finally:
                if cpu_offload_qwen and hasattr(self._qwen_extractor, "model"):
                    self._qwen_extractor.model.to("cpu")
                    torch.cuda.empty_cache()
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

        all_prompts = list(dynamic_prompts) if dynamic_prompts else []
        # Merge in previously known entity prompts so that foreground from
        # earlier events is still masked even when Qwen fails to re-detect it.
        if extra_entity_prompts:
            existing = {p.lower().strip() for p in all_prompts}
            for ep in extra_entity_prompts:
                if ep.lower().strip() not in existing:
                    all_prompts.append(ep)
                    existing.add(ep.lower().strip())
        all_prompts.append("sky")

        # Step 2: Run SAM3 per-prompt to get individual masks, then merge.
        pil_frames = [Image.fromarray(f) for f in frames]
        n_frames = len(pil_frames)

        merged: np.ndarray | None = None  # (T, H, W) bool

        for prompt in all_prompts:
            seg_result = self._sam3_segmenter.segment(
                video_path=pil_frames,
                prompts=[prompt],
                frame_index=0,
                expected_frames=n_frames,
            )
            if seg_result.size == 0:
                continue
            merged = seg_result.copy() if merged is None else merged | seg_result

        if merged is None or not merged.any():
            return None

        # Dilate dynamic mask to aggressively remove foreground edges from
        # background point cloud registration (prefer over-removal to leakage).
        dilate_kernel = np.ones((3, 3), dtype=np.uint8)
        dilate_iters = 5
        dilated = np.empty_like(merged)
        for t in range(merged.shape[0]):
            m = merged[t].astype(np.uint8)
            m = cv2.dilate(m, dilate_kernel, iterations=dilate_iters)
            dilated[t] = m.astype(bool)

        masks = [dilated[t] for t in range(dilated.shape[0])]
        return masks

    @property
    def model(self):
        """Expose the loaded STream3R model (or None if not yet loaded)."""
        return self._model

    @property
    def session(self):
        """Expose the StreamSession (or None if not yet created)."""
        return self._session

    @property
    def frames_fed(self) -> int:
        """Number of frames fed to the session so far."""
        return self._frames_fed

    @frames_fed.setter
    def frames_fed(self, value: int) -> None:
        self._frames_fed = value

    # -- Lifecycle -----------------------------------------------------------

    def init_model(self, device: torch.device) -> None:
        self._device = device

    def to(self, device: torch.device) -> Stream3RUpdater:
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def offload_to_cpu(self) -> None:
        """Move model to CPU to free GPU memory. _load_model will restore it."""
        if self._model is not None:
            self._model.to("cpu")
        torch.cuda.empty_cache()

    def cleanup(self) -> None:
        if self._session is not None:
            self._session.clear()
            self._session = None
        self._model = None
        self._frames_fed = 0
        self._points_world = None
        self._colors = None
        self._poses_c2w = None
        self._intrinsics = None
        self._intrinsics_size = None
        self._qwen_extractor = None
        self._sam3_segmenter = None
        self._depth_history = {}
        gc.collect()
        torch.cuda.empty_cache()

    def reset_session(self) -> None:
        """Clear session and accumulated state but keep model in GPU memory.

        Used by batch inference to reuse the loaded Stream3R model across
        multiple configs without paying the model-load cost each time.
        """
        if self._session is not None:
            self._session.clear()
            self._session = None
        self._frames_fed = 0
        self._points_world = None
        self._colors = None
        self._poses_c2w = None
        self._intrinsics = None
        self._intrinsics_size = None
        self._depth_history = {}
        self.last_dynamic_masks = None

    # -- First-frame reconstruction ------------------------------------------

    def reconstruct_first_frame(
        self,
        frame: np.ndarray,
        geometry_poses_c2w: np.ndarray,
        dynamic_mask: np.ndarray | None,
        options,
    ) -> ReconstructionResult:
        device = self._device

        model = self._load_model(options.stream3r_model_path, device)

        # Feed first frame to session so incremental updates have context
        if self._session is None:
            self._session = _create_stream_session(model, options)

        # Preprocess single frame
        images = _preprocess_frames(frame[np.newaxis, ...], mode=options.stream3r_preprocess_mode).to(device=device)

        # Batch inference (no session needed for single frame)
        with torch.no_grad():
            predictions = self._session.forward_stream(images[0:1])

        # with torch.no_grad():
        #     predictions = self._session(images, mode="full")         # model inference

        # JT: get c2w and K from STream3R
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        # JT: STream3R and input geometry_poses_c2w alignment
        # extrinsic is w2c [R|t] (B, S, 3, 4); convert first frame to c2w 4x4
        stream3r_w2c_34 = extrinsic[0, 0].cpu().numpy()  # (3, 4)
        stream3r_w2c = np.eye(4, dtype=np.float32)
        stream3r_w2c[:3, :] = stream3r_w2c_34
        stream3r_c2w = np.linalg.inv(stream3r_w2c)

        # Compute T_align: Stream3R world coords -> geometry world coords
        geo_c2w_0 = geometry_poses_c2w[0].astype(np.float32)
        if geo_c2w_0.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float32)
            tmp[:3, :] = geo_c2w_0
            geo_c2w_0 = tmp
        self._alignment_transform = stream3r_c2w @ np.linalg.inv(geo_c2w_0)

        # Use Stream3R predicted intrinsics instead of heuristic
        self._stream3r_intrinsics = intrinsic[0, 0].cpu().numpy().astype(np.float32)
        self._stream3r_intrinsics_proc = self._stream3r_intrinsics.copy()  # at proc resolution

        preds = dict(predictions)
        for key in preds:
            if isinstance(preds[key], torch.Tensor):
                preds[key] = preds[key].cpu().numpy().squeeze(0)

        wp = preds.get("world_points")
        if wp is None:
            raise RuntimeError("[Stream3R] No world_points in predictions")

        conf = preds.get("world_points_conf", np.ones(wp.shape[:-1]))
        img = preds.get("images")
        if img is None:
            raise RuntimeError("[Stream3R] No images in predictions")

        # Handle shapes: world_points may be (1, H, W, 3) or (H, W, 3)
        if wp.ndim == 4:
            wp = wp[0]
        if conf.ndim == 3:
            conf = conf[0]
        if img.ndim == 4 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 4 and img.shape[1] == 3:
            img = np.transpose(img[0], (1, 2, 0))
        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Compute camera-space depth from world_points at proc resolution
        proc_h, proc_w = wp.shape[:2]
        w2c = np.linalg.inv(stream3r_c2w)
        pts_cam = (w2c[:3, :3] @ wp.reshape(-1, 3).T).T + w2c[:3, 3]
        depth = pts_cam[:, 2].reshape(proc_h, proc_w).astype(np.float32)

        # Upsample to target resolution: resize depth, resize first frame, then unproject
        out_h, out_w = options.target_hw
        # Resize first frame to target resolution (use original frame, not Stream3R's processed image)
        frame_img = frame.astype(np.float32)
        if frame_img.max() > 1.5:
            frame_img = frame_img / 255.0
        frame_img = cv2.resize(frame_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        # Resize depth & conf from proc resolution, unproject at target resolution
        depth_up = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        conf = cv2.resize(conf.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        # Scale intrinsics from proc resolution to target resolution
        K_scaled = self._stream3r_intrinsics.copy()
        K_scaled[0, 0] *= out_w / proc_w
        K_scaled[1, 1] *= out_h / proc_h
        K_scaled[0, 2] *= out_w / proc_w
        K_scaled[1, 2] *= out_h / proc_h
        # Unproject at target resolution
        v_coords, u_coords = np.meshgrid(
            np.arange(out_h, dtype=np.float32),
            np.arange(out_w, dtype=np.float32),
            indexing="ij",
        )
        fx, fy = K_scaled[0, 0], K_scaled[1, 1]
        cx, cy = K_scaled[0, 2], K_scaled[1, 2]
        x_cam = (u_coords - cx) / fx * depth_up
        y_cam = (v_coords - cy) / fy * depth_up
        z_cam = depth_up
        pts_cam_up = np.stack([x_cam, y_cam, z_cam], axis=-1)
        R_c2w, t_c2w = stream3r_c2w[:3, :3], stream3r_c2w[:3, 3]
        wp = (R_c2w @ pts_cam_up.reshape(-1, 3).T).T + t_c2w
        wp = wp.reshape(out_h, out_w, 3).astype(np.float32)
        img = frame_img
        self._stream3r_intrinsics = K_scaled

        H, W = wp.shape[:2]

        # Apply confidence threshold
        conf_flat = conf.reshape(-1)
        valid_conf = conf_flat[conf_flat > 0]
        if len(valid_conf) > 0:
            conf_thresh = max(np.percentile(valid_conf, options.stream3r_keep_conf_percentile), 1e-5)
        else:
            conf_thresh = 1e-5
        valid = conf >= conf_thresh

        # Apply dynamic mask
        if dynamic_mask is not None:
            if dynamic_mask.shape != (H, W):
                dynamic_mask_resized = cv2.resize(
                    dynamic_mask.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                dynamic_mask_resized = dynamic_mask
            valid = valid & (~dynamic_mask_resized)

        pts = wp[valid].astype(np.float32)
        cols = img[valid]
        cols = (cols * 255.0).clip(0, 255).astype(np.uint8) if cols.max() <= 1.5 else cols.clip(0, 255).astype(np.uint8)

        # Voxel downsample
        if options.voxel_size and options.voxel_size > 0 and len(pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(voxel_size=options.voxel_size)
            pts = np.asarray(pcd.points).astype(np.float32)
            cols = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

        # Store internal state
        self._points_world = pts
        self._colors = cols
        # Transform geometry poses into Stream3R world coordinate system
        aligned_poses = np.array(
            [
                self._alignment_transform @ p.astype(np.float32)
                if p.shape == (4, 4)
                else self._alignment_transform @ np.vstack([p.astype(np.float32), [0, 0, 0, 1]])
                for p in geometry_poses_c2w
            ],
            dtype=np.float32,
        )
        self._poses_c2w = aligned_poses
        self._intrinsics_size = (H, W)

        # Use Stream3R predicted intrinsics (scaled to processed resolution H, W)
        self._intrinsics = self._stream3r_intrinsics

        # Save first-frame depth history using only retained static points.
        # This keeps dynamic/filtered-out pixels as unknown (inf).
        first_hist_depth = _project_points_to_depth_map(pts, stream3r_c2w, K_scaled, out_h, out_w)
        self._depth_history[0] = (first_hist_depth, stream3r_c2w.copy(), K_scaled.copy(), (out_h, out_w))
        self._frames_fed = 1

        return ReconstructionResult(
            points_world=pts,
            colors=cols,
            intrinsics=self._intrinsics,
            intrinsics_size=(H, W),
            alignment_transform=self._alignment_transform,
        )

    # -- Scene projection rendering (project to scene poses in this inference round) ------------------------------------------

    def render_scene_projection(
        self,
        target_frame_indices: list[int],
        output_size: tuple[int, int],
        vae,
        device: torch.device,
        dtype: torch.dtype,
        density_max_pixels: int | None = None,
        density_rng: np.random.Generator | None = None,
        density_blue_noise: np.ndarray | None = None,
    ) -> torch.Tensor:
        if self._points_world is None or self._poses_c2w is None:
            raise RuntimeError("No point cloud available for rendering")

        height, width = output_size
        proc_h, proc_w = self._intrinsics_size
        device_str = str(device) if not isinstance(device, str) else device
        use_density = density_max_pixels is not None and density_max_pixels > 0

        projections_list = []
        for frame_idx in target_frame_indices:
            pose_idx = _safe_frame_index(frame_idx, len(self._poses_c2w))
            K_scaled = scale_intrinsics(
                self._intrinsics,
                scale_x=width / proc_w,
                scale_y=height / proc_h,
            )

            if use_density:
                pts, cols = _limit_points_by_density(
                    self._points_world,
                    self._colors,
                    self._poses_c2w[pose_idx],
                    K_scaled,
                    (height, width),
                    density_max_pixels,
                    rng=density_rng,
                    blue_noise=density_blue_noise,
                )
            else:
                pts, cols = self._points_world, self._colors

            proj = render_projection(
                points_world=pts,
                K=K_scaled,
                c2w=self._poses_c2w[pose_idx],
                image_size=(height, width),
                channels=["rgb"],
                colors=cols,
                fill_holes_kernel=0,
                device=device_str,
            )
            projections_list.append(proj)

        projections = np.stack(projections_list, axis=0)
        projections = projections.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        proj_tensor = torch.from_numpy(projections).float() / 127.5 - 1.0

        with torch.no_grad():
            vae_device = next(vae.model.parameters()).device
            if vae_device != device:
                vae.model.to(device)
                vae.mean = vae.mean.to(device)
                vae.std = vae.std.to(device)
            proj_tensor = proj_tensor.to(device=device, dtype=dtype)
            proj_tensor = proj_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, T, H, W)
            scene_proj_latent = vae.encode_to_latent(proj_tensor)
            scene_proj_latent = scene_proj_latent.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)

        return scene_proj_latent

    # -- Incremental update --------------------------------------------------

    def update(
        self,
        iter_idx: int,
        frames: np.ndarray,
        frame_indices: list[int] | None,
        state_points: np.ndarray | None,
        state_colors: np.ndarray | None,
        options,
        debug_dir: str | None = None,
        rgb_frames: np.ndarray | None = None,
        extra_entity_prompts: list[str] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if frames.size == 0:
            return None, None

        self.last_per_frame_world_points = []

        frame_indices_used = None
        if frame_indices is not None:
            frame_indices_used = list(frame_indices)
            if len(frame_indices_used) != len(frames):
                min_len = min(len(frame_indices_used), len(frames))
                frames = frames[:min_len]
                frame_indices_used = frame_indices_used[:min_len]
        else:
            frame_indices_used = list(range(len(frames)))

        device = self._device
        model = self._load_model(options.stream3r_model_path, device)

        # rgb_frames: alternative source for point cloud colours (e.g. refined)
        color_frames = rgb_frames if rgb_frames is not None else frames

        # Detect dynamic objects before Stream3R processing.
        dynamic_masks_sampled = self._detect_dynamic_masks(
            frames,
            options,
            extra_entity_prompts=extra_entity_prompts,
        )
        self.last_dynamic_masks = dynamic_masks_sampled

        images = _preprocess_frames(frames, mode=options.stream3r_preprocess_mode).to(device=device)

        if self._session is None:
            self._session = _create_stream_session(model, options)
        session = self._session

        # Incremental merge: start from existing global point cloud
        running_points = state_points.copy() if state_points is not None else None
        running_colors = state_colors.copy() if state_colors is not None else None
        total_new_pts = 0
        update_mode = _normalize_stream3r_update_mode(getattr(options, "stream3r_update_mode", "complete"))
        freeze_old_projection = update_mode == "freeze"

        with torch.no_grad():
            for i in range(images.shape[0]):
                image = images[i : i + 1]
                self._frames_fed += 1
                total_fed = self._frames_fed

                predictions = session.forward_stream(image)
                preds = dict(predictions)
                for key in preds:
                    if isinstance(preds[key], torch.Tensor):
                        preds[key] = preds[key].cpu().numpy().squeeze(0)

                cur_preds = {}
                for key in preds:
                    if isinstance(preds[key], np.ndarray) and preds[key].ndim >= 1 and preds[key].shape[0] == total_fed:
                        cur_preds[key] = preds[key][-1:]
                    else:
                        cur_preds[key] = preds[key]

                wp = cur_preds.get("world_points")
                if wp is None:
                    continue
                conf = cur_preds.get("world_points_conf", np.ones(wp.shape[:-1]))
                img = cur_preds.get("images")
                if img is None:
                    continue
                if img.ndim == 4 and img.shape[1] == 3:
                    img = np.transpose(img, (0, 2, 3, 1))

                # Compute camera-space depth from world_points at proc resolution
                proc_h, proc_w = wp[0].shape[:2]  # 294, 518   wp.shape
                out_h, out_w = options.target_hw  # 720, 1280
                global_frame_idx = frame_indices_used[i] if frame_indices_used is not None else (total_fed - 1)
                pose_idx = min(global_frame_idx, len(self._poses_c2w) - 1)  # 1 here
                c2w = self._poses_c2w[pose_idx]
                w2c = np.linalg.inv(c2w)
                pts_cam = (w2c[:3, :3] @ wp[0].reshape(-1, 3).T).T + w2c[:3, 3]
                depth_frame = pts_cam[:, 2].reshape(proc_h, proc_w).astype(np.float32)  # depth for pose1

                # Resize frame image to target resolution
                frame_img = color_frames[i].astype(np.float32)
                if frame_img.max() > 1.5:
                    frame_img = frame_img / 255.0
                frame_img = cv2.resize(frame_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)  # 720, 1280

                # Upsample depth & conf, unproject at target resolution
                # Use proc-resolution intrinsics; _depth_based_upsample scales them to out_h/out_w
                wp_frame, conf_frame, img_frame, _ = _depth_based_upsample(
                    depth_frame, conf[0], frame_img, self._stream3r_intrinsics_proc, c2w, out_h, out_w
                )

                H, W = wp_frame.shape[:2]

                # Store pre-filtering world points (includes dynamic pixels)
                # for intermediate event fg back-projection.
                self.last_per_frame_world_points.append(wp_frame.copy())

                pts = wp_frame.reshape(-1, 3)
                cols = img_frame.reshape(-1, 3)
                c = conf_frame.reshape(-1)
                if c.size == 0:
                    continue

                # Scale intrinsics to output resolution
                K_out = self._stream3r_intrinsics_proc.copy()
                K_out[0, 0] *= out_w / proc_w
                K_out[1, 1] *= out_h / proc_h
                K_out[0, 2] *= out_w / proc_w
                K_out[1, 2] *= out_h / proc_h

                # 2D Coverage Check — use incrementally updated running_points
                use_2d_cov = (
                    running_points is not None
                    and len(running_points) > 0
                    and self._poses_c2w is not None
                    and self._intrinsics is not None
                )
                if use_2d_cov:
                    coverage, _ = _project_global_to_frame(
                        running_points, self._poses_c2w[pose_idx], self._intrinsics, H, W
                    )
                    keep_mask_2d = (~coverage).reshape(-1)
                else:
                    coverage = None
                    keep_mask_2d = None

                # Per-frame dynamic object mask
                dynamic_mask_2d = None
                dynamic_mask_flat = None
                if dynamic_masks_sampled is not None and i < len(dynamic_masks_sampled):
                    dm = dynamic_masks_sampled[i]
                    if dm.shape != (H, W):
                        dm = cv2.resize(
                            dm.astype(np.uint8),
                            (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    dynamic_mask_2d = dm
                    dynamic_mask_flat = dm.reshape(-1)

                # Save debug visualizations
                if debug_dir is not None:
                    _save_update_debug(
                        debug_dir=debug_dir,
                        iter_idx=iter_idx,
                        frame_idx=i,
                        frame_img=img[0] if img is not None else None,
                        confidence=c.reshape(H, W),
                        coverage_mask=coverage,
                        dynamic_mask=dynamic_mask_2d,
                        H=H,
                        W=W,
                    )

                # Confidence filter
                keep_thresh = max(np.percentile(c, options.stream3r_keep_conf_percentile), 1e-5)
                keep_mask = c >= keep_thresh
                conf_only_mask = keep_mask.copy()
                if keep_mask_2d is not None:
                    keep_mask = keep_mask & keep_mask_2d
                if dynamic_mask_flat is not None:
                    keep_mask = keep_mask & (~dynamic_mask_flat)

                # Save final keep mask debug
                if debug_dir is not None:
                    _save_keep_mask_debug(
                        debug_dir=debug_dir,
                        iter_idx=iter_idx,
                        frame_idx=i,
                        conf_mask=conf_only_mask.reshape(H, W),
                        keep_mask=keep_mask.reshape(H, W),
                        coverage_mask=coverage,
                        dynamic_mask=dynamic_mask_2d,
                        H=H,
                        W=W,
                    )

                if not np.any(keep_mask):
                    # No trusted static points for this view: mark as unknown depth.
                    if (not freeze_old_projection) or (pose_idx not in self._depth_history):
                        self._depth_history[pose_idx] = (
                            np.full((out_h, out_w), np.inf, dtype=np.float32),
                            c2w.copy(),
                            K_out.copy(),
                            (out_h, out_w),
                        )
                    continue

                # Extract surviving points and immediately merge into running point cloud
                frame_pts = pts[keep_mask].astype(np.float32)
                frame_cols = cols[keep_mask]
                if frame_cols.max() <= 1.5:
                    frame_cols = (frame_cols * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frame_cols = frame_cols.clip(0, 255).astype(np.uint8)

                # Multi-view depth consistency check: verify new points against all historical depth maps
                if (not freeze_old_projection) and len(self._depth_history) > 0:
                    consistency_thresh = getattr(options, "stream3r_consistency_threshold", 0.05)
                    mv_consistent = _check_multiview_depth_consistency(
                        frame_pts,
                        self._depth_history,
                        rel_threshold=consistency_thresh,
                    )
                    frame_pts = frame_pts[mv_consistent]
                    frame_cols = frame_cols[mv_consistent]

                # Optional strict freeze: do not let later points change historical
                # pose projections (no hole-filling and no front-surface replacement).
                if freeze_old_projection and len(self._depth_history) > 0 and len(frame_pts) > 0:
                    freeze_thresh = getattr(options, "stream3r_consistency_threshold", 0.05)
                    preserve_mask = _filter_points_preserve_historical_projection(
                        frame_pts,
                        self._depth_history,
                        rel_threshold=freeze_thresh,
                    )
                    frame_pts = frame_pts[preserve_mask]
                    frame_cols = frame_cols[preserve_mask]

                # Save current frame depth history from retained static points only.
                filtered_hist_depth = _project_points_to_depth_map(frame_pts, c2w, K_out, out_h, out_w)
                if (not freeze_old_projection) or (pose_idx not in self._depth_history):
                    self._depth_history[pose_idx] = (filtered_hist_depth, c2w.copy(), K_out.copy(), (out_h, out_w))

                if len(frame_pts) == 0:
                    continue

                if running_points is None or len(running_points) == 0:
                    running_points = frame_pts
                    running_colors = frame_cols
                else:
                    running_points = np.concatenate([running_points, frame_pts], axis=0)
                    running_colors = np.concatenate([running_colors, frame_cols], axis=0)

                total_new_pts += len(frame_pts)

        # Outlier removal on newly added points only

        if running_points is None or len(running_points) == 0:
            return None, None

        n_state = len(state_points) if state_points is not None else 0
        n_new = len(running_points) - n_state

        if n_new > 0 and n_state > 0:
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(running_points[n_state:])
            new_pcd.colors = o3d.utility.Vector3dVector(running_colors[n_state:].astype(np.float64) / 255.0)

            if len(new_pcd.points) > 20:
                new_pcd, _ = new_pcd.remove_statistical_outlier(
                    nb_neighbors=int(options.stream3r_outlier_nb_neighbors),
                    std_ratio=float(options.stream3r_outlier_std_ratio),
                )

            new_pts_clean = np.asarray(new_pcd.points).astype(np.float32)
            new_cols_clean = (np.asarray(new_pcd.colors) * 255).clip(0, 255).astype(np.uint8)
            running_points = np.concatenate([running_points[:n_state], new_pts_clean], axis=0)
            running_colors = np.concatenate([running_colors[:n_state], new_cols_clean], axis=0)

        new_points = running_points.astype(np.float32)
        new_colors = running_colors

        # Sync internal state
        self._points_world = new_points
        self._colors = new_colors

        return new_points, new_colors

    # -- Internal helpers ----------------------------------------------------

    def _load_model(self, model_path: str, device: torch.device):
        if self._model is None:
            self._model = STream3R.from_pretrained(model_path).to(device=device)
            self._model.eval()
        else:
            self._model = self._model.to(device=device)
        return self._model


def create_pointcloud_updater(backend: str, device) -> PointCloudUpdater:
    """Factory: instantiate a point cloud updater by backend name."""
    backend = backend.strip().lower().replace("-", "_")
    if backend == "stream3r":
        updater = Stream3RUpdater()
    else:
        raise ValueError(f"Unknown pointcloud_backend: '{backend}'. Choose from: 'stream3r'.")
    updater.init_model(device)
    return updater
