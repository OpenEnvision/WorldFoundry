"""
Rendering 4D Control Maps for Dynamic Video Generation

This script generates 4D control maps (RGB, depth, masks) by rendering 3D Gaussians
and background point clouds from different camera poses. Supports multi-frame rendering
with background-foreground compositing for video generation.

Key features:
- Render ellipsoid meshes (3D Gaussian representations) using PyTorch3D
- Composite background point clouds with foreground 3D Gaussians
- Generate control maps: RGB, depth, masks, and visualizations
- Support for camera trajectories and Gaussian parameter sequences
"""

import logging
import argparse
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from time import time

import torch
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
from tqdm import tqdm
from kornia.geometry.depth import depth_to_3d_v2

from worldfoundry.base_models.three_dimensions.three_d_four_d.pytorch3d_compat import (
    configure_pytorch3d_extension,
)

configure_pytorch3d_extension()

from pytorch3d.structures import Pointclouds, Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    TexturesVertex,
    PointLights,
)
from pytorch3d.utils import ico_sphere
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Coordinate system transform matrix
# ============================================================================
# OpenCV coordinate system: X=right, Y=down, Z=forward
# Blender world coordinate system (Z-up): X=right, Y=forward, Z=up
# Transform: Blender_X = OpenCV_X, Blender_Y = OpenCV_Z, Blender_Z = -OpenCV_Y
COORD_TRANSFORM_CV2BLENDER = np.array([
    [1,  0,  0],  # Blender X = OpenCV X (right)
    [0,  0,  1],  # Blender Y = OpenCV Z (forward)
    [0, -1,  0],  # Blender Z = -OpenCV Y (up; OpenCV Y points down)
], dtype=np.float32)


def make_ellipsoid_mesh(mean: torch.Tensor, cov: torch.Tensor, scale_factor: float = 2.0, subdivisions: int = 3,
                        color_rgb255: Optional[torch.Tensor] = None, device: str = 'cuda') -> Meshes:
    """Create an ellipsoid mesh from a 3D Gaussian (mean and covariance).

    Uses eigendecomposition of the covariance matrix to determine ellipsoid shape,
    then scales and translates a unit icosphere to match the Gaussian.

    Args:
        mean: 3D Gaussian mean (3,)
        cov: 3D covariance matrix (3, 3)
        scale_factor: Scale multiplier for the ellipsoid axes
        subdivisions: Number of subdivisions for icosphere (higher = more vertices)
        color_rgb255: Color in RGB [0, 255] format. If None, uses default red
        device: Computation device

    Returns:
        PyTorch3D Meshes object representing the ellipsoid
    """
    device = mean.device if isinstance(mean, torch.Tensor) else torch.device(device)
    sphere = ico_sphere(subdivisions, device=device)  # unit sphere at origin
    verts = sphere.verts_list()[0]  # (V,3)
    faces = sphere.faces_list()[0]  # (F,3)

    if isinstance(mean, np.ndarray):
        mean_t = torch.from_numpy(mean).float().to(device)
    else:
        mean_t = mean.to(device).float()
    if isinstance(cov, np.ndarray):
        cov_t = torch.from_numpy(cov).float().to(device)
    else:
        cov_t = cov.to(device).float()

    # Eigendecomposition of covariance: cov = U @ diag(evals) @ U^T
    evals, evecs = torch.linalg.eigh(cov_t)
    evals = torch.clamp(evals, min=1e-8)
    axes = scale_factor * torch.sqrt(evals)  # Scaling factors along principal axes
    # Transform: x = mean + R * diag(axes) * u
    M = evecs @ torch.diag(axes)  # (3,3)
    verts_world = verts @ M.T + mean_t  # (V,3)

    # Assign colors to vertices
    if color_rgb255 is None:
        color_rgb255 = torch.tensor([200, 60, 60], dtype=torch.uint8, device=device)
    colors = (color_rgb255.float() / 255.0).expand_as(verts_world)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))

    return Meshes(verts=[verts_world], faces=[faces], textures=textures)

def combine_meshes_for_scene(mesh_list: List[Meshes]) -> Optional[Meshes]:
    """Combine multiple meshes into a single batch for rendering.

    Merges vertices, faces, and colors from all meshes with proper face indexing.

    Args:
        mesh_list: List of PyTorch3D Meshes

    Returns:
        Combined Meshes object or None if list is empty
    """
    if len(mesh_list) == 0:
        return None
    device = mesh_list[0].verts_list()[0].device
    verts_all = []
    faces_all = []
    colors_all = []
    v_ofs = 0
    for m in mesh_list:
        v = m.verts_list()[0]
        f = m.faces_list()[0]
        verts_all.append(v)
        faces_all.append(f + v_ofs)  # Offset face indices
        v_ofs += v.shape[0]

        if isinstance(m.textures, TexturesVertex):
            col = m.textures.verts_features_list()[0]
        else:
            col = torch.ones_like(v) * 0.7
        colors_all.append(col)
    verts_cat = torch.cat(verts_all, dim=0)
    faces_cat = torch.cat(faces_all, dim=0)
    colors_cat = torch.cat(colors_all, dim=0)
    textures = TexturesVertex(verts_features=colors_cat.unsqueeze(0))
    return Meshes(verts=[verts_cat], faces=[faces_cat], textures=textures).to(device)

def render_meshes_pytorch3d_batch(
    meshes_list: List[Optional[Meshes]],
    Ks: torch.Tensor,
    Ts: torch.Tensor,
    image_size: Tuple[int, int],
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_fp16: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render multiple meshes in batch using PyTorch3D.

    Args:
        meshes_list: List of Meshes to render (None entries are skipped)
        Ks: Camera intrinsic matrices (B, 3, 3)
        Ts: Camera extrinsic matrices (B, 4, 4)
        image_size: (height, width) of output images
        background_color: RGB background color (0-1 range)
        use_fp16: Retained for CLI compatibility. PyTorch3D rasterization uses
            float32 because its CUDA kernels do not accept float16 geometry.

    Returns:
        Tuple of (rgb_batch, depth_batch, mask_batch)
    """
    H, W = image_size
    device = Ks.device
    B = len(meshes_list)

    rgb_batch = torch.full((B, H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device)
    depth_batch = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    mask_batch = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    # Filter out empty meshes
    valid_indices = [i for i, m in enumerate(meshes_list) if m is not None and m.num_verts_per_mesh().sum().item() > 0]

    if len(valid_indices) == 0:
        return rgb_batch, depth_batch, mask_batch

    valid_meshes = [meshes_list[i] for i in valid_indices]
    valid_Ks = Ks[valid_indices]
    valid_Ts = Ts[valid_indices]

    # PyTorch3D 0.7.x CUDA rasterizers require float32 vertices and cameras.
    # Autocasting this block produces ``expected scalar type Float but found
    # Half`` inside ``rasterize_meshes``.
    compute_dtype = torch.float32

    cameras = _build_cam_from_extrinsics(
        valid_Ks.to(compute_dtype), valid_Ts.to(compute_dtype), image_size
    )

    merged_meshes = join_meshes_as_batch(valid_meshes)

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None
    )

    lights = PointLights(location=[[0.0, 0.0, 0.0]], device=device)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    with torch.no_grad():
        rendered = renderer(merged_meshes)  # (B_valid, H, W, 4)

        rgb_valid = (torch.clamp(rendered[..., :3].float(), 0, 1) * 255).to(torch.uint8)

        # Extract depth and mask from rasterizer output
        fragments = rasterizer(merged_meshes)
        pix_to_face = fragments.pix_to_face[..., 0]  # (B_valid, H, W)
        mask_valid = pix_to_face != -1
        depth_valid = fragments.zbuf[..., 0].float().clone()  # (B_valid, H, W)
        depth_valid[~mask_valid] = 0.0

    # Fill background color for non-rendered pixels
    bg_color_uint8 = torch.tensor(
        [int(c * 255) for c in background_color],
        dtype=torch.uint8, device=device
    )
    rgb_valid[~mask_valid] = bg_color_uint8

    # Fill output batch
    for i, idx in enumerate(valid_indices):
        rgb_batch[idx] = rgb_valid[i]
        depth_batch[idx] = depth_valid[i]
        mask_batch[idx] = mask_valid[i]

    return rgb_batch, depth_batch, mask_batch

def render_point_cloud_pytorch3d_batch(
    points_3d: torch.Tensor,
    colors: torch.Tensor,
    Ks: torch.Tensor,
    Ts: torch.Tensor,
    image_size: Tuple[int, int],
    point_size: float = 0.01,
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    use_fp16: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render point cloud in batch using PyTorch3D.

    Args:
        points_3d: 3D points (N, 3)
        colors: RGB colors for points (N, 3) in [0, 255] range
        Ks: Camera intrinsic matrices (B, 3, 3)
        Ts: Camera extrinsic matrices (B, 4, 4)
        image_size: (height, width) of output images
        point_size: Radius of each rendered point
        background_color: RGB background color (0-1 range)
        use_fp16: Retained for CLI compatibility. PyTorch3D rasterization uses
            float32 because its CUDA kernels do not accept float16 geometry.

    Returns:
        Tuple of (rgb_batch, depth_batch, mask_batch)
    """
    H, W = image_size
    device = points_3d.device
    B = Ks.shape[0]

    if len(points_3d) == 0:
        return (
            torch.full((B, H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device),
            torch.zeros((B, H, W), dtype=torch.float32, device=device),
            torch.zeros((B, H, W), dtype=torch.bool, device=device)
        )

    # Validate point cloud data - filter out NaN/Inf values
    valid_mask = torch.isfinite(points_3d).all(dim=1) & torch.isfinite(colors).all(dim=1)
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum().item()
        logger.warning(f"Filtering {num_invalid} invalid points before rendering")
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]

        if len(points_3d) == 0:
            return (
                torch.full((B, H, W, 3), int(background_color[0] * 255), dtype=torch.uint8, device=device),
                torch.zeros((B, H, W), dtype=torch.float32, device=device),
                torch.zeros((B, H, W), dtype=torch.bool, device=device)
            )

    # Keep geometry, camera parameters, and point features on the dtype
    # supported by PyTorch3D's CUDA rasterizer.  The control-map renderer is
    # small compared with the diffusion stage, so this has negligible impact
    # on peak end-to-end memory.
    compute_dtype = torch.float32

    cameras = _build_cam_from_extrinsics(
        Ks.to(compute_dtype), Ts.to(compute_dtype), image_size
    )

    colors_normalized = colors.to(compute_dtype) / 255.0
    points_3d_compute = points_3d.to(compute_dtype)

    # Create point cloud repeated for each frame
    point_cloud = Pointclouds(
        points=[points_3d_compute for _ in range(B)],
        features=[colors_normalized for _ in range(B)]
    )

    # Use bin_size=None to let PyTorch3D auto-select, which is more robust
    # bin_size=128 can cause CUDA errors with certain point distributions
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_size,
        points_per_pixel=8,
        bin_size=None,
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=background_color)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    with torch.no_grad():
        rendered_images = renderer(point_cloud)  # (B, H, W, 4)

        rendered_rgb = rendered_images[..., :3].float() * 255
        rendered_rgb = torch.clamp(rendered_rgb, 0, 255).to(torch.uint8)

        # Extract depth and mask from rasterizer
        fragments = rasterizer(point_cloud)
        masks = (fragments.idx[..., 0] != -1)  # (B, H, W)
        depth_maps = fragments.zbuf[..., 0].float().clone()  # (B, H, W)
        depth_maps[~masks] = 0.0

    return rendered_rgb, depth_maps, masks

def _build_cam_from_extrinsics(Ks: torch.Tensor, Ts: torch.Tensor, image_size: Tuple[int, int]):
    """Build PyTorch3D camera objects from intrinsics and extrinsics.

    Args:
        Ks: Intrinsic matrices (B, 3, 3)
        Ts: Extrinsic matrices world-to-camera (B, 4, 4)
        image_size: (height, width) of images

    Returns:
        PerspectiveCameras object for PyTorch3D rendering
    """
    H, W = image_size
    device = Ks.device
    B = Ks.shape[0]

    # Handle FP16 precision for matrix inversion
    original_dtype = Ts.dtype
    use_fp32_for_inv = original_dtype == torch.float16

    if use_fp32_for_inv:
        Ks = Ks.float()
        Ts = Ts.float()

    # Convert world-to-camera to camera-to-world
    c2ws = torch.linalg.inv(Ts)
    c2ws[:, :3, :2] *= -1  # Flip signs for NDC convention
    w2cs = torch.linalg.inv(c2ws)

    # Extract camera parameters from intrinsics
    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]

    focal_length = torch.stack([fx, fy], dim=1)  # (B, 2)
    principal_point = torch.stack([cx, cy], dim=1)  # (B, 2)

    R_cameras = w2cs[:, :3, :3].permute(0, 2, 1)  # (B, 3, 3)
    T_cameras = w2cs[:, :3, 3]  # (B, 3)

    # Cast back to original dtype if needed
    if use_fp32_for_inv:
        focal_length = focal_length.to(original_dtype)
        principal_point = principal_point.to(original_dtype)
        R_cameras = R_cameras.to(original_dtype)
        T_cameras = T_cameras.to(original_dtype)

    cameras = PerspectiveCameras(
        device=device,
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_cameras,
        T=T_cameras,
        in_ndc=False,
        image_size=[(H, W)] * B
    )
    return cameras

def composite_by_depth_batch(
    bg_rgb: torch.Tensor,
    bg_depth: torch.Tensor,
    fg_rgb: torch.Tensor,
    fg_depth: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    take_fg = fg_mask & ((bg_depth <= 0) | ((fg_depth > 0) & (fg_depth < bg_depth - 1e-6)))
    out_rgb = bg_rgb.clone()
    out_rgb[take_fg] = fg_rgb[take_fg]
    out_depth = bg_depth.clone()
    out_depth[take_fg] = fg_depth[take_fg]
    return out_rgb, out_depth


def merge_bg_and_fg_sequences(
    bg_rgb_frames: List[torch.Tensor],
    bg_depth_frames: List[torch.Tensor],
    bg_masks: List[torch.Tensor],
    fg_rgb_frames: List[torch.Tensor],
    fg_depth_frames: List[torch.Tensor],
    fg_masks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    assert len(bg_rgb_frames) == len(fg_rgb_frames), "Background and foreground frame counts must match"

    rgb_frames = []
    depth_frames = []

    for bg_rgb, bg_depth, bg_mask, fg_rgb, fg_depth, fg_mask in zip(
        bg_rgb_frames, bg_depth_frames, bg_masks, fg_rgb_frames, fg_depth_frames, fg_masks
    ):
        rgb_out, depth_out = composite_by_depth(bg_rgb, bg_depth, fg_rgb, fg_depth, fg_mask)
        rgb_frames.append(rgb_out)
        depth_frames.append(depth_out)

    return rgb_frames, depth_frames, bg_masks, fg_masks


def composite_by_depth(
    bg_rgb: torch.Tensor,
    bg_depth: torch.Tensor,
    fg_rgb: torch.Tensor,
    fg_depth: torch.Tensor,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert bg_rgb.shape[:2] == bg_depth.shape and fg_rgb.shape[:2] == fg_depth.shape
    H, W, _ = bg_rgb.shape
    device = bg_rgb.device
    take_fg = fg_mask & ((bg_depth <= 0) | (fg_depth > 0) & (fg_depth < bg_depth - 1e-6))
    out_rgb = bg_rgb.clone()
    out_rgb[take_fg] = fg_rgb[take_fg]
    out_depth = bg_depth.clone()
    out_depth[take_fg] = fg_depth[take_fg]
    return out_rgb, out_depth

def save_video_from_frames(
    frames: List[torch.Tensor],
    output_path: Path,
    fps: int = 10
):
    """Save a list of image frames as an MP4 video.

    Args:
        frames: List of image tensors (H, W, 3)
        output_path: Output video file path
        fps: Frames per second for the video
    """
    if len(frames) == 0:
        logger.warning(f"No frames to save for {output_path}")
        return

    # Convert grayscale to RGB if needed
    first_frame = frames[0]
    if first_frame.ndim == 2:
        frames = [f.unsqueeze(-1).repeat(1, 1, 3) for f in frames]

    frames_tensor = torch.stack(frames).detach().cpu()
    if frames_tensor.dtype != torch.uint8:
        frames_tensor = frames_tensor.clamp(0, 255).round().to(torch.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TorchVision 0.20's ``write_video`` passes the string ``"NONE"`` to
    # ``VideoFrame.pict_type`` and is incompatible with PyAV 17, which expects
    # an integer enum.  ImageIO's FFmpeg backend is already a project
    # dependency and keeps video writing independent of that private API.
    imageio.mimwrite(
        str(output_path),
        frames_tensor.numpy(),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18"],
        macro_block_size=1,
    )

def visualize_depth_as_grayscale(
    depth_frames: List[torch.Tensor],
    global_min_depth: Optional[float] = None,
    global_max_depth: Optional[float] = None
) -> List[torch.Tensor]:
    grayscale_frames = []

    # Determine depth range if not provided. Sample if dataset is large.
    if global_min_depth is None or global_max_depth is None:
        valid_depth_tensors = [d[d > 0].flatten() for d in depth_frames if torch.any(d > 0)]
        if valid_depth_tensors:
            all_depths = torch.cat(valid_depth_tensors)

            # If the depth array is very large, sample up to 1M points for percentile estimation
            if len(all_depths) > 1000000:
                indices = torch.randperm(len(all_depths))[:1000000]
                sampled_depths = all_depths[indices]
            else:
                sampled_depths = all_depths

            try:
                min_depth = torch.quantile(sampled_depths, 0.001)
                max_depth = torch.quantile(sampled_depths, 0.99)
            except RuntimeError:
                # Fallback to min/max if quantile fails
                min_depth = torch.min(sampled_depths)
                max_depth = torch.max(sampled_depths)
        else:
            min_depth, max_depth = 0.0, 1.0
    else:
        min_depth = global_min_depth
        max_depth = global_max_depth

    for depth in depth_frames:
        # Convert depth to disparity (closer -> larger value)
        disp = torch.where(depth > 0, 1.0 / depth, torch.tensor(0.0, device=depth.device))

        # Normalize disparity to [0, 1] (closer -> closer to 1)
        if max_depth > 0 and min_depth > 0:
            min_disp = 1.0 / max_depth
            max_disp = 1.0 / min_depth
            disp_norm = (disp - min_disp) / (max_disp - min_disp + 1e-8)
        else:
            disp_norm = disp

        disp_norm = torch.clamp(disp_norm, 0, 1)

        # Closer -> lighter (1 -> white, 0 -> black)
        gray_value = (disp_norm * 255).to(torch.uint8)
        gray_rgb = gray_value.unsqueeze(-1).repeat(1, 1, 3)
        grayscale_frames.append(gray_rgb)

    return grayscale_frames

def compute_global_depth_range(
    depth_frames_list: List[List[torch.Tensor]]
) -> Tuple[float, float]:
    all_valid_depths = []

    for depth_frames in depth_frames_list:
        valid_depth_tensors = [d[d > 0].flatten() for d in depth_frames if torch.any(d > 0)]
        if valid_depth_tensors:
            all_valid_depths.extend(valid_depth_tensors)

    if not all_valid_depths:
        return 0.0, 1.0

    combined_depths = torch.cat(all_valid_depths)

    # If the combined depth samples are very large, sample up to 1M points for percentiles
    if len(combined_depths) > 1000000:
        indices = torch.randperm(len(combined_depths))[:1000000]
        sampled_depths = combined_depths[indices]
    else:
        sampled_depths = combined_depths

    try:
        min_depth = torch.quantile(sampled_depths, 0.001).item()
        max_depth = torch.quantile(sampled_depths, 0.99).item()
    except RuntimeError:
        # Fallback to min/max if quantile fails
        min_depth = torch.min(sampled_depths).item()
        max_depth = torch.max(sampled_depths).item()

    return min_depth, max_depth

def project_3d_gaussians_to_2d(
    gaussian_params_per_frame: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    obj_id_to_color_idx: Dict[int, int],
    intrinsics: List[np.ndarray],
    extrinsics: List[np.ndarray],
    image_size: Tuple[int, int],
    threshold: float = 0.05,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Project 3D Gaussians to 2D image plane for each frame.

    Renders 3D Gaussian density maps to 2D using perspective projection,
    then composites multiple Gaussians with proper depth ordering.

    Args:
        gaussian_params_per_frame: List of Gaussian parameters {obj_id: (mean, cov)}
        obj_id_to_color_idx: Mapping from object ID to color index
        intrinsics: List of camera intrinsic matrices
        extrinsics: List of camera extrinsic matrices
        image_size: (width, height) of output images
        threshold: Density threshold for alpha compositing
        device: Computation device

    Returns:
        Tuple of (rgb_frames, alpha_frames) lists
    """
    image_width, image_height = image_size
    rgb_frames = []
    alpha_frames = []

    for frame_idx, gaussian_params in enumerate(gaussian_params_per_frame):
        if frame_idx >= len(intrinsics) or frame_idx >= len(extrinsics):
            break

        # Load intrinsic matrix
        if isinstance(intrinsics[frame_idx], np.ndarray):
            intrinsic_matrix = torch.from_numpy(intrinsics[frame_idx]).float().to(device)
        else:
            intrinsic_matrix = intrinsics[frame_idx].float().to(device)

        # Load extrinsic matrix
        if isinstance(extrinsics[frame_idx], np.ndarray):
            extrinsic_matrix = torch.from_numpy(extrinsics[frame_idx]).float().to(device)
        else:
            extrinsic_matrix = extrinsics[frame_idx].float().to(device)

        rotation_matrix = extrinsic_matrix[:3, :3]
        translation_vector = extrinsic_matrix[:3, 3:4]

        # Initialize output buffers
        rgb_frame = torch.zeros(
            (image_height, image_width, 3),
            dtype=torch.float32,
            device=device
        )
        alpha_frame = torch.zeros(
            (image_height, image_width),
            dtype=torch.float32,
            device=device
        )

        gaussian_list = []

        # Project each Gaussian to 2D
        for obj_id, (mean, covariance) in gaussian_params.items():
            if not isinstance(mean, torch.Tensor):
                mean = torch.from_numpy(mean).float().to(device)
            else:
                mean = mean.to(device)

            if not isinstance(covariance, torch.Tensor):
                covariance = torch.from_numpy(covariance).float().to(device)
            else:
                covariance = covariance.to(device)

            # Project Gaussian to 2D image plane
            density_map, depth_z = project_gaussian_to_2d_gpu(
                mean, covariance, intrinsic_matrix, rotation_matrix,
                translation_vector, image_size, device
            )

            # Only keep Gaussians in front of the camera
            if depth_z > 0:
                color_rgb = get_object_color(
                    obj_id, obj_id_to_color_idx, device, return_float=True
                )

                max_density = density_map.max()
                if max_density > 0:
                    normalized_density = density_map / (max_density + 1e-8)
                else:
                    normalized_density = density_map

                gaussian_list.append((normalized_density, color_rgb, depth_z, obj_id))

        # Sort by depth (far to near) for proper compositing
        gaussian_list.sort(key=lambda x: x[2], reverse=True)

        # Composite Gaussians
        for normalized_density, color_rgb, depth_z, obj_id in gaussian_list:
            alpha_new = torch.where(
                normalized_density > threshold,
                (normalized_density - threshold) / (1.0 - threshold + 1e-8),
                torch.zeros_like(normalized_density)
            ).clamp(0.0, 1.0)

            # RGB blending: C_out = C_fg * alpha_fg + C_bg * (1 - alpha_fg)
            alpha_expanded = alpha_new.unsqueeze(-1)
            rgb_frame = (
                color_rgb.view(1, 1, 3) * alpha_expanded +
                rgb_frame * (1 - alpha_expanded)
            )

            # Alpha blending: alpha_out = alpha_fg + alpha_bg * (1 - alpha_fg)
            alpha_frame = alpha_new + alpha_frame * (1 - alpha_new)

        alpha_frame = alpha_frame.clamp(0, 1)
        rgb_frame_uint8 = (rgb_frame.clamp(0, 1) * 255).to(torch.uint8)

        rgb_frames.append(rgb_frame_uint8)
        alpha_frames.append(alpha_frame)

    return rgb_frames, alpha_frames

def blend_gaussian_projection_with_bg(
    gaussian_rgb_frames: List[torch.Tensor],
    gaussian_alpha_frames: List[torch.Tensor],
    background_frames: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Blend Gaussian RGB projections with background using alpha compositing.

    Performs over operation: C_out = C_fg * alpha_fg + C_bg * (1 - alpha_fg)

    Args:
        gaussian_rgb_frames: List of Gaussian RGB projection frames
        gaussian_alpha_frames: List of Gaussian alpha (opacity) maps
        background_frames: List of background RGB frames

    Returns:
        List of blended RGB frames
    """
    assert len(gaussian_rgb_frames) == len(gaussian_alpha_frames) == len(background_frames), \
        "All input lists must have the same length"

    merged_frames = []

    for gaussian_rgb, alpha, bg_rgb in zip(gaussian_rgb_frames, gaussian_alpha_frames, background_frames):
        device = gaussian_rgb.device
        bg_rgb = bg_rgb.to(device)

        # Convert to float for blending
        gaussian_rgb_f = gaussian_rgb.float() / 255.0  # (H, W, 3)
        bg_rgb_f = bg_rgb.float() / 255.0  # (H, W, 3)

        # Alpha blend: C_out = C_fg * alpha + C_bg * (1 - alpha)
        alpha_3d = alpha.unsqueeze(-1)  # (H, W, 1)
        merged_f = gaussian_rgb_f * alpha_3d + bg_rgb_f * (1 - alpha_3d)

        merged_uint8 = (merged_f.clamp(0, 1) * 255).to(torch.uint8)
        merged_frames.append(merged_uint8)

    return merged_frames

def merge_bg_and_fg_mask(
    background_depth_frames: List[torch.Tensor],
    foreground_depth_frames: List[torch.Tensor],
    background_masks: List[torch.Tensor],
    foreground_masks: List[torch.Tensor],
    device: str = 'cuda'
) -> List[torch.Tensor]:

    mask_frames = []

    for bg_depth, fg_depth, bg_mask, fg_mask in zip(
        background_depth_frames, foreground_depth_frames, background_masks, foreground_masks
    ):
        # Foreground visible condition: has foreground mask AND (background has no depth OR foreground is closer)
        take_fg = fg_mask & ((bg_depth <= 0) | ((fg_depth > 0) & (fg_depth < bg_depth - 1e-6)))

        # Invert background mask: True where background should be visible
        bg_mask = ~bg_mask
        out_mask = bg_mask.clone()

        out_mask[take_fg] = fg_mask[take_fg]

        mask_rgb = torch.stack([out_mask, out_mask, out_mask], dim=-1)
        mask_uint8 = (mask_rgb * 255).to(torch.uint8)

        mask_frames.append(mask_uint8)

    return mask_frames

def project_gaussian_to_2d_gpu(
    mean: torch.Tensor,
    cov: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size: Tuple[int, int],
    device='cuda'
) -> Tuple[torch.Tensor, float]:
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).float().to(device)
    if isinstance(cov, np.ndarray):
        cov = torch.from_numpy(cov).float().to(device)
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).float().to(device)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float().to(device)

    means_3d = mean.unsqueeze(0)
    covs_3d = cov.unsqueeze(0)

    if t.dim() == 2:
        t_vec = t.squeeze()
    else:
        t_vec = t
    mean_3d_cam = R @ mean + t_vec

    density = compute_probability_density_map_gpu(
        means_3d, covs_3d, K, R, t, image_size, device
    )
    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    return density, mean_3d_cam[2].item()

def compute_probability_density_map_gpu(
    means_3d: torch.Tensor,
    covs_3d: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_size: Tuple[int, int],
    device='cuda'
) -> torch.Tensor:
    width, height = image_size

    u_coords = torch.arange(width, device=device, dtype=torch.float32)
    v_coords = torch.arange(height, device=device, dtype=torch.float32)
    u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
    pixel_coords = torch.stack([u_grid, v_grid], dim=-1)  # (H, W, 2)

    density_map = torch.zeros((height, width), device=device, dtype=torch.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    num_gaussians = means_3d.shape[0]

    for i in range(num_gaussians):
        mean_3d_world = means_3d[i]
        cov_3d_world = covs_3d[i]

        if t.dim() == 2:
            t_vec = t.squeeze()
        else:
            t_vec = t
        mean_3d_cam = R @ mean_3d_world + t_vec
        cov_3d_cam = R @ cov_3d_world @ R.T

        x, y, z = mean_3d_cam[0], mean_3d_cam[1], mean_3d_cam[2]

        # Avoid very small z values that would blow up the projected covariance
        if z <= 0.5:
            continue

        # Jacobian of the projection
        J = torch.tensor([
            [fx / z, 0, -fx * x / (z * z)],
            [0, fy / z, -fy * y / (z * z)]
        ], device=device, dtype=torch.float32)

        # 2D projected mean
        mean_2d = torch.tensor([
            fx * x / z + cx,
            fy * y / z + cy
        ], device=device, dtype=torch.float32)

        cov_2d = J @ cov_3d_cam @ J.T
        cov_2d += torch.eye(2, device=device) * 1e-6

        # If 2D covariance determinant is extremely large, skip (Gaussian too spread)
        det_cov = torch.det(cov_2d)
        if det_cov > 1e11:
            continue

        try:
            diff = pixel_coords - mean_2d  # (H, W, 2)
            diff_flat = diff.reshape(-1, 2)  # (H*W, 2)

            # Inverse covariance and determinant
            cov_inv = torch.linalg.inv(cov_2d)
            det_cov = torch.det(cov_2d)

            # Mahalanobis distance
            mahal_dist = torch.sum((diff_flat @ cov_inv) * diff_flat, dim=1)  # (H*W,)

            # PDF values
            coeff = 1.0 / (2 * torch.pi * torch.sqrt(det_cov))
            pdf_values = coeff * torch.exp(-0.5 * mahal_dist)
            pdf_map = pdf_values.reshape(height, width)

            density_map += pdf_map

        except Exception as e:
            logger.warning(f"Skipping Gaussian {i} due to error: {e}")
            continue

    return density_map

def get_object_color(obj_id: int, obj_id_to_color_idx: Dict[int, int], device: str = 'cuda',
                     return_float: bool = False) -> torch.Tensor:
    """Get distinct color for an object using tab20 colormap.

    Args:
        obj_id: Object identifier
        obj_id_to_color_idx: Mapping from object ID to color index
        device: Computation device
        return_float: If True, return color as float [0, 1]. If False, uint8 [0, 255]

    Returns:
        Color tensor
    """
    cmap = matplotlib.colormaps['tab20']
    color_idx = obj_id_to_color_idx.get(obj_id, 0)
    color_rgb = cmap(color_idx % 20)[:3]  # 0~1

    if return_float:
        return torch.tensor(color_rgb, dtype=torch.float32, device=device)
    else:
        color_255 = torch.tensor([c * 255 for c in color_rgb], dtype=torch.uint8, device=device)
        return color_255

def build_background(
    png_path: str,
    npz_path: str,
    mask_dir: str,
    device: str = 'cuda',
    target_height: int = None,
    target_width: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load background RGB, depth, masks and return 3D point cloud.

    Args:
        png_path: Path to RGB image
        npz_path: Path to NPZ file with depth and intrinsics
        mask_dir: Directory containing object masks
        device: Computation device
        target_height: Target height for resizing (if None, use original)
        target_width: Target width for resizing (if None, use original)

    Returns:
        Tuple of (points_3d, colors, depth, intrinsic, height, width)
    """
    # Load RGB image
    image = cv2.imread(png_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = image.shape[:2]

    # Determine target resolution
    if target_height is not None and target_width is not None:
        H, W = target_height, target_width
        logger.info(f"Resizing from {H_orig}×{W_orig} to {H}×{W} for faster rendering")
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        H, W = H_orig, W_orig

    image_tensor = torch.from_numpy(image).to(device)

    # Load depth and intrinsics
    data = np.load(npz_path)
    depth_orig = torch.from_numpy(data['depth'].astype(np.float32)).to(device)
    intrinsic = torch.from_numpy(data['intrinsic'].astype(np.float32)).to(device)

    # Resize depth map if needed
    if target_height is not None and target_width is not None:
        depth_orig_np = depth_orig.cpu().numpy()
        depth_resized = cv2.resize(depth_orig_np, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = torch.from_numpy(depth_resized).to(device)
    else:
        depth = depth_orig

    # Denormalize intrinsics if in normalized format
    intrinsic_denorm = intrinsic.clone()
    intrinsic_denorm[0, 0] *= W  # fx
    intrinsic_denorm[1, 1] *= H  # fy
    intrinsic_denorm[0, 2] *= W  # cx
    intrinsic_denorm[1, 2] *= H  # cy
    intrinsic = intrinsic_denorm

    # Assume camera at origin with identity extrinsic by default
    extrinsic = torch.eye(4, dtype=torch.float32, device=device)
    logger.info(f"Loaded NPZ data: depth={depth.shape}, intrinsic={intrinsic.shape}")
    logger.info(f"Intrinsic converted to pixel coordinates (fx={intrinsic[0,0]:.2f}, fy={intrinsic[1,1]:.2f})")

    # Load and combine all masks
    mask_files = sorted(Path(mask_dir).glob('*.png'))
    combined_mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask > 127).to(device)
        combined_mask |= mask_tensor

    logger.info(f"Loaded {len(mask_files)} mask files")

    # Dilate combined mask to remove boundary noise
    dilate_kernel_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    combined_mask_np = combined_mask.cpu().numpy().astype(np.uint8) * 255
    combined_mask_dilated = cv2.dilate(combined_mask_np, kernel, iterations=1)
    combined_mask = torch.from_numpy(combined_mask_dilated > 127).to(device)
    logger.info(f"Mask dilated (kernel_size={dilate_kernel_size})")

    # Kornia's recent ``depth_to_3d_v2`` implementation indexes the leading
    # camera dimension even for a single image.  Keep an explicit batch axis
    # here so both depth and intrinsics follow its documented ``(*, ...)``
    # contract instead of being interpreted as three separate cameras.
    pts3d_cam = depth_to_3d_v2(
        depth.unsqueeze(0), intrinsic.unsqueeze(0), normalize_points=False
    ).reshape(-1, 3)
    c2w = torch.linalg.inv(extrinsic)
    pts3d_hom = torch.cat([pts3d_cam, torch.ones(len(pts3d_cam), 1, device=pts3d_cam.device)], dim=1)
    pts3d_world_opencv = (c2w @ pts3d_hom.T).T[:, :3]

    # Convert point cloud from OpenCV coordinates to Blender coordinates
    coord_transform = torch.from_numpy(COORD_TRANSFORM_CV2BLENDER).to(device)
    pts3d_world = (coord_transform @ pts3d_world_opencv.T).T
    logger.info(f"Point cloud coordinates converted from OpenCV to Blender convention")

    # Extract background point cloud (inverse of combined mask)
    bg_mask = ~combined_mask
    bg_points = pts3d_world[bg_mask.reshape(-1)]
    bg_colors = image_tensor[bg_mask]

    # Filter out invalid points (NaN, Inf, or extreme values)
    valid_mask = torch.isfinite(bg_points).all(dim=1)
    # Also filter out points with extreme values that can cause CUDA errors
    coord_max = 1e6  # Maximum reasonable coordinate value
    valid_mask &= (bg_points.abs() < coord_max).all(dim=1)

    if valid_mask.sum() < len(bg_points):
        num_filtered = len(bg_points) - valid_mask.sum().item()
        logger.warning(f"Filtered out {num_filtered} invalid points from background point cloud")
        bg_points = bg_points[valid_mask]
        bg_colors = bg_colors[valid_mask]

    logger.info(f"Background point cloud: {len(bg_points)} valid points")

    return bg_points, bg_colors, intrinsic, extrinsic, H, W


def load_camera_trajectory(trajectory_npz: str, device: str = 'cuda') -> torch.Tensor:
    data = np.load(trajectory_npz)
    c2w_blender = torch.from_numpy(data['extrinsics'].astype(np.float32)).to(device)

    # Convert to OpenCV world-to-camera (w2c) convention
    c2w_blender[:,:3,1:3] *= -1
    extrinsics = torch.linalg.inv(c2w_blender)

    return extrinsics


def load_ellipsoid_parameters(
    json_path: str,
    device: str = 'cuda'
) -> Tuple[List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]], Dict[int, int], Dict[int, Dict[int, torch.Tensor]]]:

    with open(json_path, 'r') as f:
        data = json.load(f)

    num_frames = data['metadata']['num_frames']
    num_objects = data['metadata']['num_objects']

    obj_id_to_color_idx = {}
    for key, value in data['metadata']['obj_id_to_color_idx'].items():
        obj_id_to_color_idx[key] = value

    gaussian_params_per_frame = []
    center_points_per_frame = {}

    for frame_data in data['frames']:
        frame_idx = frame_data['frame_index']
        frame_params = {}

        if frame_idx not in center_points_per_frame:
            center_points_per_frame[frame_idx] = {}

        for obj_data in frame_data['objects']:
            obj_id = obj_data['object_id']

            mean = torch.tensor(obj_data['gaussian_3d']['mean'], dtype=torch.float32, device=device)
            cov = torch.tensor(obj_data['gaussian_3d']['covariance'], dtype=torch.float32, device=device)
            frame_params[obj_id] = (mean, cov)

            if obj_data['gaussian_3d']['mean'] is not None:
                center_points_per_frame[frame_idx][obj_id] = torch.tensor(
                    obj_data['gaussian_3d']['mean'], dtype=torch.float32, device=device
                )

        gaussian_params_per_frame.append(frame_params)

    return gaussian_params_per_frame, obj_id_to_color_idx, center_points_per_frame


def render_video_with_bg_and_fg(
    background_points: torch.Tensor,
    background_colors: torch.Tensor,
    foreground_meshes_per_frame: List[Optional[Meshes]],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    mode: str = 'full',
    point_size: float = 0.01,
    device: str = 'cuda',
    batch_size: int = 1,
    use_fp16: bool = False,
    pin_memory: bool = False
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    num_frames = len(foreground_meshes_per_frame)
    rgb_frames = [None] * num_frames
    depth_frames = [None] * num_frames
    background_masks = [None] * num_frames
    foreground_masks = [None] * num_frames

    H, W = image_size

    if background_points is not None:
        if pin_memory and not background_points.is_pinned():
            background_points = background_points.pin_memory()
            background_colors = background_colors.pin_memory()

    num_batches = (num_frames + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Rendering(mesh-batch) {mode}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_frames)
        current_batch_size = end_idx - start_idx

        Ks_batch = intrinsics[start_idx:end_idx]
        Ts_batch = extrinsics[start_idx:end_idx]

        # 渲染背景
        rgb_bg_batch = None
        depth_bg_batch = None
        mask_bg_batch = None
        if mode in ('full', 'background') and background_points is not None:
            rgb_bg_batch, depth_bg_batch, mask_bg_batch = render_point_cloud_pytorch3d_batch(
                background_points, background_colors, Ks_batch, Ts_batch, image_size, point_size, use_fp16=use_fp16
            )

        if rgb_bg_batch is None:
            rgb_bg_batch = torch.zeros((current_batch_size, H, W, 3), dtype=torch.uint8, device=device)
            depth_bg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.float32, device=device)
            mask_bg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.bool, device=device)

        # 渲染前景
        rgb_fg_batch = None
        if mode in ('full', 'foreground'):
            meshes_batch = foreground_meshes_per_frame[start_idx:end_idx]
            rgb_fg_batch, depth_fg_batch, mask_fg_batch = render_meshes_pytorch3d_batch(
                meshes_batch, Ks_batch, Ts_batch, image_size, use_fp16=use_fp16
            )
        else:
            rgb_fg_batch = torch.zeros((current_batch_size, H, W, 3), dtype=torch.uint8, device=device)
            depth_fg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.float32, device=device)
            mask_fg_batch = torch.zeros((current_batch_size, H, W), dtype=torch.bool, device=device)

        # 合成
        if mode == 'foreground':
            rgb_out_batch = rgb_fg_batch
            depth_out_batch = depth_fg_batch
        elif mode == 'background':
            rgb_out_batch = rgb_bg_batch
            depth_out_batch = depth_bg_batch
        else:
            rgb_out_batch, depth_out_batch = composite_by_depth_batch(
                rgb_bg_batch, depth_bg_batch, rgb_fg_batch, depth_fg_batch, mask_fg_batch
            )

        for i in range(current_batch_size):
            idx = start_idx + i
            rgb_frames[idx] = rgb_out_batch[i]
            depth_frames[idx] = depth_out_batch[i]
            background_masks[idx] = mask_bg_batch[i]
            foreground_masks[idx] = mask_fg_batch[i]

        del rgb_bg_batch, depth_bg_batch, mask_bg_batch
        del rgb_fg_batch, depth_fg_batch, mask_fg_batch
        del rgb_out_batch, depth_out_batch
        if device == 'cuda':
            torch.cuda.empty_cache()

    return rgb_frames, depth_frames, background_masks, foreground_masks



def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference mode: Render video from pre-computed parameters"
    )
    parser.add_argument('--png_path', type=str, required=False, help='Path to first frame PNG image (optional)')
    parser.add_argument('--video_path', type=str, required=False, help='Path to input MP4 video (optional)')
    parser.add_argument('--npz_path', type=str, required=True, help='Path to NPZ file with depth and camera pose')
    parser.add_argument('--mask_dir', type=str, required=False, help='Directory containing mask images')
    parser.add_argument('--mask_video', type=str, required=False, help='Path to mask video (mp4)')
    parser.add_argument('--trajectory_npz', type=str, required=True, help='Path to camera trajectory NPZ file')
    parser.add_argument('--ellipsoid_json', type=str, required=True, help='Path to ellipsoid parameters JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/inference', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--point_size', type=float, default=0.005, help='Point size for rendering')
    parser.add_argument('--fps', type=int, default=10, help='Output video FPS')
    parser.add_argument('--render_batch_size', type=int, default=27, help='Batch size for rendering')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 for rendering')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory')
    parser.add_argument('--ellipsoid_subdiv', type=int, default=3, help='Icosphere subdivisions for ellipsoid mesh')
    parser.add_argument('--trajectory_radius', type=float, default=0.03, help='Trajectory line radius')
    parser.add_argument('--gaussian_mask_threshold', type=float, default=0.003, help='Gaussian projection threshold')
    parser.add_argument('--sample_frames', type=int, default=10, help='Number of frames to sample')
    parser.add_argument('--target_height', type=int, default=720, help='Target rendering height')
    parser.add_argument('--target_width', type=int, default=1280, help='Target rendering width')
    return parser.parse_args()


def main():
    """Main pipeline for rendering 4D control maps.

    Steps:
    1. Load background point cloud from RGB, depth, and masks
    2. Load camera trajectory
    3. Load 3D Gaussian parameters for each frame
    4. Build ellipsoid meshes representing the 3D Gaussians
    5. Render background (point cloud only)
    6. Render foreground (ellipsoids only)
    7. Composite and generate control maps (RGB, depth, masks)
    8. Project 3D Gaussians to 2D and generate projection maps
    9. Blend Gaussian projections with background
    10. Save all outputs as MP4 videos
    """
    args = parse_args()
    device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load background point cloud
    logger.info("Step 1: Loading background point cloud")
    bg_points, bg_colors, first_intrinsics, first_extrinsic, image_height, image_width = build_background(
        args.png_path, args.npz_path, args.mask_dir, device,
        target_height=args.target_height,
        target_width=args.target_width
    )

    # Step 2: Load camera trajectory
    logger.info("Step 2: Loading camera trajectory")
    extrinsics = load_camera_trajectory(args.trajectory_npz, device)
    num_frames = len(extrinsics)

    if first_intrinsics.ndim == 3:
        intrinsics = first_intrinsics
    else:
        intrinsics = first_intrinsics.unsqueeze(0).repeat(num_frames, 1, 1)

    # Step 3: Load 3D Gaussian parameters
    logger.info("Step 3: Loading 3D Gaussian trajectory")
    gaussian_params_per_frame, obj_id_to_color_idx, center_points_per_frame = load_ellipsoid_parameters(
        args.ellipsoid_json, device
    )

    # Step 4: Build ellipsoid meshes for each frame
    logger.info("Step 4: Building ellipsoid meshes")
    ellipsoid_meshes_per_frame = []

    for frame_idx in range(num_frames):
        if frame_idx < len(gaussian_params_per_frame):
            gaussian_params = gaussian_params_per_frame[frame_idx]
        else:
            gaussian_params = {}

        ellipsoid_meshes = []

        for obj_id, (mean, cov) in gaussian_params.items():
            obj_color = get_object_color(obj_id, obj_id_to_color_idx, device)

            ellipsoid_scale_factor = 2.5  # default value

            mesh = make_ellipsoid_mesh(
                mean, cov,
                scale_factor=ellipsoid_scale_factor,
                subdivisions=args.ellipsoid_subdiv,
                color_rgb255=obj_color,
                device=device
            )
            ellipsoid_meshes.append(mesh)

        if len(ellipsoid_meshes) > 0:
            combined_mesh = combine_meshes_for_scene(ellipsoid_meshes)
            ellipsoid_meshes_per_frame.append(combined_mesh)
        else:
            ellipsoid_meshes_per_frame.append(None)

    # Step 5: Render background
    logger.info("Step 5: Rendering background")
    bg_rgb_frames, bg_depth_frames, bg_masks, _ = render_video_with_bg_and_fg(
        bg_points, bg_colors,
        ellipsoid_meshes_per_frame,
        intrinsics, extrinsics, (image_height, image_width),
        mode='background',
        point_size=args.point_size,
        device=device,
        batch_size=args.render_batch_size,
        use_fp16=args.use_fp16,
        pin_memory=args.pin_memory
    )

    save_video_from_frames(bg_rgb_frames, output_dir / "background_RGB.mp4", args.fps)

    # 6. Render ellipsoid foreground
    logger.info("Rendering ellipsoid foreground")

    fg_rgb_frames, fg_depth_frames, _, fg_masks = render_video_with_bg_and_fg(
        bg_points, bg_colors,
        ellipsoid_meshes_per_frame,
        intrinsics, extrinsics, (image_height, image_width),
        mode='foreground',
        point_size=args.point_size,
        device=device,
        batch_size=args.render_batch_size,
        use_fp16=args.use_fp16,
        pin_memory=args.pin_memory
    )

    # 7. Background and foreground depth
    logger.info("Computing background and foreground depth")

    _, combined_depth_frames, _, _ = merge_bg_and_fg_sequences(
        bg_rgb_frames, bg_depth_frames, bg_masks,
        fg_rgb_frames, fg_depth_frames, fg_masks
    )

    global_min_depth, global_max_depth = compute_global_depth_range([
        bg_depth_frames, fg_depth_frames, combined_depth_frames
    ])

    logger.info(f"Global depth range: min={global_min_depth:.4f}, max={global_max_depth:.4f}")

    bg_depth_grayscale = visualize_depth_as_grayscale(bg_depth_frames, global_min_depth, global_max_depth)
    save_video_from_frames(bg_depth_grayscale, output_dir / "background_depth.mp4", args.fps)

    fg_depth_grayscale = visualize_depth_as_grayscale(fg_depth_frames, global_min_depth, global_max_depth)
    save_video_from_frames(fg_depth_grayscale, output_dir / "3D_gaussian_depth.mp4", args.fps)

    # 8. Generate merged masks for background and foreground
    merged_mask_frames = merge_bg_and_fg_mask(
        bg_depth_frames, fg_depth_frames, bg_masks, fg_masks, device=device
    )
    save_video_from_frames(merged_mask_frames, output_dir / "merged_mask.mp4", args.fps)


    # 9. Generate 3D Gaussian RGB projections
    logger.info("Generating 3D Gaussian RGB projections")

    if len(gaussian_params_per_frame) > 0 and any(len(frame_params) > 0 for frame_params in gaussian_params_per_frame):
        while len(gaussian_params_per_frame) < num_frames:
            gaussian_params_per_frame.append({})

        gaussian_rgb_frames, gaussian_alpha_frames = project_3d_gaussians_to_2d(
            gaussian_params_per_frame,
            obj_id_to_color_idx,
            intrinsics.cpu().numpy(),
            extrinsics.cpu().numpy(),
            (image_width, image_height),
            threshold=args.gaussian_mask_threshold,
            device=device
        )

        gaussian_projection_frames = []
        for rgb, alpha in zip(gaussian_rgb_frames, gaussian_alpha_frames):
            alpha_3d = alpha.unsqueeze(-1)  # (H, W, 1)
            rgb_normalized = rgb.float() / 255.0
            masked_gaussian_rgb = (rgb_normalized * alpha_3d * 255).to(torch.uint8)
            gaussian_projection_frames.append(masked_gaussian_rgb)

        save_video_from_frames(
            gaussian_projection_frames,
            output_dir / "3D_gaussian_RGB.mp4",
            args.fps
        )
    else:
        gaussian_rgb_frames = [
            torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)
            for _ in range(num_frames)
        ]
        gaussian_alpha_frames = [
            torch.zeros((image_height, image_width), dtype=torch.float32, device=device)
            for _ in range(num_frames)
        ]
        gaussian_projection_frames = [
            torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)
            for _ in range(num_frames)
        ]
        save_video_from_frames(
            gaussian_projection_frames,
            output_dir / "3D_gaussian_RGB.mp4",
            args.fps
        )
        logger.info("No objects detected; generated empty Gaussian projection video")

    # 10. Composite 3D Gaussian RGB with background RGB
    logger.info("Generating background + 3D Gaussian composite video")

    gaussian_with_bg_frames = blend_gaussian_projection_with_bg(
        gaussian_rgb_frames,
        gaussian_alpha_frames,
        bg_rgb_frames
    )

    gaussian_visibility_masks = [alpha > 0.001 for alpha in gaussian_alpha_frames]

    gaussian_with_bg_frames, _, _, _ = merge_bg_and_fg_sequences(
        bg_rgb_frames, bg_depth_frames, bg_masks,
        gaussian_with_bg_frames, fg_depth_frames, gaussian_visibility_masks
    )

    save_video_from_frames(
        gaussian_with_bg_frames,
        output_dir / "background_and_3D_gaussian.mp4",
        args.fps
    )

    logger.info("=" * 80)
    logger.info("Rendering complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
