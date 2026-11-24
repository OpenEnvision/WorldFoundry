"""Official Runtime visual generation pipeline module."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Infer-only adapter based on the official VGGT-Omega Gradio demo.

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import trimesh
from matplotlib import colormaps
from scipy.spatial.transform import Rotation
from trimesh.visual import ColorVisuals

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_image_path(path: str | Path) -> bool:
    """Check if the given value represents a valid image file path."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _is_video_path(path: str | Path) -> bool:
    """Check if the given value represents a valid video file path."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def collect_input_images(input_source: str | Path | Sequence[str | Path]) -> list[str]:
    """Gather and filter input conditioning images."""
    if isinstance(input_source, (list, tuple)):
        return sorted(str(Path(p)) for p in input_source)

    source = Path(cast("str | Path", input_source))
    if source.is_dir():
        image_dir = source / "images" if (source / "images").is_dir() else source
        return sorted(str(p) for p in image_dir.iterdir() if p.is_file() and _is_image_path(p))
    if source.is_file():
        if _is_image_path(source) or _is_video_path(source):
            return [str(source)]
        if source.suffix.lower() == ".txt":
            return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported VGGT-Omega input source: {input_source}")


def prepare_target_images(
    input_source: str | Path | Sequence[str | Path],
    target_dir: str | Path,
    *,
    video_sample_fps: float = 1.0,
) -> list[str]:
    """Normalize and scale target images for the runtime."""
    target_dir = Path(target_dir)
    images_dir = target_dir / "images"
    if images_dir.exists():
        # Safely clean up any pre-existing output directory structure
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    for source_path in collect_input_images(input_source):
        source = Path(source_path)
        if _is_image_path(source):
            destination = images_dir / source.name
            # Copy source file to target directory while preserving original metadata
            shutil.copy2(source, destination)
            image_paths.append(str(destination))
            continue
        if _is_video_path(source):
            image_paths.extend(_extract_video_frames(source, images_dir, video_sample_fps=video_sample_fps))
            continue
        raise ValueError(f"Unsupported VGGT-Omega input file: {source}")

    image_paths = sorted(image_paths)
    if not image_paths:
        raise ValueError(f"No images found for VGGT-Omega input: {input_source}")
    return image_paths


def _extract_video_frames(video_path: Path, images_dir: Path, *, video_sample_fps: float = 1.0) -> list[str]:
    """Extract individual frames from a video file."""
    # Initialize video capture stream to extract frame data and FPS properties
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video input: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    video_sample_fps = max(float(video_sample_fps), 0.1)
    frame_interval = max(round((fps if fps and fps > 0 else 1) / video_sample_fps), 1)

    image_paths: list[str] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            image_path = images_dir / f"{saved_idx:06d}.png"
            cv2.imwrite(str(image_path), frame)
            image_paths.append(str(image_path))
            saved_idx += 1
        frame_idx += 1
    video.release()
    return image_paths


def run_model(
    image_paths: Sequence[str | Path],
    model: Any,
    *,
    device: str | None = None,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    patch_size: int = 16,
) -> dict[str, Any]:
    """Run VGGT-Omega on the images, sorted by path, and return NumPy predictions.

    Model tensors and decoded camera matrices are converted to float32, with a leading dimension
    of size 1 removed. The predictions include ``extrinsic`` and ``intrinsic`` from the pose encoding
    and float64 ``world_points_from_depth`` unprojected from the depth map.

    Raises:
        ValueError: CUDA is requested but unavailable, or ``preprocess_mode`` is unsupported.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.vggt_omega.utils.load_fn import (
        load_and_preprocess_images,
    )
    from worldfoundry.base_models.three_dimensions.point_clouds.vggt_omega.vggt_omega.utils.pose_enc import (
        encoding_to_camera,
    )

    sorted_paths = sorted(str(Path(p)) for p in image_paths)
    images = load_and_preprocess_images(
        sorted_paths,
        mode=_normalize_preprocess_mode(preprocess_mode),
        image_resolution=image_resolution,
        patch_size=patch_size,
    ).to(device)

    model = model.to(device).eval()
    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    if device_type == "cuda":
        torch.cuda.empty_cache()
    return predictions_np


def _normalize_preprocess_mode(preprocess_mode: str) -> str:
    if preprocess_mode in {"balanced", "max_size"}:
        return preprocess_mode
    if preprocess_mode in {"crop", "pad", "square"}:
        return "balanced"
    raise ValueError(f"Unsupported VGGT-Omega preprocess_mode: {preprocess_mode}")


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject depth maps to world points.

    Args:
        depth_map: Depth of shape ``[S, H, W, 1]``.
        extrinsic: World-to-camera matrices of shape ``[S, 3, 4]``.
        intrinsic: Pinhole intrinsics of shape ``[S, 3, 3]``.

    Returns:
        World points of shape ``[S, H, W, 3]``.
    """
    depth = np.asarray(depth_map)[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def predictions_to_glb(
    predictions: dict[str, Any],
    *,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> trimesh.Scene:
    """Build a ``trimesh`` scene of the depth-based point cloud and, optionally, camera frustums.

    Points at depth edges (when ``filter_depth_edges``), below the ``conf_thres`` confidence percentile
    (at least 2), and optionally near-black or near-white points are dropped; at most ``max_points``
    evenly spaced points are kept. The scene is aligned to the first camera.

    Raises:
        ValueError: ``predictions`` is not a dict.
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    conf_thres = max(2.0, float(conf_thres))
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        mask &= conf >= np.percentile(conf[mask], conf_thres)
    mask &= conf > 1e-5

    if mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)

    if vertices.size == 0:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower = np.percentile(vertices, 5, axis=0)
        upper = np.percentile(vertices, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices
    extrinsics[:, 3, 3] = 1.0

    if show_cam:
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, world_to_camera in enumerate(extrinsics):
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba = colormap(i / max(len(extrinsics), 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, camera_to_world, color, scene_scale)

    return apply_scene_alignment(scene, extrinsics)


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    """Mark pixels whose ``kernel_size`` neighborhood depth range exceeds ``rtol`` times their depth."""
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple[int, ...],
    scene_scale: float,
) -> None:
    """Add a frustum mesh for the camera pose ``transform``, sized by ``scene_scale``, to ``scene``."""
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    vertices = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices = transform_points(complete_transform, vertices)

    camera_mesh = trimesh.Trimesh(vertices=vertices, faces=compute_camera_faces(camera_cone_shape))
    cast(ColorVisuals, camera_mesh.visual).face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene: trimesh.Scene, extrinsics: np.ndarray) -> trimesh.Scene:
    """Flip y/z, then apply the first camera-to-world transform to ``scene`` in place.

    ``extrinsics`` contains world-to-camera matrices. Returns the same scene.
    """
    scene.apply_transform(np.linalg.inv(extrinsics[0]) @ get_opengl_conversion_matrix())
    return scene


def get_opengl_conversion_matrix() -> np.ndarray:
    """Return the 4x4 matrix that flips the y and z axes between OpenCV and OpenGL camera frames."""
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    """Apply a homogeneous ``transformation`` to ``points`` of shape ``[..., D]`` and keep the first ``dim`` coordinates."""
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """Return the triangle faces of a camera frustum built from three stacked copies of ``cone_shape``'s vertices."""
    faces = []
    num_vertices = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices
        faces.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )
    faces += [(v3, v2, v1) for v1, v2, v3 in faces]
    return np.array(faces)


def glb_path(
    target_dir: str | Path,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    max_points_k: int,
) -> str:
    """Return the GLB file path in ``target_dir`` that encodes the export settings."""
    return str(
        Path(target_dir)
        / (
            f"scene_conf{conf_thres}_black{mask_black_bg}_white{mask_white_bg}_"
            f"cam{show_cam}_skyFalse_max{int(max_points_k)}k.glb"
        )
    )


def run_official_scene_export(
    input_source: str | Path | Sequence[str | Path],
    model: Any,
    *,
    output_dir: str | Path,
    device: str | None = None,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    patch_size: int = 16,
    video_sample_fps: float = 1.0,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    max_points_k: int = 1000,
    output_name: str | None = None,
) -> dict[str, str]:
    """Run VGGT-Omega on an input source and export ``predictions.npz`` and a GLB scene to ``output_dir``.

    Returns:
        Dict with ``glb_path``, ``prediction_path`` and ``image_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = prepare_target_images(input_source, output_dir, video_sample_fps=video_sample_fps)

    predictions = run_model(
        image_paths,
        model,
        device=device,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
        patch_size=patch_size,
    )

    prediction_save_path = output_dir / "predictions.npz"
    np.savez(str(prediction_save_path), **predictions)

    if output_name is None:
        glbfile = glb_path(output_dir, conf_thres, mask_black_bg, mask_white_bg, show_cam, max_points_k)
    else:
        glbfile = str(output_dir / output_name)

    scene = predictions_to_glb(
        predictions,
        conf_thres=max(3.0, float(conf_thres)),
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        max_points=int(max_points_k * 1000),
    )
    scene.export(file_obj=glbfile)

    if torch.device(device or "cpu").type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "glb_path": glbfile,
        "prediction_path": str(prediction_save_path),
        "image_dir": str(output_dir / "images"),
    }


__all__ = [
    "collect_input_images",
    "predictions_to_glb",
    "prepare_target_images",
    "run_model",
    "run_official_scene_export",
]
