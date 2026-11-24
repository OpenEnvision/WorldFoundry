"""Official Runtime visual generation pipeline module."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This infer-only adapter is based on the official VGGT demo utilities.

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.visual import ColorVisuals

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import VGGT
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.geometry import (
    unproject_depth_map_to_point_map,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import (
    load_and_preprocess_images,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

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
        paths = [str(Path(p)) for p in input_source]
        return sorted(paths)

    source = Path(cast("str | Path", input_source))
    if source.is_dir():
        image_dir = source / "images" if (source / "images").is_dir() else source
        return sorted(str(p) for p in image_dir.iterdir() if p.is_file() and _is_image_path(p))
    if source.is_file():
        if _is_image_path(source) or _is_video_path(source):
            return [str(source)]
        if source.suffix.lower() == ".txt":
            return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported VGGT input source: {input_source}")


def prepare_target_images(
    input_source: str | Path | Sequence[str | Path],
    target_dir: str | Path,
) -> list[str]:
    """Normalize and scale target images for the runtime."""
    target_dir = Path(target_dir)
    images_dir = target_dir / "images"
    if images_dir.exists():
        # Safely clean up any pre-existing output directory structure
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    source_paths = collect_input_images(input_source)
    copied_paths: list[str] = []
    for source_path in source_paths:
        source = Path(source_path)
        if _is_image_path(source):
            destination = images_dir / source.name
            # Copy source file to target directory while preserving original metadata
            shutil.copy2(source, destination)
            copied_paths.append(str(destination))
            continue
        if _is_video_path(source):
            copied_paths.extend(_extract_video_frames(source, images_dir))
            continue
        raise ValueError(f"Unsupported VGGT input file: {source}")

    copied_paths = sorted(copied_paths)
    if not copied_paths:
        raise ValueError(f"No images found for VGGT input: {input_source}")
    return copied_paths


def _extract_video_frames(video_path: Path, images_dir: Path) -> list[str]:
    """Extract individual frames from a video file."""
    # Initialize video capture stream to extract frame data and FPS properties
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video input: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps * 1), 1)
    image_paths: list[str] = []
    count = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        if count % frame_interval != 0:
            continue
        frame_path = images_dir / f"{frame_index:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        image_paths.append(str(frame_path))
        frame_index += 1
    capture.release()
    return image_paths


def run_model(
    image_paths: Sequence[str | Path],
    model: VGGT,
    *,
    device: str | None = None,
    preprocess_mode: str = "crop",
) -> dict[str, Any]:
    """Run VGGT on the images, sorted by path, and return NumPy predictions without the batch dimension.

    The predictions include ``extrinsic`` and ``intrinsic`` from the pose encoding and
    ``world_points_from_depth`` unprojected from the depth map. CUDA runs under bfloat16 autocast on
    compute capability 8.0 and newer, float16 otherwise.

    Raises:
        ValueError: CUDA is requested but unavailable, or no image is given.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    sorted_paths = sorted(str(Path(p)) for p in image_paths)
    if not sorted_paths:
        raise ValueError("At least one image is required for VGGT official inference.")

    model = model.to(device).eval()
    images = load_and_preprocess_images(sorted_paths, mode=preprocess_mode).to(device)
    if device_type == "cuda":
        # Use bfloat16 precision to balance memory efficiency and numeric range
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type == "cuda")):
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    numpy_predictions: dict[str, Any] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            numpy_predictions[key] = value.detach().cpu().numpy().squeeze(0)
        else:
            numpy_predictions[key] = value
    numpy_predictions["pose_enc_list"] = None

    depth_map = numpy_predictions["depth"]
    world_points = unproject_depth_map_to_point_map(
        depth_map,
        numpy_predictions["extrinsic"],
        numpy_predictions["intrinsic"],
    )
    numpy_predictions["world_points_from_depth"] = world_points

    if device_type == "cuda":
        torch.cuda.empty_cache()
    return numpy_predictions


def predictions_to_glb(
    predictions: dict[str, Any],
    *,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    prediction_mode: str = "Predicted Pointmap",
) -> trimesh.Scene:
    """Build a ``trimesh`` scene of the predicted point cloud and, optionally, camera frustums.

    Points come from ``world_points`` in a pointmap ``prediction_mode`` when present, otherwise from
    ``world_points_from_depth``. Points below the ``conf_thres`` confidence percentile, and optionally
    near-black or near-white points, are dropped; ``filter_by_frames`` such as ``"3: name"`` keeps one
    frame. The scene is aligned to the first camera.

    Raises:
        ValueError: ``predictions`` is not a dict.
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")
    if conf_thres is None:
        conf_thres = 10.0

    selected_frame_idx = _parse_frame_filter(filter_by_frames)
    if "Pointmap" in prediction_mode:
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    colors_rgb = np.transpose(images, (0, 2, 3, 1)) if images.ndim == 4 and images.shape[1] == 3 else images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    conf_threshold = 0.0 if conf_thres == 0.0 else np.percentile(conf, conf_thres)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        conf_mask = conf_mask & (colors_rgb.sum(axis=1) >= 16)
    if mask_white_bg:
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d.size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1.0
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper_percentile - lower_percentile))

    scene_3d = trimesh.Scene()
    scene_3d.add_geometry(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    return apply_scene_alignment(scene_3d, extrinsics_matrices)


def _parse_frame_filter(filter_by_frames: str) -> int | None:
    if filter_by_frames in {"all", "All"}:
        return None
    try:
        return int(str(filter_by_frames).split(":")[0])
    except (ValueError, IndexError):
        return None


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
    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    cast(ColorVisuals, camera_mesh.visual).face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """Apply a y-axis half-turn, flip y/z, then apply the first camera-to-world transform.

    ``extrinsics_matrices`` contains world-to-camera matrices. Mutates and returns ``scene_3d``.
    """
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ get_opengl_conversion_matrix() @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


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
    transformed = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return transformed[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """Return the triangle faces of a camera frustum built from three stacked copies of ``cone_shape``'s vertices."""
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone
        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def run_official_scene_export(
    input_source: str | Path | Sequence[str | Path],
    model: VGGT,
    *,
    output_dir: str | Path,
    device: str | None = None,
    preprocess_mode: str = "crop",
    conf_thres: float = 3.0,
    frame_filter: str = "All",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    prediction_mode: str = "Pointmap Regression",
    output_name: str | None = None,
) -> dict[str, str]:
    """Run VGGT on an input source and export ``predictions.npz`` and a GLB scene to ``output_dir``.

    Returns:
        Dict with ``glb_path``, ``prediction_path`` and ``image_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = prepare_target_images(input_source, output_dir)

    predictions = run_model(
        image_paths,
        model,
        device=device,
        preprocess_mode=preprocess_mode,
    )

    prediction_save_path = output_dir / "predictions.npz"
    np.savez(str(prediction_save_path), **predictions)

    if output_name is None:
        output_name = (
            f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}"
            f"_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}"
            f"_pred{prediction_mode.replace(' ', '_')}.glb"
        )
    glb_path = output_dir / output_name

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=str(glb_path))

    if torch.device(device or "cpu").type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "glb_path": str(glb_path),
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
