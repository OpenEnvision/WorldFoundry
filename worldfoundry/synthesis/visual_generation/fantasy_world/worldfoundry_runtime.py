"""
This module provides utility functions for handling camera parameters, normalizing frame counts,
and saving prediction artifacts specifically for the FantasyWorld model and related pipelines.
It includes functionalities for parsing various camera input formats, ensuring correct frame counts,
and storing generated videos and point clouds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

def build_camera_params(
    camera_source: Any,
    image_size: tuple[int, int],
    K=None,
    *,
    model_name: str = "FantasyWorld",
):
    """Constructs camera parameters for the FantasyWorld model from various input sources.

    This function supports loading camera trajectories from JSON files, dictionaries,
    NumPy arrays, or lists of existing camera objects. It ensures the camera parameters
    are in a format suitable for downstream processing within the FantasyWorld ecosystem.

    Args:
        camera_source: The source of camera parameters. Can be:
            - A string or Path to a JSON file containing camera data.
            - A dictionary with camera data, expected to contain "cameras_interp".
            - A 3D NumPy array representing camera poses.
            - A list or tuple of camera objects (e.g., with 'w2c_mat' attribute)
              or lists/arrays convertible to camera poses.
            - None, which raises a ValueError.
        image_size: A tuple (width, height) specifying the dimensions of the images.
        K: Optional camera intrinsic matrix. If None, it will be derived or default.
        model_name: The name of the model requiring camera parameters, used for error messages.

    Returns:
        A list of camera objects or parameters, formatted for FantasyWorld.

    Raises:
        ValueError: If `camera_source` is None or if a dictionary source
                    does not contain "cameras_interp".
        TypeError: If `camera_source` is of an unsupported type.
    """
    from . import utils as fw_utils

    if camera_source is None:
        raise ValueError(f"{model_name} requires a camera trajectory input.")

    if isinstance(camera_source, (str, Path)):
        # Handle camera data provided as a path to a JSON file
        with Path(camera_source).expanduser().resolve().open("r", encoding="utf-8") as file:
            camera_data = json.load(file)
        return fw_utils.cameras_json_to_camera_list(camera_data, image_size=image_size, K=K)

    if isinstance(camera_source, dict):
        # Handle camera data provided as a dictionary
        if "cameras_interp" not in camera_source:
            raise ValueError("FantasyWorld camera dict must contain `cameras_interp`.")
        return fw_utils.cameras_json_to_camera_list(camera_source, image_size=image_size, K=K)

    if isinstance(camera_source, np.ndarray) and camera_source.ndim == 3:
        # Handle camera data provided as a 3D NumPy array
        return fw_utils.cameras_json_to_camera_list(
            {"cameras_interp": camera_source.tolist()},
            image_size=image_size,
            K=K,
        )

    if isinstance(camera_source, (list, tuple)):
        # Handle camera data provided as a list or tuple
        if camera_source and hasattr(camera_source[0], "w2c_mat"):
            # If the list contains pre-built camera objects, return them directly
            return list(camera_source)
        # Otherwise, assume it's a list of poses to be converted
        return fw_utils.cameras_json_to_camera_list(
            {"cameras_interp": [np.asarray(item).tolist() for item in camera_source]},
            image_size=image_size,
            K=K,
        )

    raise TypeError(f"Unsupported FantasyWorld camera input type: {type(camera_source)!r}")


def normalize_wan_num_frames(num_frames: int) -> int:
    """Normalizes the number of frames for the WAN (While-and-Normalize) model.

    Ensures that the number of frames adheres to the `4k+1` format required by the model.
    If the input `num_frames` is not already in this format, it will be adjusted
    to the nearest valid value that is greater than or equal to the input.

    Args:
        num_frames: The desired number of frames as an integer.

    Returns:
        The normalized number of frames.

    Raises:
        ValueError: If `num_frames` is not positive.
    """
    num_frames = int(num_frames)
    if num_frames <= 0:
        raise ValueError(f"FantasyWorld num_frames must be positive, got {num_frames}.")
    # Adjust num_frames to be in the format 4k+1, which is a requirement for some models.
    # This calculation finds the smallest 4k+1 number greater than or equal to num_frames.
    if num_frames % 4 != 1:
        return (num_frames + 2) // 4 * 4 + 1
    return num_frames


def pad_camera_params_to_frames(camera_params, num_frames: int):
    """Pads a list of camera parameters to a specified number of frames.

    If the number of provided camera parameters is less than `num_frames`,
    the list is extended by repeating the last camera parameter until `num_frames` is reached.
    If `camera_params` already has `num_frames` or more, it is returned as-is (converted to a list).

    Args:
        camera_params: A list or iterable of camera parameters.
        num_frames: The target total number of frames.

    Returns:
        A list of camera parameters padded to `num_frames`.

    Raises:
        ValueError: If `camera_params` is empty.
    """
    camera_params = list(camera_params)
    if not camera_params:
        raise ValueError("FantasyWorld requires at least one camera pose.")
    # If there are fewer camera parameters than target frames, pad by repeating the last camera
    if len(camera_params) < num_frames:
        camera_params.extend([camera_params[-1]] * (num_frames - len(camera_params)))
    return camera_params


def save_prediction_artifacts(
    *,
    frames,
    prediction,
    output_dir: str,
    scene_name: str,
    fps: int,
    conf_threshold: float,
    stride: int,
    mask_operator: str,
) -> dict[str, str]:
    """Saves generated video and a colored point cloud from model predictions.

    This function processes the input frames and prediction data to create
    a video file and a PLY format point cloud, saving them to a specified
    output directory. The point cloud generation includes filtering based on
    a confidence threshold.

    Args:
        frames: The input frames (e.g., as a list of images or a NumPy array)
                to be saved as a video and used for point cloud coloring.
        prediction: A dictionary containing model prediction results, expected
                    to include 'depth_conf' for confidence-based filtering.
        output_dir: The base directory where the scene-specific output will be saved.
        scene_name: The name of the scene, used to create a subdirectory within `output_dir`.
        fps: Frames per second for the output video.
        conf_threshold: The confidence threshold used to filter points in the point cloud.
        stride: Stride to apply when sampling points for the point cloud.
        mask_operator: The operator to use for confidence filtering (e.g., ">=" or ">").

    Returns:
        A dictionary containing the paths to the generated video, point cloud,
        and the scene output directory.

    Raises:
        ValueError: If an unsupported `mask_operator` is provided.
    """
    from . import utils as fw_utils

    scene_dir = Path(output_dir).expanduser().resolve() / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    video_path = scene_dir / "video.mp4"
    fw_utils.save_video_imageio(frames, video_path, fps=fps)

    recon_worldpoints = fw_utils.get_pointclouds(prediction, fix_first_frame=True)

    # Determine the valid mask for point cloud filtering based on confidence and operator
    if mask_operator == ">=":
        valid_mask = prediction["depth_conf"] >= conf_threshold
    elif mask_operator == ">":
        valid_mask = prediction["depth_conf"] > conf_threshold
    else:
        raise ValueError(f"Unsupported FantasyWorld mask operator: {mask_operator!r}")

    ply_path = scene_dir / f"recon_confthresh{conf_threshold}.ply"
    fw_utils.save_colored_pointcloud_ply(
        points=recon_worldpoints,
        colors=frames,
        out_path=ply_path,
        stride=stride,
        max_points=None,
        # Convert the confidence mask (Tensor) to a NumPy array for compatibility,
        # and select the first batch dimension if present.
        valid_mask=valid_mask.cpu().numpy()[0],
    )

    return {
        "generated_video_path": str(video_path),
        "pointcloud_path": str(ply_path),
        "output_dir": str(scene_dir),
    }


__all__ = [
    "build_camera_params",
    "normalize_wan_num_frames",
    "pad_camera_params_to_frames",
    "save_prediction_artifacts",
]
