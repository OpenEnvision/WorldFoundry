"""NeoVerse 3D geometry transformations."""

import numpy as np
import torch


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def project_world_points_to_image(world_points, extrinsics_cam, intrinsics_cam, return_camera_points=False):
    """
    Project 3D world points to 2D image coordinates using camera extrinsics and intrinsics.

    Args:
        world_points (torch.Tensor): Batch of world points of shape (S, N, 3).
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4) or (S, 4, 4).
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3).

    Returns:
        torch.Tensor: Batch of 2D image coordinates of shape (S, N, 2).
        torch.Tensor: Batch of depth values of shape (S, N, 1).
        torch.Tensor: Batch of camera points of shape (S, N, 3).
    """
    # Project world points to camera coordinates
    world_to_cam_R = extrinsics_cam[:, :3, :3]  # (S, 3, 3)
    world_to_cam_T = extrinsics_cam[:, :3, 3]  # (S, 3)

    camera_points = torch.einsum("bij, bnj -> bni", world_to_cam_R, world_points) + world_to_cam_T[:, None]

    # Normalize camera points
    camera_points_normalized = camera_points / (camera_points[..., 2:3] + 1e-8)  # Avoid division by zero

    pixel_coords = torch.einsum("bij, bnj -> bni", intrinsics_cam, camera_points_normalized)
    # Extract depth values
    depth_values = camera_points[..., 2:3]  # Keep the depth channel

    if return_camera_points:
        return pixel_coords[..., :2], depth_values, camera_points

    return pixel_coords[..., :2], depth_values
