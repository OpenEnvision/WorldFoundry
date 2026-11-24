"""Utility functions for depth estimation pipeline.

Consolidated from DeployService/utils/ and DeployService/model/pi3/utils/.
"""

import cv2
import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────
# Depth I/O (from image_utils.py)
# ──────────────────────────────────────────────


def save_depth_rgba_float(path, depth_float32):
    """Save float32 depth map as lossless RGBA PNG.

    Each float32 pixel is reinterpreted as 4 uint8 bytes → RGBA channels.
    """
    depth_bytes = depth_float32.astype(np.float32).tobytes()
    depth_uint8 = np.frombuffer(depth_bytes, dtype=np.uint8).reshape(depth_float32.shape[0], depth_float32.shape[1], 4)
    img = Image.fromarray(depth_uint8, mode="RGBA")
    img.save(path, format="PNG", compress_level=0)


# ──────────────────────────────────────────────
# PLY writing (from pi3/utils/basic.py)
# ──────────────────────────────────────────────


def rotate_target_dim_to_last_axis(x, target_dim=3):
    """Move the last axis of size *target_dim* to the final position."""
    shape = x.shape
    axis_to_move = -1
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        ret = x.transpose(*dims_order)
    else:
        ret = x
    return ret


def write_ply(xyz, rgb=None, path="output.ply"):
    """Write a coloured point cloud to a PLY file."""
    from plyfile import PlyData, PlyElement

    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.0

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)

    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)

        hue = 0.7 * normalized_coord[:, 0] + 0.2 * normalized_coord[:, 1] + 0.1 * normalized_coord[:, 2]
        hsv = np.stack([hue, 0.9 * np.ones_like(hue), 0.8 * np.ones_like(hue)], axis=1)

        c = hsv[:, 2:] * hsv[:, 1:2]
        x_val = c * (1 - np.abs((hsv[:, 0:1] * 6) % 2 - 1))
        m = hsv[:, 2:] - c

        rgb = np.zeros_like(hsv)
        cond = (hsv[:, 0] * 6 % 6 >= 0) & (hsv[:, 0] * 6 % 6 < 1)
        rgb[cond] = np.hstack([c[cond], x_val[cond], np.zeros_like(x_val[cond])])
        cond = (hsv[:, 0] * 6 % 6 >= 1) & (hsv[:, 0] * 6 % 6 < 2)
        rgb[cond] = np.hstack([x_val[cond], c[cond], np.zeros_like(x_val[cond])])
        cond = (hsv[:, 0] * 6 % 6 >= 2) & (hsv[:, 0] * 6 % 6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x_val[cond]), c[cond], x_val[cond]])
        cond = (hsv[:, 0] * 6 % 6 >= 3) & (hsv[:, 0] * 6 % 6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x_val[cond]), x_val[cond], c[cond]])
        cond = (hsv[:, 0] * 6 % 6 >= 4) & (hsv[:, 0] * 6 % 6 < 5)
        rgb[cond] = np.hstack([x_val[cond], np.zeros_like(x_val[cond]), c[cond]])
        cond = (hsv[:, 0] * 6 % 6 >= 5) & (hsv[:, 0] * 6 % 6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x_val[cond]), x_val[cond]])
        rgb = rgb + m

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# ──────────────────────────────────────────────
# Ground-plane alignment (from ground_align.py)
# ──────────────────────────────────────────────


def align_ground_plane(points, ground_mask, ransac_iterations=2000, ransac_threshold=0.05, min_inliers_ratio=0.2):
    """Fit a ground plane via RANSAC and compute RT matrix to make it horizontal.

    Args:
        points: 3D point cloud (H, W, 3).
        ground_mask: binary mask (H, W).

    Returns:
        rt_matrix (4x4), aligned_points, plane_params, inliers, error
    """
    if ground_mask is None:
        return np.eye(4), None, None, None, None

    h, w = points.shape[0], points.shape[1]
    if points.shape[0:2] != ground_mask.shape:
        ground_mask = cv2.resize(ground_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ground_points = points[ground_mask == 1]
    if np.inf in ground_points or np.nan in ground_points:
        valid_points_mask = (ground_points != np.inf) & (ground_points != np.nan)
        ground_points = ground_points[valid_points_mask].reshape(-1, 3)
    if len(ground_points) < 10:
        print(f"Warning: too few ground points ({len(ground_points)} < 10)")
        return np.eye(4), None, None, None, None

    if ground_points.ndim == 1:
        ground_points = ground_points.reshape(1, -1)

    print(f"Ground plane fitting: {len(ground_points)} points")

    best_plane_params, best_inliers, best_error = _robust_ransac_plane_fitting(
        ground_points, ransac_iterations, ransac_threshold, min_inliers_ratio
    )

    # keep y-down as default
    if best_plane_params[1] > 0:
        best_plane_params = -best_plane_params

    ground_center = np.mean(ground_points, axis=0)
    rotation_matrix = _compute_ground_alignment_rotation(
        best_plane_params, ground_points_center=ground_center, normal_axis="y", normal_direction="down"
    )

    inlier_count = len(best_inliers) if best_inliers is not None else 0
    print(
        f"Plane params: a={best_plane_params[0]:.4f}, b={best_plane_params[1]:.4f}, "
        f"c={best_plane_params[2]:.4f}, d={best_plane_params[3]:.4f}"
    )
    print(f"Inliers: {inlier_count}/{len(ground_points)}, error: {best_error:.4f}")

    C_rotated = rotation_matrix @ ground_center
    t = np.array([0, -C_rotated[1], 0])

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = t

    aligned_points = (rotation_matrix @ points.reshape(-1, 3).T).T + t

    return rt_matrix, aligned_points, best_plane_params, best_inliers, best_error


def _robust_ransac_plane_fitting(points, iterations, threshold, min_inliers_ratio):
    best_plane_params = None
    best_inliers = None
    best_error = float("inf")
    best_inliers_count = 0

    adaptive_thresholds = [threshold * 2, threshold * 1.5, threshold]

    for _stage, current_threshold in enumerate(adaptive_thresholds):
        stage_iterations = iterations // len(adaptive_thresholds)

        for _ in range(stage_iterations):
            if len(points) >= 3:
                random_indices = np.random.choice(len(points), 3, replace=False)
                p1, p2, p3 = points[random_indices]

                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)

                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-10:
                    continue

                normal = normal / normal_norm
                a, b, c = normal
                d = -np.dot(normal, p1)

                distances = np.abs(np.dot(points, normal) + d)
                inliers = distances < current_threshold
                inliers_count = np.sum(inliers)

                if inliers_count < 3:
                    continue

                mean_error = np.mean(distances[inliers])

                if inliers_count > best_inliers_count or (
                    inliers_count == best_inliers_count and mean_error < best_error
                ):
                    best_plane_params = np.array([a, b, c, d])
                    best_inliers = np.where(inliers)[0]
                    best_error = mean_error
                    best_inliers_count = inliers_count

        if best_inliers_count >= len(points) * min_inliers_ratio:
            break

    if best_inliers is None or best_inliers_count < len(points) * min_inliers_ratio:
        print("RANSAC insufficient inliers, falling back to least-squares")
        best_plane_params, best_inliers, best_error = _least_squares_fallback(points)

    return best_plane_params, best_inliers, best_error


def _compute_ground_alignment_rotation(
    plane_params, ground_points_center=None, normal_axis="y", normal_direction="down"
):
    a, b, c, d = plane_params
    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    axis_map = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
    target_normal = axis_map[normal_axis]
    if normal_direction == "down":
        target_normal = -target_normal

    if np.abs(np.dot(plane_normal, target_normal)) > 0.9999:
        return np.eye(3)

    rotation_axis = np.cross(plane_normal, target_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    cos_angle = np.clip(np.dot(plane_normal, target_normal), -1.0, 1.0)
    rotation_angle = np.arccos(cos_angle)

    return _rodrigues_rotation_matrix(rotation_axis, rotation_angle)


def _rodrigues_rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    I = np.eye(3)  # noqa: E741 - Ambiguous upstream name kept as written.
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def _least_squares_fallback(points):
    try:
        X = points
        A = np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])
        b = -X[:, 2]
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        a, b_coeff, d = coeffs
        c = -1.0
        norm_factor = np.sqrt(a**2 + b_coeff**2 + c**2)
        plane_params = np.array([a / norm_factor, b_coeff / norm_factor, c / norm_factor, d / norm_factor])
        return plane_params, np.arange(len(points)), 0.0
    except np.linalg.LinAlgError:
        return np.array([0, 0, 1, 0]), np.arange(len(points)), float("inf")


# ──────────────────────────────────────────────
# Pose smoothing (from pose_smooth.py)
# ──────────────────────────────────────────────


def smooth_gaussian(poses, sigma=1.0):
    """Gaussian-smooth camera poses (quaternion + translation)."""
    from scipy import ndimage
    from scipy.spatial.transform import Rotation as R

    quats = []
    translations = []
    for pose in poses:
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        quats.append(R.from_matrix(rot).as_quat())
        translations.append(trans)

    quats = np.array(quats)
    translations = np.array(translations)

    smooth_quats = np.zeros_like(quats)
    smooth_trans = np.zeros_like(translations)

    for i in range(4):
        smooth_quats[:, i] = ndimage.gaussian_filter1d(quats[:, i], sigma=sigma)
    for i in range(3):
        smooth_trans[:, i] = ndimage.gaussian_filter1d(translations[:, i], sigma=sigma)

    norms = np.linalg.norm(smooth_quats, axis=1)
    smooth_quats = smooth_quats / norms[:, np.newaxis]

    smooth_poses = []
    for q, t in zip(smooth_quats, smooth_trans, strict=False):
        rot = R.from_quat(q).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = t
        smooth_poses.append(pose)

    return smooth_poses
