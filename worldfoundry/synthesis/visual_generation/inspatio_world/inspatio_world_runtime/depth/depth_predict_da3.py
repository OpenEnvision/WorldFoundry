"""DepthPredictDA3 — standalone depth estimation using Depth-Anything-3.

Extracted from DeployService/model/depth_predict_da3.py.
No dependency on BaseModel, Ray, ConfigManager, or DeployService utilities.
"""

import gc
import glob
import logging
import os
import time

import cv2
import numpy as np
import torch

from .depth_utils import (
    align_ground_plane,
    save_depth_rgba_float,
    smooth_gaussian,
    write_ply,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default config values (match old model_config.yaml)
# ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = {
    "model_path": os.path.join(_PROJECT_ROOT, "checkpoints", "DA3"),
    "fix_resize": True,
    "fix_resize_height": 480,
    "fix_resize_width": 832,
    "num_frames": 1000,
    "save_point_cloud": True,
}


# ──────────────────────────────────────────────
# Video / image loaders
# ──────────────────────────────────────────────


def load_video(video_path: str, max_frames: int = 81):
    frames_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    return frames_list


def load_images(image_paths: list[str], max_frames: int | None = None):
    frames_list = []
    max_frames = len(image_paths) if max_frames is None else min(max_frames, len(image_paths))
    for i in range(max_frames):
        img_bgr = cv2.imread(image_paths[i])
        if img_bgr is None:
            raise OSError(f"Failed to load image: {image_paths[i]}")
        frames_list.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return frames_list


def filter_depth_outliers(depth_map, valid_mask, method="iqr", iqr_factor=1.5, percentile_low=1, percentile_high=99):
    filtered_depth_map = depth_map.copy()
    outlier_mask = np.zeros_like(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        return filtered_depth_map, outlier_mask

    valid_depths = depth_map[valid_mask]
    if len(valid_depths) == 0:
        return filtered_depth_map, outlier_mask

    if method == "iqr":
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        outlier_indices = (valid_depths < lower_bound) | (valid_depths > upper_bound)
    elif method == "percentile":
        lower_bound = np.percentile(valid_depths, percentile_low)
        upper_bound = np.percentile(valid_depths, percentile_high)
        outlier_indices = (valid_depths < lower_bound) | (valid_depths > upper_bound)
    elif method == "zscore":
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)
        z_scores = np.abs((valid_depths - mean_depth) / std_depth)
        outlier_indices = z_scores > 3
    else:
        raise ValueError(f"Unsupported outlier method: {method}")

    valid_indices = np.where(valid_mask)
    outlier_mask[valid_indices[0][outlier_indices], valid_indices[1][outlier_indices]] = True
    filtered_depth_map[outlier_mask] = 0
    return filtered_depth_map, outlier_mask


# ──────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────


class DepthPredictDA3:
    """Depth estimation + pose extraction using Depth-Anything-3."""

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        logger.info("DepthPredictDA3 init, model_path=%s", self.config["model_path"])

        from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import DepthAnything3

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnything3.from_pretrained(self.config["model_path"])
        self.model = model.to(device=DEVICE)
        self.device = DEVICE
        logger.info("DepthPredictDA3 model loaded on %s", DEVICE)

        self.outlier_filter_config = {
            "enable": self.config.get("enable_depth_outlier_filter", False),
            "method": self.config.get("outlier_filter_method", "iqr"),
            "iqr_factor": self.config.get("outlier_filter_iqr_factor", 1.5),
            "percentile_low": self.config.get("outlier_filter_percentile_low", 1),
            "percentile_high": self.config.get("outlier_filter_percentile_high", 99),
        }

    # ── helpers ──

    @staticmethod
    def save_frames_to_images(frames, w, h, output_dir):
        for i, frame in enumerate(frames):
            if torch.is_tensor(frame):
                frame_np = frame.detach().cpu().numpy()
                frame_np = np.transpose(frame_np, (1, 2, 0))
                frame_np = (frame_np * 255).astype(np.uint8)
                frame = frame_np
            elif isinstance(frame, np.ndarray):
                frame = frame.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported frame type: {type(frame)}")

            frame_path = f"{output_dir}/{i:04d}.jpg"
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (w, h):
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if not cv2.imwrite(frame_path, frame_bgr):
                raise RuntimeError(f"Failed to save frame {i} to {frame_path}")

    @staticmethod
    def load_seg_masks(mask_path: str, max_frames: int = 41):
        from PIL import Image as PILImage

        sources = []
        cap = cv2.VideoCapture(mask_path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {mask_path}")
        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sources.append(PILImage.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
        return np.stack(sources)[:, :, :, 0] / 255.0

    @staticmethod
    def depthmap_to_local_points(depth_map: np.ndarray, intrinsics: np.ndarray):
        H, W = depth_map.shape
        if intrinsics.shape == (3, 3):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        else:
            fx, fy, cx, cy = intrinsics[:4]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth_map.reshape(-1)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1)

    # ── main pipeline ──

    def run(self, input_files: dict[str, str | list[str]], output_dir: str) -> bool:
        st = time.time()

        image_files = input_files.get("images")
        video_files = input_files.get("videos")
        frames_list = None

        fix_resize_params = None
        if self.config["fix_resize"]:
            fix_resize_params = (self.config["fix_resize_width"], self.config["fix_resize_height"])

        if fix_resize_params:
            target_height, target_width = fix_resize_params[1], fix_resize_params[0]

        max_frames = self.config["num_frames"]

        if image_files:
            if not isinstance(image_files, list) or len(image_files) == 0:
                raise ValueError("images must be a non-empty list")
            image_files = sorted(image_files)
            frames_list = load_images(image_files, max_frames=max_frames)
            target_height = frames_list[0].shape[0]
            target_width = frames_list[0].shape[1]
            logger.info("Loaded %d frames from %d image files", len(frames_list), len(image_files))
        elif video_files:
            video_path = video_files[0]
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            frames_list = load_video(video_path, max_frames=max_frames)
            logger.info("Loaded %d frames from video", len(frames_list))
        else:
            raise ValueError("input_files must contain 'images' or 'videos' key")

        # Ground mask
        ground_seg_idx = -100
        ground_mask = None
        try:
            ground_mask_file = glob.glob(os.path.join(output_dir, "ground_mask_*.png"))[0]
            ground_mask = cv2.imread(ground_mask_file, cv2.IMREAD_GRAYSCALE) / 255.0
            ground_mask[ground_mask < 0.5] = 0.0
            ground_mask[ground_mask >= 0.5] = 1.0
            ground_seg_idx = int(os.path.basename(ground_mask_file).split("_")[-1].split(".")[0])
        except Exception:
            pass

        if frames_list is None:
            raise ValueError("Could not parse input files")

        logger.info("Loaded %d images total", len(frames_list))

        # Create output directories
        frame_dir = os.path.join(output_dir, "frames")
        pcd_dir = os.path.join(output_dir, "frames_pcd")
        depth_dir = os.path.join(output_dir, "depth")
        depth_half_dir = os.path.join(output_dir, "depth_half")
        for d in [output_dir, frame_dir, pcd_dir, depth_dir, depth_half_dir]:
            os.makedirs(d, exist_ok=True)

        self.save_frames_to_images(frames_list, target_width, target_height, frame_dir)

        # Seg masks
        mask_file = os.path.join(output_dir, "mask.mp4")
        seg_masks = None
        if os.path.exists(mask_file):
            seg_masks = self.load_seg_masks(mask_file, max_frames=max_frames)
            seg_masks[seg_masks < 0.5] = 0.0
            seg_masks[seg_masks >= 0.5] = 1.0

        masked_frames = []
        for idx, frame in enumerate(frames_list):
            if seg_masks is not None:
                frame[seg_masks[idx] == 0] = np.array([0, 0, 0])
            masked_frames.append(frame)

        # Inference
        torch.cuda.reset_peak_memory_stats()
        s_infer = time.time()
        logger.info("Running DA3 inference...")
        res = self.model.inference(masked_frames, use_ray_pose=False)
        infer_h, infer_w = res.processed_images.shape[1:3]
        logger.info("Inference done in %.1fs", time.time() - s_infer)
        peak = torch.cuda.max_memory_allocated()
        logger.info("Peak GPU memory: %.2f GB", peak / 1024**3)

        ratio_w = target_width * 1.0 / infer_w
        ratio_h = target_height * 1.0 / infer_h
        seg_masks = None

        camera_poses = res.extrinsics
        depth_maps = res.depth
        frames = res.processed_images
        intrinsics = res.intrinsics

        # Convert to 4x4 c2w poses
        if torch.is_tensor(camera_poses):
            camera_poses = camera_poses.cpu().numpy()
        N = camera_poses.shape[0]
        camera_poses_4x4 = np.zeros((N, 4, 4), dtype=camera_poses.dtype)
        camera_poses_4x4[:, :3, :4] = camera_poses
        camera_poses_4x4[:, 3, 3] = 1.0
        camera_poses = np.linalg.inv(camera_poses_4x4)  # w2c -> c2w

        # Normalize to first frame
        T0_inv = np.linalg.inv(camera_poses[0])
        camera_poses = T0_inv @ camera_poses

        depth_scale_factor = 1.0
        camera_poses[:, :3, 3] *= depth_scale_factor

        # Ground plane alignment
        ground_rt_matrix = np.eye(4)
        frame_pts = []
        frame_cls = []
        frame_bboxes = []
        valid_depths = []

        for idx in range(len(frames)):
            local_pts = self.depthmap_to_local_points(depth_maps[idx], intrinsics[idx])
            local_pts = local_pts * depth_scale_factor
            if ground_seg_idx == idx and ground_mask is not None:
                local_pts_homo = np.concatenate(
                    [local_pts.reshape(-1, 3), np.ones((local_pts.reshape(-1, 3).shape[0], 1))], axis=1
                )
                global_pts = camera_poses[idx] @ local_pts_homo.T
                global_pts = global_pts.T[:, :3]
                global_pts = global_pts.reshape(depth_maps[idx].shape[0], depth_maps[idx].shape[1], 3)
                rotation_matrix, aligned_points, plane_params, inliers, error = align_ground_plane(
                    global_pts, ground_mask, ransac_iterations=2000, ransac_threshold=0.05, min_inliers_ratio=0.2
                )
                ground_rt_matrix = rotation_matrix

        # Save depth maps
        for idx in range(len(frames)):
            local_seg_mask = seg_masks[idx] if seg_masks is not None else np.ones_like(depth_maps[idx])

            depth_map = depth_maps[idx] * depth_scale_factor
            depth_map[local_seg_mask == 0] = 0.0

            if seg_masks is not None and self.outlier_filter_config["enable"]:
                depth_map, outlier_mask = filter_depth_outliers(
                    depth_map,
                    local_seg_mask.numpy(),
                    method=self.outlier_filter_config["method"],
                    iqr_factor=self.outlier_filter_config["iqr_factor"],
                    percentile_low=self.outlier_filter_config["percentile_low"],
                    percentile_high=self.outlier_filter_config["percentile_high"],
                )

            valid_depth_mask = depth_map > 0
            valid_depths.append(valid_depth_mask)

            depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            depth_half = cv2.resize(
                depth_map, (int(target_width * 0.5), int(target_height * 0.5)), interpolation=cv2.INTER_NEAREST
            )

            save_depth_rgba_float(f"{depth_dir}/{idx:04d}.png", depth_map)
            save_depth_rgba_float(f"{depth_half_dir}/{idx:04d}.png", depth_half)

        # Save point clouds
        if self.config["save_point_cloud"]:
            for idx in range(len(frames)):
                local_pts = self.depthmap_to_local_points(depth_maps[idx], intrinsics[idx])
                local_pts = (local_pts * depth_scale_factor).reshape(
                    depth_maps[idx].shape[0], depth_maps[idx].shape[1], 3
                )
                valid_depth_mask = valid_depths[idx]

                R_c2w = camera_poses[idx][:3, :3]
                T_c2w = camera_poses[idx][:3, 3]

                glb_pts = local_pts[valid_depth_mask]
                glb_pts = glb_pts @ R_c2w.T + T_c2w
                frame_pts.append(glb_pts)
                frame_cls.append(frames[idx][valid_depth_mask])

                glb_pts_homo = np.concatenate([glb_pts, np.ones((glb_pts.shape[0], 1))], axis=1)
                glb_pts_new = (ground_rt_matrix @ glb_pts_homo.T).T
                glb_pts = glb_pts_new[:, :3]
                write_ply(glb_pts, frames[idx][valid_depth_mask], os.path.join(pcd_dir, f"{idx:04d}.ply"))

                min_xyz = np.min(glb_pts, axis=0)
                max_xyz = np.max(glb_pts, axis=0)
                frame_bboxes.append([min_xyz[0], min_xyz[1], min_xyz[2], max_xyz[0], max_xyz[1], max_xyz[2]])

        if len(frame_bboxes) > 0:
            with open(os.path.join(output_dir, "frame_bboxes.txt"), "w") as fp:
                for bbox in frame_bboxes:
                    fp.write(" ".join(f"{v}" for v in bbox) + "\n")

        if len(frame_pts):
            write_ply(frame_pts[0], frame_cls[0], os.path.join(output_dir, "point_cloud.ply"))

        # Save intrinsics
        intrinsic_path = os.path.join(output_dir, "intrinsic.txt")
        with open(intrinsic_path, "w") as fp:
            for mat in intrinsics:
                mat[0, :] *= ratio_w
                mat[1, :] *= ratio_h
                np.savetxt(fp, mat)

        # Save extrinsics (smoothed)
        extrinsic_path = os.path.join(output_dir, "extrinsic.txt")
        try:
            smooth_poses = smooth_gaussian(camera_poses, sigma=2.0)
        except Exception:
            smooth_poses = camera_poses
        with open(extrinsic_path, "w") as fp:
            for mat in smooth_poses:
                mat_rot = ground_rt_matrix @ mat
                mat_rot_w2c = np.linalg.inv(mat_rot)
                np.savetxt(fp, mat_rot_w2c[:3, :])

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Finished in %.1fs", time.time() - st)
        return True
