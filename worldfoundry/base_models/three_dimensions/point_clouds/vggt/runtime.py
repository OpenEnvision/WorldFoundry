import os
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.flash_world.render import (
    gaussian_render,
)
from worldfoundry.base_models.three_dimensions.point_clouds.ply_io import (
    read_point_cloud as fetchPly,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.models.vggt import (
    VGGT,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.geometry import (
    unproject_depth_map_to_point_map,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.load_fn import (
    load_and_preprocess_images,
    load_and_preprocess_images_square,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)

_SH_C0 = 0.28209479177387814


def _rgb_to_sh0(colors: torch.Tensor) -> torch.Tensor:
    """Encode linear RGB as degree-zero spherical-harmonic coefficients."""

    return (colors - 0.5) / _SH_C0


def _opencv_world_to_opengl(points: np.ndarray) -> np.ndarray:
    """Convert VGGT's OpenCV-aligned world points to right/up/backward axes."""

    converted = np.asarray(points).copy()
    converted[..., 1:3] *= -1.0
    return converted


def _opencv_w2c_to_opengl_c2w(world_to_camera: np.ndarray) -> np.ndarray:
    """Convert an OpenCV world-to-camera pose into an OpenGL c2w pose."""

    w2c = np.eye(4, dtype=np.float64)
    pose = np.asarray(world_to_camera, dtype=np.float64)
    if pose.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected a 3x4 or 4x4 pose, got {pose.shape}")
    w2c[: pose.shape[0], : pose.shape[1]] = pose
    conversion = np.diag((1.0, -1.0, -1.0, 1.0))
    return conversion @ np.linalg.inv(w2c) @ conversion


def _look_at_c2w_opengl(look_at: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """Build a proper right-handed OpenGL camera-to-world matrix.

    Columns are camera right, up, and backward.  ``gaussian_render`` performs
    the single OpenGL-to-COLMAP conversion expected by gsplat afterwards.
    """

    backward = eye - look_at
    backward = backward / (np.linalg.norm(backward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, backward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(world_up, backward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(backward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = backward
    c2w[:3, 3] = eye
    return c2w


class VGGTRepresentation:
    """VGGT Representation model for 3D scene reconstruction."""
    
    def __init__(self, model: Optional[VGGT] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()[0]
                self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> 'VGGTRepresentation':
        try:
            local_model_path = None
            if os.path.isdir(pretrained_model_path):
                candidate = os.path.join(pretrained_model_path, "model.pt")
                if os.path.isfile(candidate):
                    local_model_path = candidate
            elif os.path.isfile(pretrained_model_path) and pretrained_model_path.endswith(".pt"):
                local_model_path = pretrained_model_path

            if local_model_path is not None:
                model = VGGT()
                state_dict = torch.load(local_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict)
            else:
                model = VGGT.from_pretrained(pretrained_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load VGGT model: {e}")
        
        instance = cls(model=model, device=device)
        instance.preprocess_mode = kwargs.get('preprocess_mode', 'crop')
        instance.resolution = kwargs.get('resolution', 518)
        return instance
    
    
    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get representation from input data using VGGT model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() first.")
        
        images_input = data['images']
        predict_cameras = data.get('predict_cameras', True)
        predict_depth = data.get('predict_depth', True)
        predict_points = data.get('predict_points', True)
        predict_tracks = data.get('predict_tracks', False)
        query_points = data.get('query_points', None)
        preprocess_mode = data.get('preprocess_mode', self.preprocess_mode)
        resolution = data.get('resolution', self.resolution)
        if isinstance(images_input, list):
            image_list = images_input
        elif isinstance(images_input, np.ndarray):
            if images_input.ndim == 3:
                image_list = [images_input]
            elif images_input.ndim == 4:
                image_list = [images_input[i] for i in range(images_input.shape[0])]
            else:
                image_list = [images_input]
        else:
            if isinstance(images_input, str):
                image_list = [images_input]
            else:
                image_list = images_input if isinstance(images_input, list) else [images_input]
        
        has_paths = any(isinstance(img, str) for img in image_list)
        
        if has_paths:
            if preprocess_mode == "square":
                images, _ = load_and_preprocess_images_square(image_list, target_size=resolution)
            else:
                images = load_and_preprocess_images(image_list, mode=preprocess_mode)
        else:
            image_tensors = []
            for img_array in image_list:
                if isinstance(img_array, np.ndarray):
                    if img_array.max() > 1.0:
                        img_array = img_array / 255.0
                    
                    if img_array.ndim == 3 and img_array.shape[2] == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                    elif img_array.ndim == 2:
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
                        img_tensor = img_tensor.repeat(3, 1, 1)
                    else:
                        raise ValueError(f"Unsupported image array shape: {img_array.shape}")
                    img_tensor = F.interpolate(
                        img_tensor.unsqueeze(0),
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    image_tensors.append(img_tensor)
                else:
                    raise ValueError(f"Unsupported image type: {type(img_array)}")
            
            images = torch.stack(image_tensors)
        
        images = images.to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        query_points_tensor = None
        if predict_tracks and query_points is not None:
            if isinstance(query_points, np.ndarray):
                query_points_tensor = torch.FloatTensor(query_points).to(self.device)
            elif isinstance(query_points, torch.Tensor):
                query_points_tensor = query_points.to(self.device)
        
        results = {}
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=(self.device == "cuda")):
                images_batch = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                
                if predict_cameras:
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images_batch.shape[-2:]
                    )
                    results['extrinsic'] = extrinsic.squeeze(0).cpu().numpy()
                    results['intrinsic'] = intrinsic.squeeze(0).cpu().numpy()
                
                if predict_depth:
                    depth_map, depth_conf = self.model.depth_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['depth_map'] = depth_map.squeeze(0).cpu().numpy()
                    results['depth_conf'] = depth_conf.squeeze(0).cpu().numpy()
                
                if predict_points:
                    point_map, point_conf = self.model.point_head(
                        aggregated_tokens_list, images_batch, ps_idx
                    )
                    results['point_map'] = point_map.squeeze(0).cpu().numpy()
                    results['point_conf'] = point_conf.squeeze(0).cpu().numpy()
                
                if predict_tracks and query_points_tensor is not None:
                    if query_points_tensor.dim() == 2:
                        query_points_tensor = query_points_tensor.unsqueeze(0)
                    track, vis_score, conf_score = self.model.track_head(
                        aggregated_tokens_list, images_batch, ps_idx,
                        query_points=query_points_tensor
                    )
                    results['tracks'] = track.squeeze(0).cpu().numpy()
                    results['track_vis_score'] = vis_score.squeeze(0).cpu().numpy()
                    results['track_conf_score'] = conf_score.squeeze(0).cpu().numpy()
                
                if predict_depth and predict_cameras and predict_points:
                    point_map_from_depth = unproject_depth_map_to_point_map(
                        results['depth_map'],
                        results['extrinsic'],
                        results['intrinsic']
                    )
                    results['point_map_from_depth'] = point_map_from_depth
        
        return results

    @staticmethod
    def _estimate_gaussian_scale(points: np.ndarray, scene_center: np.ndarray) -> float:
        """
        Estimate a robust Gaussian scale for 3DGS points based on nearest-neighbor statistics.
        """
        if len(points) < 4:
            scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
            return max(scene_radius / 2000.0, 1e-4)

        sample_n = min(len(points), 2048)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=sample_n, replace=False)
        sample = torch.from_numpy(points[idx]).float()
        dist = torch.cdist(sample, sample, p=2)
        dist.fill_diagonal_(1e9)
        nn = dist.min(dim=1).values
        nn_med = float(nn.median().item())

        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        # ``nn_med`` is measured only among the sampled points.  VGGT clouds
        # are dense image surfaces, so 2D sampling density makes neighbor
        # spacing scale with sqrt(sample/full).  Without this correction the
        # 6.7M-point kitchen cloud hit the old maximum scale and rendered as
        # large blurry blobs instead of a point-aligned splat surface.
        density_correction = float(np.sqrt(sample_n / max(len(points), 1)))
        min_scale = max(scene_radius / 100000.0, 1e-6)
        max_scale = max(scene_radius / 500.0, min_scale)
        return float(np.clip(nn_med * density_correction * 1.2, min_scale, max_scale))

    def render_with_3dgs(
        self,
        ply_path: str,
        camera_config: Dict[str, Any],
        image_width: int = 704,
        image_height: int = 480,
        device: Optional[str] = None,
        near_plane: float = 0.01,
        far_plane: float = 1000.0,
    ) -> Image.Image:
        """
        Render a single frame from a VGGT reconstruction using 3D Gaussian Splatting.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pcd = fetchPly(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if points.size == 0:
            raise RuntimeError(f"No points loaded from PLY: {ply_path}")

        scene_center = points.mean(axis=0)
        scene_radius = float(np.linalg.norm(points - scene_center[None, :], axis=1).max() + 1e-8)
        scene_radius = max(scene_radius, 1e-6)

        # Normalize scene for rendering stability across large/small VGGT scales.
        points_norm = (points - scene_center[None, :]) / scene_radius
        center = np.asarray(camera_config.get("center", scene_center.tolist()), dtype=np.float32)
        center_norm = (center - scene_center) / scene_radius

        radius_raw = float(camera_config.get("radius", 1.0 * scene_radius))
        radius_norm = max(radius_raw / scene_radius, 1e-3)
        yaw_deg = float(camera_config.get("yaw", 0.0))
        pitch_deg = float(camera_config.get("pitch", 0.0))

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam_x = center_norm[0] + radius_norm * np.cos(pitch) * np.sin(yaw)
        cam_y = center_norm[1] + radius_norm * np.sin(pitch)
        cam_z = center_norm[2] + radius_norm * np.cos(pitch) * np.cos(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        fx = 0.5 * image_width / np.tan(np.deg2rad(60.0) / 2.0)
        fy = 0.5 * image_height / np.tan(np.deg2rad(45.0) / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0

        xyz = torch.from_numpy(points_norm).to(device=device, dtype=torch.float32)
        scale_value = self._estimate_gaussian_scale(points_norm, center_norm)
        scale = torch.full((xyz.shape[0], 3), scale_value, device=device, dtype=torch.float32)
        rotation = torch.zeros((xyz.shape[0], 4), device=device, dtype=torch.float32)
        rotation[:, 0] = 1.0
        opacity = torch.full((xyz.shape[0], 1), 0.95, device=device, dtype=torch.float32)
        color_tensor = torch.from_numpy(np.clip(colors, 0.0, 1.0)).to(device=device, dtype=torch.float32)
        color_tensor = _rgb_to_sh0(color_tensor)

        gaussian_params = torch.cat([xyz, opacity, scale, rotation, color_tensor], dim=-1).unsqueeze(0)
        intr = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32, device=device).unsqueeze(0)

        # Dynamic planes are more robust for arbitrary VGGT world scales.
        near_dynamic = max(near_plane, radius_norm * 0.01)
        far_dynamic = max(far_plane, radius_norm * 20.0)

        def render_candidate():
            c2w_local = _look_at_c2w_opengl(look_at=center_norm, eye=cam_pos)
            test_c2ws_local = torch.from_numpy(c2w_local).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            rgb_local, _ = gaussian_render(
                gaussian_params,
                test_c2ws_local,
                intr,
                image_width,
                image_height,
                near_plane=near_dynamic,
                far_plane=far_dynamic,
                use_checkpoint=False,
                sh_degree=0,
                bg_mode="black",
            )
            rgb_img_local = rgb_local[0, 0].clamp(-1.0, 1.0).add(1.0).div(2.0)
            gray = rgb_img_local.mean(dim=0)
            non_bg_ratio = float((gray > 0.03).float().mean().item())
            std_v = float(rgb_img_local.std().item())
            score = non_bg_ratio + 0.5 * std_v
            return rgb_img_local, score, non_bg_ratio

        rgb_img, best_score, best_non_bg_ratio = render_candidate()

        # If gsplat still fails (near-empty), fallback to deterministic point projection.
        if best_score < 0.03 or best_non_bg_ratio < 0.001:
            c2w_best = _look_at_c2w_opengl(look_at=center_norm, eye=cam_pos)

            img_fallback = np.zeros((image_height, image_width, 3), dtype=np.float32)
            depth_buf = np.full((image_height, image_width), np.inf, dtype=np.float32)

            max_points = 300000
            if points_norm.shape[0] > max_points:
                rng = np.random.default_rng(42)
                keep_idx = rng.choice(points_norm.shape[0], size=max_points, replace=False)
                proj_points = points_norm[keep_idx]
                proj_colors = colors[keep_idx]
            else:
                proj_points = points_norm
                proj_colors = colors

            w2c = np.linalg.inv(c2w_best).astype(np.float32)
            pts_h = np.concatenate(
                [proj_points, np.ones((proj_points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            cam_pts = (w2c @ pts_h.T).T[:, :3]

            best_proj_count = -1
            best_proj_payload = None
            for depth_sign in [1.0, -1.0]:
                z = cam_pts[:, 2] * depth_sign
                valid_z = z > 1e-4
                cam_pts_s = cam_pts[valid_z]
                z_s = z[valid_z]
                c_s = proj_colors[valid_z]
                if cam_pts_s.shape[0] == 0:
                    continue

                u = (fx * (cam_pts_s[:, 0] / z_s) + cx).astype(np.int32)
                v = (fy * (cam_pts_s[:, 1] / z_s) + cy).astype(np.int32)
                in_view = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
                view_count = int(in_view.sum())
                if view_count > best_proj_count:
                    best_proj_count = view_count
                    best_proj_payload = (u[in_view], v[in_view], z_s[in_view], c_s[in_view])

            if best_proj_payload is not None and best_proj_count > 0:
                u, v, z, c_proj = best_proj_payload
                order = np.argsort(z)
                u = u[order]
                v = v[order]
                z = z[order]
                c_proj = c_proj[order]

                for uu, vv, zz, cc in zip(u, v, z, c_proj):
                    if zz < depth_buf[vv, uu]:
                        depth_buf[vv, uu] = zz
                        img_fallback[vv, uu] = np.clip(cc, 0.0, 1.0)

                valid_mask = np.isfinite(depth_buf).astype(np.uint8)
                if valid_mask.any():
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate((img_fallback * 255).astype(np.uint8), kernel, iterations=1)
                    filled = cv2.dilate(valid_mask, kernel, iterations=1)
                    img_fallback[filled > 0] = dilated[filled > 0] / 255.0

                rgb_img = torch.from_numpy(img_fallback).permute(2, 0, 1).to(torch.float32)

        rgb_np = (
            rgb_img.mul(255.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return Image.fromarray(rgb_np)
