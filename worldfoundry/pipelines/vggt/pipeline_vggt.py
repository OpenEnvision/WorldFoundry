"""Vggt visual generation pipeline module."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.runtime import (
    VGGTRepresentation,
    _opencv_w2c_to_opengl_c2w,
    _opencv_world_to_opengl,
)
from worldfoundry.core.io import artifact_root_path, write_video
from worldfoundry.core.media.artifacts import depths_to_pil_images

from ...base_models.three_dimensions.point_clouds.ply_io import (
    write_point_cloud as storePly,
)
from ...operators.vggt_operator import VGGTOperator
from ..pipeline_utils import PipelineABC


def _default_output_dir(name: str = "vggt_output") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


class VGGTResult:
    """Container class for VGGT inference results."""
    
    def __init__(
        self,
        images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        data_type: str = "image"
    ):
        """Initialize the pipeline and configure runtime components."""
        self.images = images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.data_type = data_type
    
    def __len__(self):
        """Len for VGGTResult."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Getitem for VGGTResult."""
        return {
            'image': self.images[idx],
            'camera_params': self.camera_params[idx] if idx < len(self.camera_params) else None,
            'numpy_data': {
                k: v[idx]
                if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == len(self.images)
                else v
                for k, v in self.numpy_data.items()
            }
        }
    
    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """Save VGGT results to files."""
        if output_dir is None:
            output_dir = _default_output_dir()
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []
        
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        for i, img in enumerate(self.images):
            img_path = os.path.join(vis_dir, f"result_{i:04d}.png")
            img.save(img_path)
            saved_files.append(img_path)
        
        np_dir = os.path.join(output_dir, "numpy")
        os.makedirs(np_dir, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                np_path = os.path.join(np_dir, f"{key}.npy")
                np.save(np_path, value)
                saved_files.append(np_path)
        
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        for i, camera_param in enumerate(self.camera_params):
            json_path = os.path.join(json_dir, f"camera_{i:04d}.json")
            with open(json_path, 'w') as f:
                json.dump(camera_param, f, indent=2)
            saved_files.append(json_path)
        
        return saved_files


class VGGTPipeline(PipelineABC):
    """Pipeline for VGGT 3D scene reconstruction."""
    
    def __init__(
        self,
        representation_model: Optional[VGGTRepresentation] = None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[VGGTOperator] = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or VGGTOperator()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, Dict[str, Any]]] = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        representation_path: Optional[str] = None,
        reasoning_path: Optional[str] = None,
        synthesis_path: Optional[str] = None,
        **kwargs
    ) -> 'VGGTPipeline':
        """Load the pipeline from pretrained checkpoints and configurations."""
        component_options = dict(required_components or {})
        if isinstance(model_path, dict):
            component_options.update(model_path)
            model_path = component_options.pop("model_path", None)
        representation_path = component_options.pop("representation_path", representation_path)
        reasoning_path = component_options.pop("reasoning_path", reasoning_path)
        synthesis_path = component_options.pop("synthesis_path", synthesis_path)
        kwargs = cls._strip_framework_loading_options({**component_options, **kwargs})
        representation_path = representation_path or model_path
        if representation_path is None:
            raise ValueError("VGGTPipeline.from_pretrained requires model_path or representation_path.")

        representation_model = VGGTRepresentation.from_pretrained(
            pretrained_model_path=representation_path,
            device=device,
            **kwargs
        )
        reasoning_model = None
        synthesis_model = None
        return cls(
            representation_model=representation_model,
            reasoning_model=reasoning_model,
            synthesis_model=synthesis_model,
        )
    
    def process(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> VGGTResult:
        """Process and normalize input arguments and conditions for inference."""
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        images_data = self.operator.process_perception(input_)
        if not isinstance(images_data, list):
            images_data = [images_data]
        
        if interaction is None:
            interaction_dict = {
                'predict_cameras': True,
                'predict_depth': True,
                'predict_points': True,
                'predict_tracks': False,
            }
        elif isinstance(interaction, str):
            self.operator.get_interaction(interaction)
            interaction_dict = self.operator.process_interaction()
        else:
            interaction_dict = interaction
        
        data = {
            'images': images_data,
            'predict_cameras': interaction_dict.get('predict_cameras', True),
            'predict_depth': interaction_dict.get('predict_depth', True),
            'predict_points': interaction_dict.get('predict_points', True),
            'predict_tracks': interaction_dict.get('predict_tracks', False),
            'query_points': kwargs.get('query_points', None),
            'preprocess_mode': kwargs.get('preprocess_mode', 'crop'),
            'resolution': kwargs.get('resolution', 518),
        }
        
        results = self.representation_model.get_representation(data)
        
        numpy_data = {}
        for key in ['extrinsic', 'intrinsic', 'depth_map', 'depth_conf', 
                   'point_map', 'point_conf', 'point_map_from_depth',
                   'tracks', 'track_vis_score', 'track_conf_score']:
            if key in results:
                numpy_data[key] = results[key]
        
        camera_params = []
        if 'extrinsic' in results and 'intrinsic' in results:
            num_images = results['extrinsic'].shape[0] if results['extrinsic'].ndim > 2 else 1
            for i in range(num_images):
                if results['extrinsic'].ndim > 2:
                    extrinsic = results['extrinsic'][i].tolist()
                    intrinsic = results['intrinsic'][i].tolist()
                else:
                    extrinsic = results['extrinsic'].tolist()
                    intrinsic = results['intrinsic'].tolist()
                camera_params.append({
                    'extrinsic': extrinsic,
                    'intrinsic': intrinsic,
                })
        
        return_visualization = kwargs.get('return_visualization', True)
        images = []
        
        if return_visualization and 'depth_map' in results:
            depth_maps = results['depth_map']
            if depth_maps.ndim == 2:
                depth_maps = depth_maps[np.newaxis, ...]
            images.extend(depths_to_pil_images(depth_maps, mode="grayscale"))
        else:
            for img_data in images_data:
                if isinstance(img_data, np.ndarray):
                    img_uint8 = (img_data * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_uint8)
                    images.append(img_pil)
        
        return VGGTResult(
            images=images,
            numpy_data=numpy_data,
            camera_params=camera_params,
            data_type="image"
        )

    @staticmethod
    def _to_uint8_rgb(frame: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """To uint8 rgb for VGGTPipeline."""
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"))
        else:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 255.0)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])

    def _export_video(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        output_path: str,
        fps: int = 12,
    ) -> str:
        """Export video for VGGTPipeline."""
        if len(frames) == 0:
            raise RuntimeError("No frames to export.")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frames_u8 = [self._to_uint8_rgb(f) for f in frames]

        # Some encoders/players can fail with a single-frame mp4.
        if len(frames_u8) == 1:
            frames_u8 = [frames_u8[0], frames_u8[0]]

        # Save first frame as a simple preview image.
        first_frame_path = os.path.join(os.path.dirname(output_path), "first_frame.png")
        Image.fromarray(frames_u8[0]).save(first_frame_path)

        write_video(frames_u8, output_path, fps=fps)
        return output_path

    @staticmethod
    def _normalize_interaction_sequence(
        interaction: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """Normalize interaction sequence for VGGTPipeline."""
        if interaction is None:
            return []
        if isinstance(interaction, str):
            return [interaction]
        return [str(sig) for sig in interaction if str(sig).strip()]

    @staticmethod
    def _apply_camera_view_to_camera_cfg(
        camera_cfg: Dict[str, Any],
        camera_view: Optional[List[float]],
        camera_range: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        camera_view: [dx, dy, dz, theta_x, theta_z]
        dx,dy,dz: center offset in world space
        theta_x: pitch offset (deg)
        theta_z: yaw offset (deg)
        """
        if camera_view is None:
            return camera_cfg

        if len(camera_view) != 5:
            raise ValueError(f"camera_view must be a 5D vector [dx,dy,dz,theta_x,theta_z], got {camera_view}")

        dx, dy, dz, theta_x, theta_z = camera_view
        center = np.asarray(camera_cfg.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
        center = center + np.array([dx, dy, dz], dtype=np.float32)

        yaw = float(camera_cfg.get("yaw", 0.0) + theta_z)
        pitch = float(camera_cfg.get("pitch", 0.0) + theta_x)

        if camera_range is not None:
            yaw = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
            pitch = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))

        camera_cfg["center"] = center.tolist()
        camera_cfg["yaw"] = yaw
        camera_cfg["pitch"] = pitch
        return camera_cfg

    @staticmethod
    def _resize_colors_to_pointmap(
        colors: List[np.ndarray],
        n_views: int,
        height: int,
        width: int,
    ) -> List[np.ndarray]:
        """Resize colors to pointmap for VGGTPipeline."""
        resized: List[np.ndarray] = []
        for i in range(n_views):
            src = np.asarray(colors[min(i, len(colors) - 1)], dtype=np.float32)
            if src.max() > 1.0:
                src = src / 255.0
            if src.shape[0] != height or src.shape[1] != width:
                src = cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)
            resized.append(np.clip(src, 0.0, 1.0))
        return resized

    def reconstruct_ply(
        self,
        input_: Union[str, np.ndarray, List[str], List[np.ndarray]],
        ply_path: Optional[str] = None,
        interaction: Optional[Union[str, Dict[str, Any]]] = None,
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
    ) -> Dict[str, Any]:
        """
        Stage 1: reconstruct colored point cloud PLY and estimate camera range.

        Returns:
            dict with keys:
            - ply_path
            - camera_range
            - default_camera
        """
        # For VGGT, reconstruction requires cameras, depth, and points.
        # We bypass interaction strings here and directly request these predictions.
        interaction_dict: Dict[str, Any] = {
            "predict_cameras": True,
            "predict_depth": True,
            "predict_points": True,
            "predict_tracks": False,
        }

        result = self.process(
            input_=input_,
            interaction=interaction_dict,
            return_visualization=False,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        if "point_map" not in result.numpy_data:
            raise RuntimeError("VGGT output does not contain point_map.")

        point_map = np.asarray(result.numpy_data["point_map"])
        if point_map.ndim == 3:
            point_map = point_map[None, ...]
        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise RuntimeError(f"Unexpected point_map shape: {point_map.shape}")

        point_conf = result.numpy_data.get("point_conf", None)
        if point_conf is None:
            point_conf = np.ones(point_map.shape[:3], dtype=np.float32)
        else:
            point_conf = np.asarray(point_conf)
            if point_conf.ndim == 2:
                point_conf = point_conf[None, ...]

        source_colors = self.operator.process_perception(input_)
        if not isinstance(source_colors, list):
            source_colors = [source_colors]

        n_views, h, w, _ = point_map.shape
        color_maps = self._resize_colors_to_pointmap(source_colors, n_views, h, w)

        points_flat = point_map.reshape(-1, 3)
        conf_flat = point_conf.reshape(-1)
        colors_flat = np.concatenate([c.reshape(-1, 3) for c in color_maps], axis=0)

        valid = np.isfinite(points_flat).all(axis=1) & np.isfinite(colors_flat).all(axis=1)
        valid &= conf_flat >= point_conf_threshold

        # VGGT point maps are world-space but retain the normalized first
        # camera's OpenCV axes (right/down/forward).  Persist an OpenGL-style
        # right/up/backward cloud so Viser and the stage-2 orbit renderer share
        # one proper, non-reflected coordinate frame.
        points = _opencv_world_to_opengl(points_flat[valid].astype(np.float32))
        colors = np.clip(colors_flat[valid], 0.0, 1.0)
        if points.shape[0] == 0:
            raise RuntimeError("No valid points after confidence filtering.")

        if ply_path is None:
            output_dir = _default_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            ply_path = os.path.join(output_dir, "pointcloud.ply")
        else:
            if not ply_path.endswith(".ply"):
                os.makedirs(ply_path, exist_ok=True)
                ply_path = os.path.join(ply_path, "pointcloud.ply")
            else:
                os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

        rgb_uint8 = (colors * 255.0).astype(np.uint8)
        storePly(ply_path, points, rgb_uint8)

        center = points.mean(axis=0)
        dists = np.linalg.norm(points - center[None, :], axis=1)
        # A few low-confidence boundary points can be several times farther
        # than the reconstructed scene.  Use a robust extent for navigation;
        # retain a generous upper bound without forcing the initial camera to
        # frame those outliers.
        radius = float(np.percentile(dists, 99.0) + 1e-6)

        camera_range = {
            "center": center.tolist(),
            "radius_min": max(radius * 0.2, 1e-3),
            "radius_max": radius * 3.0,
            "yaw_min": -180.0,
            "yaw_max": 180.0,
            "pitch_min": -75.0,
            "pitch_max": 75.0,
        }

        default_radius = radius
        default_yaw = 0.0
        default_pitch = 0.0
        extrinsics = result.numpy_data.get("extrinsic")
        if extrinsics is not None:
            try:
                first_c2w = _opencv_w2c_to_opengl_c2w(np.asarray(extrinsics)[0])
                offset = first_c2w[:3, 3] - center
                source_radius = float(np.linalg.norm(offset))
                if np.isfinite(source_radius) and source_radius > 1e-6:
                    # Pull back from the recovered source center enough to
                    # frame the scene while retaining its verified yaw/pitch.
                    default_radius = float(
                        np.clip(source_radius * 1.75, camera_range["radius_min"], camera_range["radius_max"])
                    )
                    default_yaw = float(np.degrees(np.arctan2(offset[0], offset[2])))
                    default_pitch = float(
                        np.degrees(np.arcsin(np.clip(offset[1] / source_radius, -1.0, 1.0)))
                    )
            except (ValueError, np.linalg.LinAlgError, IndexError):
                pass

        default_camera = {
            "center": center.tolist(),
            "radius": default_radius,
            "yaw": default_yaw,
            "pitch": default_pitch,
        }

        return {
            "ply_path": ply_path,
            "camera_range": camera_range,
            "default_camera": default_camera,
        }

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
        Thin wrapper: delegate 3DGS rendering to VGGTRepresentation.
        """
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")
        return self.representation_model.render_with_3dgs(
            ply_path=ply_path,
            camera_config=camera_config,
            image_width=image_width,
            image_height=image_height,
            device=device,
            near_plane=near_plane,
            far_plane=far_plane,
        )

    def render_orbit_video_with_3dgs(
        self,
        ply_path: str,
        base_camera_config: Dict[str, Any],
        num_frames: int = 24,
        yaw_step: float = 6.0,
        image_width: int = 704,
        image_height: int = 480,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        """Render orbit video with 3dgs for VGGTPipeline."""
        frames: List[Image.Image] = []
        center = base_camera_config.get("center")
        radius = float(base_camera_config.get("radius", 4.0))
        base_yaw = float(base_camera_config.get("yaw", 0.0))
        pitch = float(base_camera_config.get("pitch", 0.0))

        for i in range(num_frames):
            camera_config = {
                "center": center,
                "radius": radius,
                "yaw": base_yaw + i * yaw_step,
                "pitch": pitch,
            }
            frames.append(
                self.render_with_3dgs(
                    ply_path=ply_path,
                    camera_config=camera_config,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        if output_path is not None and len(frames) > 0:
            self._export_video(frames, output_path, fps=fps)
        return frames

    @staticmethod
    def _apply_interaction_to_camera(
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 22.0,
        pitch_step: float = 15.0,
        zoom_factor: float = 0.88,
    ) -> Dict[str, Any]:
        """
        Map interaction token to camera delta. Uses unified 3D schema (forward, left, camera_*)
        and legacy names (move_left, zoom_in). Default steps are large for visible motion.
        """
        yaw = float(camera_cfg.get("yaw", 0.0))
        pitch = float(camera_cfg.get("pitch", 0.0))
        radius = float(camera_cfg.get("radius", 4.0))
        sig = interaction.strip().lower()

        # Yaw (left/right)
        if sig in ["move_left", "rotate_left", "left", "camera_l"]:
            yaw -= yaw_step
        elif sig in ["move_right", "rotate_right", "right", "camera_r"]:
            yaw += yaw_step
        elif sig in ["camera_ul"]:
            yaw -= yaw_step
            pitch += pitch_step
        elif sig in ["camera_ur"]:
            yaw += yaw_step
            pitch += pitch_step
        elif sig in ["camera_dl"]:
            yaw -= yaw_step
            pitch -= pitch_step
        elif sig in ["camera_dr"]:
            yaw += yaw_step
            pitch -= pitch_step
        # Pitch (up/down)
        elif sig in ["move_up", "camera_up"]:
            pitch += pitch_step
        elif sig in ["move_down", "camera_down"]:
            pitch -= pitch_step
        # Radius (forward/backward, zoom)
        elif sig in ["forward", "zoom_in", "camera_zoom_in"]:
            radius *= zoom_factor
        elif sig in ["backward", "zoom_out", "camera_zoom_out"]:
            radius /= zoom_factor
        elif sig == "forward_left":
            yaw -= yaw_step
            radius *= zoom_factor
        elif sig == "forward_right":
            yaw += yaw_step
            radius *= zoom_factor
        elif sig == "backward_left":
            yaw -= yaw_step
            radius /= zoom_factor
        elif sig == "backward_right":
            yaw += yaw_step
            radius /= zoom_factor

        camera_cfg["yaw"] = max(camera_range["yaw_min"], min(camera_range["yaw_max"], yaw))
        camera_cfg["pitch"] = max(camera_range["pitch_min"], min(camera_range["pitch_max"], pitch))
        camera_cfg["radius"] = max(camera_range["radius_min"], min(camera_range["radius_max"], radius))
        return camera_cfg

    def apply_interaction_to_camera(
        self,
        camera_cfg: Dict[str, Any],
        interaction: str,
        camera_range: Dict[str, Any],
        yaw_step: float = 22.0,
        pitch_step: float = 15.0,
        zoom_factor: float = 0.88,
    ) -> Dict[str, Any]:
        """
        Public wrapper for camera update with interaction signals (unified schema + legacy names).
        """
        return self._apply_interaction_to_camera(
            camera_cfg=camera_cfg,
            interaction=interaction,
            camera_range=camera_range,
            yaw_step=yaw_step,
            pitch_step=pitch_step,
            zoom_factor=zoom_factor,
        )

    def render_interaction_video_with_3dgs(
        self,
        ply_path: str,
        camera_range: Dict[str, Any],
        base_camera_config: Dict[str, Any],
        interaction_sequence: List[str],
        image_width: int = 704,
        image_height: int = 480,
        frames_per_interaction: int = 1,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> List[Image.Image]:
        """Render interaction video with 3dgs for VGGTPipeline."""
        frames: List[Image.Image] = []
        camera_cfg = {
            "center": base_camera_config.get("center", camera_range["center"]),
            "radius": float(base_camera_config.get("radius", 4.0)),
            "yaw": float(base_camera_config.get("yaw", 0.0)),
            "pitch": float(base_camera_config.get("pitch", 0.0)),
        }

        steps = max(int(frames_per_interaction), 1)
        for sig in interaction_sequence:
            # One interaction token describes one complete action.  Multiple
            # video frames interpolate that action; they must not apply the
            # full 22-degree/0.88 delta repeatedly (which produced 220-degree
            # flips and extreme zoom for the default ten frames).
            for _ in range(steps):
                camera_cfg = self._apply_interaction_to_camera(
                    camera_cfg,
                    sig,
                    camera_range,
                    yaw_step=22.0 / steps,
                    pitch_step=15.0 / steps,
                    zoom_factor=0.88 ** (1.0 / steps),
                )
                frames.append(
                    self.render_with_3dgs(
                        ply_path=ply_path,
                        camera_config=camera_cfg,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

        if output_path is not None and len(frames) > 0:
            self._export_video(frames, output_path, fps=fps)
        return frames

    def run_two_stage_3dgs_video(
        self,
        image_path: Union[str, np.ndarray, List[str], List[np.ndarray]],
        interactions: Optional[Union[str, List[str]]] = None,
        frames_per_interaction: int = 10,
        output_dir: Optional[str] = None,
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
        camera_radius: Optional[float] = None,
        camera_yaw: Optional[float] = None,
        camera_pitch: Optional[float] = None,
        camera_view: Optional[List[float]] = None,
        image_width: int = 704,
        image_height: int = 480,
        output_name: str = "vggt_3dgs_demo.mp4",
        fps: int = 12,
    ) -> str:
        """Run two stage 3dgs video for VGGTPipeline."""
        output_dir = output_dir or _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        recon_info = self.reconstruct_ply(
            input_=image_path,
            ply_path=output_dir,
            interaction=None,
            point_conf_threshold=point_conf_threshold,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        default_camera = recon_info["default_camera"]
        base_camera = {
            "center": camera_range["center"],
            "radius": float(camera_radius if camera_radius is not None else default_camera["radius"]),
            "yaw": float(default_camera.get("yaw", 0.0) if camera_yaw is None else camera_yaw),
            "pitch": float(default_camera.get("pitch", 0.0) if camera_pitch is None else camera_pitch),
        }

        # Apply high-level 5D camera_view if provided.
        base_camera = self._apply_camera_view_to_camera_cfg(
            camera_cfg=base_camera,
            camera_view=camera_view,
            camera_range=camera_range,
        )

        output_video_path = os.path.join(output_dir, output_name)
        interaction_sequence = self._normalize_interaction_sequence(interactions)
        if interaction_sequence:
            self.render_interaction_video_with_3dgs(
                ply_path=ply_path,
                camera_range=camera_range,
                base_camera_config=base_camera,
                interaction_sequence=interaction_sequence,
                image_width=image_width,
                image_height=image_height,
                frames_per_interaction=frames_per_interaction,
                fps=fps,
                output_path=output_video_path,
            )
        else:
            self.render_orbit_video_with_3dgs(
                ply_path=ply_path,
                base_camera_config=base_camera,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        return output_video_path

    def run_stage2_3dgs_video_from_reconstruction(
        self,
        recon_info: Dict[str, Any],
        interactions: Optional[Union[str, List[str]]] = None,
        frames_per_interaction: int = 10,
        output_dir: Optional[str] = None,
        camera_radius: Optional[float] = None,
        camera_yaw: Optional[float] = None,
        camera_pitch: Optional[float] = None,
        camera_view: Optional[List[float]] = None,
        image_width: int = 704,
        image_height: int = 480,
        output_name: str = "vggt_3dgs_demo.mp4",
        fps: int = 12,
    ) -> str:
        """
        Stage 2 only: render video from existing reconstruction info.
        """
        output_dir = output_dir or _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        default_camera = recon_info["default_camera"]
        base_camera = {
            "center": camera_range["center"],
            "radius": float(camera_radius if camera_radius is not None else default_camera["radius"]),
            "yaw": float(default_camera.get("yaw", 0.0) if camera_yaw is None else camera_yaw),
            "pitch": float(default_camera.get("pitch", 0.0) if camera_pitch is None else camera_pitch),
        }

        base_camera = self._apply_camera_view_to_camera_cfg(
            camera_cfg=base_camera,
            camera_view=camera_view,
            camera_range=camera_range,
        )

        output_video_path = os.path.join(output_dir, output_name)
        interaction_sequence = self._normalize_interaction_sequence(interactions)
        if interaction_sequence:
            self.render_interaction_video_with_3dgs(
                ply_path=ply_path,
                camera_range=camera_range,
                base_camera_config=base_camera,
                interaction_sequence=interaction_sequence,
                image_width=image_width,
                image_height=image_height,
                frames_per_interaction=frames_per_interaction,
                fps=fps,
                output_path=output_video_path,
            )
        else:
            self.render_orbit_video_with_3dgs(
                ply_path=ply_path,
                base_camera_config=base_camera,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                output_path=output_video_path,
            )
        return output_video_path

    def run_two_stage_3dgs_stream_cli(
        self,
        image_path: Union[str, np.ndarray, List[str], List[np.ndarray]],
        output_dir: Optional[str] = None,
        point_conf_threshold: float = 0.2,
        resolution: int = 518,
        preprocess_mode: str = "crop",
        image_width: int = 704,
        image_height: int = 480,
        fps: int = 12,
        output_name: str = "vggt_stream_demo.mp4",
    ) -> str:
        """Run two stage 3dgs stream cli for VGGTPipeline."""
        output_dir = output_dir or _default_output_dir("vggt_stream_output")
        os.makedirs(output_dir, exist_ok=True)
        recon_info = self.reconstruct_ply(
            input_=image_path,
            ply_path=output_dir,
            interaction=None,
            point_conf_threshold=point_conf_threshold,
            resolution=resolution,
            preprocess_mode=preprocess_mode,
        )

        # Unified 3D schema (same as operator); legacy names still supported in _apply_interaction_to_camera
        available_interactions = [
            "forward", "backward", "left", "right",
            "forward_left", "forward_right", "backward_left", "backward_right",
            "camera_up", "camera_down", "camera_l", "camera_r",
            "camera_ul", "camera_ur", "camera_dl", "camera_dr",
            "camera_zoom_in", "camera_zoom_out",
        ]

        ply_path = recon_info["ply_path"]
        camera_range = recon_info["camera_range"]
        camera_cfg = dict(recon_info["default_camera"])

        print("Stage-1 reconstruction done.")
        print(f"PLY saved to: {ply_path}")
        print("Camera range:", camera_range)
        print("Default camera:", camera_cfg)
        print("\nAvailable interactions:")
        for i, interaction in enumerate(available_interactions):
            print(f"  {i + 1}. {interaction}")
        print("Tips:")
        print("  - Input multiple interactions separated by comma (e.g., 'move_left,zoom_in')")
        print("  - Input 'n' or 'q' to stop and export video")

        all_frames: List[np.ndarray] = []
        first_frame = self.render_with_3dgs(
            ply_path=ply_path,
            camera_config=camera_cfg,
            image_width=image_width,
            image_height=image_height,
        )
        all_frames.append(np.array(first_frame))

        turn_idx = 0
        print("\n--- VGGT Interactive Stream Started ---")
        while True:
            interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()
            if interaction_input in ["n", "q"]:
                print("Stopping interaction loop...")
                break

            current_signal = [s.strip() for s in interaction_input.split(",") if s.strip()]
            invalid = [s for s in current_signal if s not in available_interactions]
            if invalid:
                print(f"Invalid interaction(s): {invalid}")
                print(f"Please choose from: {available_interactions}")
                continue
            if not current_signal:
                print("No valid interaction provided. Please try again.")
                continue

            try:
                frames_input = input(f"[Turn {turn_idx}] Enter frame units (e.g., 1 or 2): ").strip()
                frame_units = int(frames_input)
                if frame_units <= 0:
                    print("Frame units must be a positive integer.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                continue

            for sig in current_signal:
                for _ in range(frame_units):
                    camera_cfg = self._apply_interaction_to_camera(
                        camera_cfg,
                        sig,
                        camera_range,
                        yaw_step=22.0,
                        pitch_step=15.0,
                        zoom_factor=0.88,
                    )
                    frame = self.render_with_3dgs(
                        ply_path=ply_path,
                        camera_config=camera_cfg,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    all_frames.append(np.array(frame))

            print(f"[Turn {turn_idx}] done. Total frames: {len(all_frames)}")
            print(f"Current camera: {camera_cfg}")
            turn_idx += 1

        output_video_path = os.path.join(output_dir, output_name)
        self._export_video(all_frames, output_video_path, fps=fps)
        print(f"Total frames generated: {len(all_frames)}")
        print(f"Stream video saved to: {output_video_path}")
        return output_video_path
    
    def stream(
        self,
        image_path: Optional[Union[str, List[str]]] = None,
        images: Any = None,
        interactions: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ):
        """
        Stream interface. Input: image_path or images; interactions (for fallback process).
        task_type: \"vggt_two_stage_3dgs_stream_cli\" -> yields the output video path
        as a single element; else yields image tensors.
        """
        data = images if images is not None else image_path
        if data is None:
            raise ValueError("Provide image_path or images.")
        if task_type == "vggt_two_stage_3dgs_stream_cli":
            # ``stream`` is a generator function, so a ``return <value>`` here would
            # silently discard the path inside StopIteration. Yield it instead.
            yield self.run_two_stage_3dgs_stream_cli(image_path=data, **kwargs)
            return

        result = self.process(input_=data, interaction=interactions, **kwargs)
        for img in result.images:
            yield torch.from_numpy(np.array(img))

    def run_official_scene_export(
        self,
        image_path: Union[str, List[str]],
        output_dir: Optional[str] = None,
        preprocess_mode: str = "crop",
        conf_thres: float = 3.0,
        frame_filter: str = "All",
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        show_cam: bool = True,
        prediction_mode: str = "Pointmap Regression",
        output_name: Optional[str] = None,
        **_unused: Any,
    ) -> Dict[str, str]:
        """Run official scene export for VGGTPipeline."""
        if self.representation_model is None or self.representation_model.model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        output_dir = output_dir or _default_output_dir()
        if output_name is not None and not output_name.lower().endswith(".glb"):
            output_name = os.path.splitext(output_name)[0] + ".glb"

        from .official_runtime import (
            run_official_scene_export as _run_official_scene_export,
        )

        return _run_official_scene_export(
            input_source=image_path,
            model=self.representation_model.model,
            output_dir=output_dir,
            device=self.representation_model.device,
            preprocess_mode=preprocess_mode,
            conf_thres=conf_thres,
            frame_filter=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            prediction_mode=prediction_mode,
            output_name=output_name,
        )
    
    def __call__(
        self,
        image_path: Optional[Union[str, List[str]]] = None,
        images: Any = None,
        interactions: Optional[List[str]] = None,
        camera_view: Optional[List[float]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> Union[VGGTResult, str, Dict[str, str]]:
        """
        Main call interface. Input: image_path or images; interactions; camera_view (5D, for two-stage).
        task_type: None | \"vggt_base\" -> VGGTResult; \"vggt_two_stage_3dgs\" -> output_video_path (str).
        """
        data = images if images is not None else image_path
        if data is None:
            raise ValueError("Provide image_path or images.")
        interaction = kwargs.pop("interaction", interactions)
        if task_type == "vggt_two_stage_3dgs":
            return self.run_two_stage_3dgs_video(
                image_path=data,
                interactions=interaction,
                camera_view=camera_view,
                **kwargs,
            )
        if task_type in {"vggt_official_scene_export", "vggt_official_glb", "official"}:
            return self.run_official_scene_export(
                image_path=data,
                **kwargs,
            )
        return self.process(input_=data, interaction=interaction, **kwargs)


__all__ = ["VGGTPipeline", "VGGTResult"]
