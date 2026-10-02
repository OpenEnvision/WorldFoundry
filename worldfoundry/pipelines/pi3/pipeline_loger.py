"""LoGeR (Pi3-based long-sequence geometry reconstruction) pipeline module.

Note: "loger" in this filename refers to the LoGeR model, not a logging utility.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.loger.runtime import (
    LoGeRRepresentation,
)
from worldfoundry.core.io import artifact_root_path
from worldfoundry.core.io.artifacts import render_point_cloud

from ...operators.pi3_operator import Pi3Operator
from ..pipeline_utils import PipelineABC

logger = logging.getLogger(__name__)


def _default_output_dir(name: str = "loger_output") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


def _apply_camera_delta(c2w: np.ndarray, delta: List[float]) -> np.ndarray:
    """Apply camera delta helper function."""
    dx, dy, dz, theta_x, theta_z = delta
    result = c2w.copy()

    rx = np.eye(4)
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    rx[1, 1], rx[1, 2] = cx, -sx
    rx[2, 1], rx[2, 2] = sx, cx

    rz = np.eye(4)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    rz[0, 0], rz[0, 1] = cz, -sz
    rz[1, 0], rz[1, 1] = sz, cz

    result = result @ rx @ rz
    result[0, 3] += dx
    result[1, 3] += dy
    result[2, 3] += dz
    return result


class LoGeRResult:

    """Result container representing outputs of LoGeR pipeline execution."""
    def __init__(
        self,
        depth_images: List[Image.Image],
        numpy_data: Dict[str, np.ndarray],
        camera_params: List[Dict[str, Any]],
        camera_range: Dict[str, Any],
        input_images: Optional[List[np.ndarray]] = None,
        data_type: str = "image",
    ):
        """Initialize the pipeline and configure runtime components."""
        self.depth_images = depth_images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.camera_range = camera_range
        self.input_images = input_images
        self.data_type = data_type

    def __len__(self):
        """Len for LoGeRResult."""
        return len(self.depth_images)

    def __getitem__(self, idx):
        """Getitem for LoGeRResult."""
        return {
            "depth_image": self.depth_images[idx] if idx < len(self.depth_images) else None,
            "camera_params": self.camera_params[idx] if idx < len(self.camera_params) else None,
        }

    def save(self, output_dir: Optional[str] = None) -> List[str]:
        """Save for LoGeRResult."""
        if output_dir is None:
            output_dir = _default_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        saved_files: List[str] = []

        ply_dir = os.path.join(output_dir, "point_cloud")
        os.makedirs(ply_dir, exist_ok=True)
        if "points" in self.numpy_data and "masks" in self.numpy_data and self.input_images is not None:
            try:
                from plyfile import PlyData, PlyElement
                pts = self.numpy_data["points"][0][self.numpy_data["masks"][0].astype(bool)].astype(np.float32)
                colors_all = np.stack(self.input_images, axis=0)
                col = (colors_all[self.numpy_data["masks"][0].astype(bool)] * 255).clip(0, 255).astype(np.uint8)
                vertices = np.zeros(pts.shape[0], dtype=[
                    ("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                ])
                vertices["x"], vertices["y"], vertices["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
                vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
                vertices["red"], vertices["green"], vertices["blue"] = col[:, 0], col[:, 1], col[:, 2]
                ply_path = os.path.join(ply_dir, "result.ply")
                PlyData([PlyElement.describe(vertices, "vertex")]).write(ply_path)
                saved_files.append(ply_path)
            except ImportError:
                pass

        raw_dir = os.path.join(output_dir, "raw_data")
        os.makedirs(raw_dir, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                npy_path = os.path.join(raw_dir, f"{key}.npy")
                np.save(npy_path, value)
                saved_files.append(npy_path)

        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        for i, img in enumerate(self.depth_images):
            path = os.path.join(depth_dir, f"depth_{i:04d}.png")
            img.save(path)
            saved_files.append(path)

        if self.input_images:
            rgb_dir = os.path.join(output_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            for i, arr in enumerate(self.input_images):
                path = os.path.join(rgb_dir, f"frame_{i:04d}.png")
                Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(path)
                saved_files.append(path)

        poses_dir = os.path.join(output_dir, "camera_poses")
        os.makedirs(poses_dir, exist_ok=True)
        for i, cam in enumerate(self.camera_params):
            path = os.path.join(poses_dir, f"pose_{i:04d}.json")
            with open(path, "w") as f:
                json.dump(cam, f, indent=2)
            saved_files.append(path)

        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump({"camera_range": self.camera_range}, f, indent=2)
        saved_files.append(meta_path)
        return saved_files


def _build_camera_range(camera_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build camera range helper function."""
    n = len(camera_params)
    if n == 0:
        return {}
    translations = np.array([np.array(c["camera_to_world"])[:3, 3] for c in camera_params])
    return {
        "available_view_indices": list(range(n)),
        "default_view_index": 0,
        "num_views": n,
        "translation_min": translations.min(axis=0).tolist(),
        "translation_max": translations.max(axis=0).tolist(),
    }


class LoGeRPipeline(PipelineABC):

    """Pipeline implementation for LoGeR visual generation."""
    def __init__(
        self,
        representation_model=None,
        reasoning_model: Optional[Any] = None,
        synthesis_model: Optional[Any] = None,
        operator: Optional[Pi3Operator] = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or Pi3Operator()
        self._cached_result: Optional[LoGeRResult] = None
        self._current_camera: Optional[np.ndarray] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[str] = None,
        required_components: Optional[Dict[str, str]] = None,
        mode: str = "loger",
        device: Optional[str] = None,
        weight_dtype: Optional[str] = None,
        representation_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> "LoGeRPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        path = model_path or representation_path
        if path is None:
            raise ValueError("model_path is required.")
        m = model_type or mode

        if m in ("loger", "loger_star"):
            subfolder = "LoGeR" if m == "loger" else "LoGeR_star"
            representation_model = LoGeRRepresentation.from_pretrained(
                pretrained_model_path=path, device=device, subfolder=subfolder, **kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {m}. Choose 'loger' or 'loger_star'.")
        return cls(representation_model=representation_model)

    def process(
        self,
        images: Union[str, np.ndarray, List[str], List[np.ndarray], None] = None,
        interactions: Optional[List[str]] = None,
        interval: int = -1,
    ) -> Dict[str, Any]:
        """Operator-level processing: perception preprocessing + interaction delta.

        Args:
            images:       Raw image input passed to operator.process_perception().
                          Pass None to skip perception (e.g. render-only turns).
            interactions: Navigation signal list, e.g. ["forward", "camera_r"].
                          Pass None to skip interaction processing.
            interval:     Frame-sampling interval forwarded to process_perception().

        Returns:
            Dict with keys:
                "images_data"  – preprocessed image list from operator (or None)
                "imgs_tensor"  – batched tensor ready for representation_model (or None)
                "resized_images" – list of HWC float32 arrays (or None)
                "delta"        – [dx,dy,dz,theta_x,theta_z] from process_interaction()
                                  (or None when no interactions given)
        """
        result: Dict[str, Any] = {
            "images_data": None,
            "imgs_tensor": None,
            "resized_images": None,
            "delta": None,
        }

        # ── 1. Perception preprocessing (operator) ──
        if images is not None:
            images_data = self.operator.process_perception(images, interval=interval)
            if not isinstance(images_data, list):
                images_data = [images_data]

            device = self.representation_model.device
            imgs_tensor = self.operator.images_to_tensor(images_data, device=device)
            resized_images = [
                imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy()
                for i in range(imgs_tensor.shape[1])
            ]

            result["images_data"] = images_data
            result["imgs_tensor"] = imgs_tensor
            result["resized_images"] = resized_images

        # ── 2. Interaction → delta (operator) ──
        if interactions is not None:
            self.operator.get_interaction(interactions)
            delta = self.operator.process_interaction()
            result["delta"] = delta

        return result

    def _run_inference(
        self,
        processed: Dict[str, Any],
        **kwargs,
    ) -> LoGeRResult:
        """Run representation model with already-processed operator outputs."""
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        imgs_tensor = processed["imgs_tensor"]
        resized_images = processed["resized_images"]

        data = {
            "images": imgs_tensor,
            "conf_threshold": kwargs.get("conf_threshold", 0.1),
            "edge_rtol": kwargs.get("edge_rtol", 0.03),
        }

        _LOGER_KEYS = (
            "window_size", "overlap_size", "num_iterations",
            "sim3", "se3", "reset_every",
            "turn_off_ttt", "turn_off_swa", "sim3_scale_mode",
        )
        for k in _LOGER_KEYS:
            if k in kwargs:
                data[k] = kwargs[k]

        conditions_path = kwargs.get("conditions_path")
        if conditions_path is not None and os.path.exists(conditions_path):
            import torch as _torch
            cond_data = np.load(conditions_path, allow_pickle=True)
            if "poses" in cond_data:
                data["poses"] = _torch.from_numpy(cond_data["poses"]).float().unsqueeze(0)
            if "depths" in cond_data:
                data["depths"] = _torch.from_numpy(cond_data["depths"]).float().unsqueeze(0)
            if "intrinsics" in cond_data:
                data["intrinsics"] = _torch.from_numpy(cond_data["intrinsics"]).float().unsqueeze(0)

        results = self.representation_model.get_representation(data)

        # ── Build depth images ──
        depth_images = []
        depth_maps = results.get("depth_map")
        if depth_maps is not None:
            depth_b0 = depth_maps[0]
            if depth_b0.ndim == 2:
                depth_b0 = depth_b0[np.newaxis, ...]
            for i in range(depth_b0.shape[0]):
                d = depth_b0[i].astype(np.float64)
                d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
                depth_images.append(Image.fromarray((d_norm * 255).astype(np.uint8), mode="L"))

        # ── Build camera params ──
        camera_params = []
        cam_poses = results.get("camera_poses")
        if cam_poses is not None:
            for i in range(cam_poses[0].shape[0]):
                camera_params.append({"camera_to_world": cam_poses[0][i].tolist()})

        camera_range = _build_camera_range(camera_params)

        return LoGeRResult(
            depth_images=depth_images,
            numpy_data=results,
            camera_params=camera_params,
            camera_range=camera_range,
            input_images=resized_images,
            data_type="image",
        )

    def render_view(
        self,
        result: Optional[LoGeRResult] = None,
        camera_view=None,
        camera_to_world: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """Render view for LoGeRPipeline."""
        res = result or self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run inference first via __call__().")
        if "points" not in res.numpy_data or "masks" not in res.numpy_data:
            raise RuntimeError("Result does not contain point cloud data.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(res.input_images, axis=0) if res.input_images else None
        if colors_all is None:
            raise RuntimeError("No input images in result for coloring.")

        pts = pts_all[masks].astype(np.float64)
        cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)

        if camera_to_world is not None:
            c2w = np.array(camera_to_world, dtype=np.float64)
        elif isinstance(camera_view, int):
            c2w = np.array(res.camera_params[camera_view]["camera_to_world"], dtype=np.float64)
        elif isinstance(camera_view, (list, tuple)):
            base = np.array(res.camera_params[0]["camera_to_world"], dtype=np.float64)
            c2w = _apply_camera_delta(base, camera_view)
        else:
            c2w = np.array(res.camera_params[0]["camera_to_world"], dtype=np.float64)

        h = pts_all.shape[1] if pts_all.ndim >= 3 else 480
        w = pts_all.shape[2] if pts_all.ndim >= 4 else 640
        if pts_all.ndim >= 4:
            h, w = pts_all.shape[1], pts_all.shape[2]

        return render_point_cloud(pts, cols, c2w, h, w)

    def _render_trajectory(self, n_interp: int = 15, **kwargs) -> List[Image.Image]:

        """Render trajectory for LoGeRPipeline."""
        res = self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run reconstruction first.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(res.input_images, axis=0)
        pts = pts_all[masks].astype(np.float64)
        cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)
        h, w = pts_all.shape[1], pts_all.shape[2]

        c2ws = [np.array(c["camera_to_world"], dtype=np.float64) for c in res.camera_params]

        frames = []
        for vi in range(len(c2ws) - 1):
            for j in range(n_interp):
                t = j / n_interp
                c2w = c2ws[vi] * (1 - t) + c2ws[vi + 1] * t
                frames.append(render_point_cloud(pts, cols, c2w, h, w))
        frames.append(render_point_cloud(pts, cols, c2ws[-1], h, w))

        return frames

    def __call__(
        self,
        images: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        videos: Optional[Union[str, List[str]]] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        task_type: str = "reconstruction",
        interactions: Optional[List[str]] = None,
        camera_view=None,
        visualize_ops: bool = True,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        visual_input = videos or video_path or images or image_path

        # Unified pre-step: if new visual input is provided, reconstruct and cache scene first.
        if visual_input is not None:
            processed = self.process(images=visual_input, interval=kwargs.get("interval", -1))
            result = self._run_inference(processed, **kwargs)
            self._cached_result = result
            if result.camera_params:
                self._current_camera = np.array(result.camera_params[0]["camera_to_world"])

        if task_type == "reconstruction":
            if self._cached_result is None:
                raise ValueError("images is required for task_type='reconstruction'.")
            return self._cached_result

        elif task_type == "render_view":
            if self._cached_result is None:
                raise RuntimeError("No cached scene. Run 'reconstruction' first.")

            if interactions is not None:
                n_move = kwargs.get("frames_per_interaction", 30)
                n_hold = kwargs.get("hold_frames", 10)
                frames = []
                if self._current_camera is None:
                    self._current_camera = np.array(
                        self._cached_result.camera_params[0]["camera_to_world"]
                    )
                for sig in interactions:
                    hold_img = self.render_view(camera_to_world=self._current_camera)
                    frames.extend([hold_img] * n_hold)

                    # operator 层：单条交互信号 → delta
                    processed = self.process(interactions=[sig])
                    delta = processed["delta"]
                    sub_delta = [d / n_move for d in delta]

                    for _ in range(n_move):
                        self._current_camera = _apply_camera_delta(self._current_camera, sub_delta)
                        frames.append(self.render_view(camera_to_world=self._current_camera))

                hold_img = self.render_view(camera_to_world=self._current_camera)
                frames.extend([hold_img] * n_hold)
                return frames

            return self.render_view(camera_view=camera_view)

        elif task_type == "render_trajectory":
            if self._cached_result is None:
                raise RuntimeError("No cached scene. Provide images or run 'reconstruction' first.")
            return self._render_trajectory(**kwargs)

        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. "
                "Choose 'reconstruction', 'render_view', or 'render_trajectory'."
            )

    def stream(
        self,
        images: Optional[Union[str, np.ndarray, List[str], List[np.ndarray]]] = None,
        videos: Optional[Union[str, List[str]]] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        task_type: str = "render_view",
        interactions: Optional[Union[str, List[str]]] = None,
        camera_view=None,
        visualize_ops: bool = True,
        **kwargs,
    ) -> Image.Image:
        """Stream visual generation outputs chunk by chunk."""
        visual_input = videos or video_path or images or image_path

        # ── 有新输入：先做重建 ──
        if visual_input is not None:
            logger.info("--- Stream: reconstructing scene ---")
            processed = self.process(images=visual_input, interval=kwargs.get("interval", -1))
            result = self._run_inference(processed, **kwargs)
            self._cached_result = result
            if result.camera_params:
                self._current_camera = np.array(result.camera_params[0]["camera_to_world"])

        if self._cached_result is None:
            raise RuntimeError(
                "No scene available. Provide images/videos on the first stream() call."
            )

        if self._current_camera is None:
            if not self._cached_result.camera_params:
                raise RuntimeError("No camera parameters available in cached result.")
            self._current_camera = np.array(
                self._cached_result.camera_params[0]["camera_to_world"]
            )

        # ── 无交互：直接渲染当前视角 ──
        if interactions is None:
            return self.render_view(
                result=self._cached_result,
                camera_view=camera_view,
                camera_to_world=self._current_camera,
            )

        # ── 有交互：operator 层处理，更新相机，渲染 ──
        if isinstance(interactions, str):
            interactions = [interactions]

        processed = self.process(interactions=interactions)
        delta = processed["delta"]

        self._current_camera = _apply_camera_delta(self._current_camera, delta)

        return self.render_view(
            result=self._cached_result,
            camera_to_world=self._current_camera,
        )


__all__ = ["LoGeRPipeline", "LoGeRResult"]
