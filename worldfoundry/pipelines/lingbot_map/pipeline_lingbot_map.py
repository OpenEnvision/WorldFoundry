"""Lingbot Map visual generation pipeline module."""

from __future__ import annotations

from ...synthesis.visual_generation.memory.runtime import RuntimeMemory
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from worldfoundry.core.io.artifacts import depths_to_pil_images

from ...operators.lingbot_map_operator import LingBotMapOperator
from ...representations.point_clouds_generation.lingbot_map import LingBotMapRepresentation
from ..pipeline_utils import PipelineABC


class LingBotMapResult:
    """Container for LingBot-Map reconstruction outputs."""

    def __init__(self, numpy_data: Dict[str, Any], data_type: str = "image_sequence"):
        """Initialize the pipeline and configure runtime components."""
        self.numpy_data = numpy_data
        self.data_type = data_type
        self.images = self._preview_images()
        self.camera_params = self._camera_params()

    def _preview_images(self) -> list[Image.Image]:
        """Preview images for LingBotMapResult."""
        depth = self.numpy_data.get("depth")
        if depth is None:
            return []
        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 4 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[None]
        return depths_to_pil_images(depth_arr, mode="grayscale")

    def _camera_params(self) -> list[dict[str, Any]]:
        """Camera params for LingBotMapResult."""
        extrinsic = self.numpy_data.get("extrinsic")
        intrinsic = self.numpy_data.get("intrinsic")
        if extrinsic is None or intrinsic is None:
            return []
        extrinsic_arr = np.asarray(extrinsic)
        intrinsic_arr = np.asarray(intrinsic)
        if extrinsic_arr.ndim == 2:
            extrinsic_arr = extrinsic_arr[None]
        if intrinsic_arr.ndim == 2:
            intrinsic_arr = intrinsic_arr[None]
        return [
            {
                "extrinsic": extrinsic_arr[index].tolist(),
                "intrinsic": intrinsic_arr[index].tolist(),
            }
            for index in range(min(len(extrinsic_arr), len(intrinsic_arr)))
        ]

    def _point_cloud_arrays(self, max_points: int = 50000) -> tuple[np.ndarray, np.ndarray] | None:
        """Point cloud arrays for LingBotMapResult."""
        world_points = self.numpy_data.get("world_points")
        colors = self.numpy_data.get("images")
        if colors is None:
            colors = self.numpy_data.get("input_images")
        if world_points is not None:
            points = np.asarray(world_points, dtype=np.float32)
            if points.ndim > 2:
                points = points.reshape(-1, points.shape[-1])
            if points.shape[-1] < 3:
                return None
            points = points[:, :3]
            rgb = self._flatten_colors(colors, len(points))
        else:
            projected = self._project_depth_to_points(max_points=max_points)
            if projected is None:
                return None
            points, rgb = projected

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        rgb = rgb[finite] if len(rgb) == len(finite) else np.full((len(points), 3), 200, dtype=np.uint8)
        if len(points) == 0:
            return None
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[indices]
            rgb = rgb[indices]
        return points.astype(np.float32), rgb.astype(np.uint8)

    def _flatten_colors(self, colors: Any, target_count: int) -> np.ndarray:
        """Flatten colors for LingBotMapResult."""
        if colors is None:
            return np.full((target_count, 3), 200, dtype=np.uint8)
        array = np.asarray(colors)
        if array.ndim == 4 and array.shape[1] == 3:
            array = np.transpose(array, (0, 2, 3, 1))
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))[None]
        if array.ndim == 3 and array.shape[-1] == 3:
            array = array[None]
        if array.ndim < 4 or array.shape[-1] < 3:
            return np.full((target_count, 3), 200, dtype=np.uint8)
        flat = array[..., :3].reshape(-1, 3)
        if flat.dtype != np.uint8:
            if flat.size and np.nanmax(flat) <= 1.0:
                flat = flat * 255.0
            flat = np.clip(flat, 0, 255).astype(np.uint8)
        if len(flat) >= target_count:
            return flat[:target_count]
        pad = np.full((target_count - len(flat), 3), 200, dtype=np.uint8)
        return np.concatenate([flat, pad], axis=0)

    def _project_depth_to_points(self, max_points: int = 50000) -> tuple[np.ndarray, np.ndarray] | None:
        """Project depth to points for LingBotMapResult."""
        depth = self.numpy_data.get("depth")
        intrinsic = self.numpy_data.get("intrinsic")
        extrinsic = self.numpy_data.get("extrinsic")
        if depth is None or intrinsic is None:
            return None
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 4 and depth_arr.shape[-1] == 1:
            depth_arr = depth_arr[..., 0]
        if depth_arr.ndim == 2:
            depth_arr = depth_arr[None]
        if depth_arr.ndim != 3:
            return None
        intrinsic_arr = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic_arr.ndim == 2:
            intrinsic_arr = intrinsic_arr[None]
        extrinsic_arr = np.asarray(extrinsic, dtype=np.float32) if extrinsic is not None else None
        if extrinsic_arr is not None and extrinsic_arr.ndim == 2:
            extrinsic_arr = extrinsic_arr[None]

        color_frames = self.numpy_data.get("images")
        if color_frames is None:
            color_frames = self.numpy_data.get("input_images")
        color_arr = np.asarray(color_frames) if color_frames is not None else None
        if color_arr is not None and color_arr.ndim == 4 and color_arr.shape[1] == 3:
            color_arr = np.transpose(color_arr, (0, 2, 3, 1))
        points_all: list[np.ndarray] = []
        colors_all: list[np.ndarray] = []
        per_frame_budget = max(1, max_points // max(1, depth_arr.shape[0]))
        for frame_idx, frame_depth in enumerate(depth_arr):
            height, width = frame_depth.shape
            mask = np.isfinite(frame_depth) & (frame_depth > 0)
            valid = np.flatnonzero(mask.reshape(-1))
            if valid.size == 0:
                continue
            if valid.size > per_frame_budget:
                valid = valid[np.linspace(0, valid.size - 1, per_frame_budget, dtype=np.int64)]
            ys, xs = np.divmod(valid, width)
            z = frame_depth.reshape(-1)[valid]
            intr = intrinsic_arr[min(frame_idx, len(intrinsic_arr) - 1)]
            fx = float(intr[0, 0]) if intr.shape[0] > 0 else 0.0
            fy = float(intr[1, 1]) if intr.shape[0] > 1 else 0.0
            cx = float(intr[0, 2]) if intr.shape[1] > 2 else width / 2.0
            cy = float(intr[1, 2]) if intr.shape[1] > 2 else height / 2.0
            fx = fx if abs(fx) > 1e-6 else max(width, 1)
            fy = fy if abs(fy) > 1e-6 else max(height, 1)
            pts = np.stack(((xs.astype(np.float32) - cx) / fx * z, (ys.astype(np.float32) - cy) / fy * z, z), axis=1)
            if extrinsic_arr is not None and len(extrinsic_arr):
                ext = extrinsic_arr[min(frame_idx, len(extrinsic_arr) - 1)]
                if ext.shape == (3, 4):
                    ext_h = np.eye(4, dtype=np.float32)
                    ext_h[:3, :] = ext
                elif ext.shape == (4, 4):
                    ext_h = ext
                else:
                    ext_h = None
                if ext_h is not None:
                    camera_to_world = ext_h
                    pts = pts @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
            if color_arr is not None and color_arr.ndim == 4:
                color_frame = color_arr[min(frame_idx, color_arr.shape[0] - 1)]
                frame_rgb = color_frame.reshape(-1, color_frame.shape[-1])[valid, :3]
                if frame_rgb.dtype != np.uint8:
                    if frame_rgb.size and np.nanmax(frame_rgb) <= 1.0:
                        frame_rgb = frame_rgb * 255.0
                    frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
            else:
                frame_rgb = np.full((len(pts), 3), 200, dtype=np.uint8)
            points_all.append(pts.astype(np.float32))
            colors_all.append(frame_rgb.astype(np.uint8))
        if not points_all:
            return None
        return np.concatenate(points_all, axis=0), np.concatenate(colors_all, axis=0)

    def _save_point_cloud(self, path: Path) -> str | None:
        """Save point cloud for LingBotMapResult."""
        arrays = self._point_cloud_arrays()
        if arrays is None:
            return None
        points, colors = arrays
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
            handle.write("end_header\n")
            for point, color in zip(points, colors):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return str(path)

    def save(self, output_dir: str | Path | None = None, output_name: str = "lingbot_map_predictions.npz") -> list[str]:
        """Save for LingBotMapResult."""
        output_path = Path(output_dir or "./lingbot_map_output")
        output_path.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        npz_payload = {
            key: value
            for key, value in self.numpy_data.items()
            if isinstance(value, np.ndarray) or np.isscalar(value)
        }
        npz_path = output_path / output_name
        np.savez_compressed(npz_path, **npz_payload)
        saved.append(str(npz_path))

        point_cloud_path = self._save_point_cloud(output_path / "lingbot_map_point_cloud.ply")
        if point_cloud_path:
            saved.append(point_cloud_path)

        if self.camera_params:
            camera_path = output_path / "camera_params.json"
            camera_path.write_text(json.dumps(self.camera_params, indent=2), encoding="utf-8")
            saved.append(str(camera_path))

        if self.images:
            vis_dir = output_path / "depth_visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(self.images):
                frame_path = vis_dir / f"depth_{index:06d}.png"
                image.save(frame_path)
                saved.append(str(frame_path))
        return saved


class LingBotMapPipeline(PipelineABC):
    """WorldFoundry pipeline for LingBot-Map streaming 3D reconstruction."""

    MODEL_ID = "lingbot-map"

    def __init__(
        self,
        representation_model: Optional[LingBotMapRepresentation] = None,
        operator: Optional[LingBotMapOperator] = None,
        memory_module: Optional[RuntimeMemory] = None,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.representation_model = representation_model
        self.operator = operator or LingBotMapOperator()
        self.memory_module = memory_module or RuntimeMemory(model_id='lingbot-map')
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LingBotMapPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        options: Dict[str, Any] = {}
        checkpoint_source = pretrained_model_path if pretrained_model_path is not None else model_path
        if isinstance(checkpoint_source, dict):
            options.update(checkpoint_source)
        elif checkpoint_source is not None:
            options["pretrained_model_path"] = str(checkpoint_source)
        options.update(required_components or {})
        options.update(kwargs)
        pretrained_model_path = (
            options.pop("pretrained_model_path", None)
            or options.pop("checkpoint_path", None)
            or options.pop("representation_path", None)
            or options.pop("model_path", None)
            or options.pop("repo_root", None)
        )
        representation_model = LingBotMapRepresentation.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            device=device,
            **cls._strip_framework_loading_options(options),
        )
        return cls(
            representation_model=representation_model,
            operator=LingBotMapOperator(),
            memory_module=RuntimeMemory(model_id='lingbot-map'),
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

    def process(self, images: Any = None, interactions: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        if images is None:
            raise ValueError("LingBotMapPipeline requires images as a directory, list, tensor, PIL image, or numpy array.")
        self.operator.get_interaction(interactions)
        try:
            interaction = self.operator.process_interaction(**kwargs)
        finally:
            self.operator.delete_last_interaction()
        return {
            "images": self.operator.process_perception(images),
            **interaction,
            **kwargs,
        }

    def __call__(
        self,
        images: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Execute the complete pipeline generation flow."""
        if self.representation_model is None:
            raise RuntimeError("LingBot-Map representation model is not loaded. Use from_pretrained() first.")
        processed = self.process(images=images, interactions=interactions, **kwargs)
        result = LingBotMapResult(self.representation_model.get_representation(processed))
        self.memory_module.record(result, metadata={"type": "lingbot_map_result", "model_id": self.model_id})

        saved_files: list[str] = []
        if output_path is not None:
            output_path = Path(output_path)
            output_dir = output_path if output_path.suffix == "" else output_path.parent
            output_name = output_path.name if output_path.suffix else "lingbot_map_predictions.npz"
            saved_files = result.save(output_dir=output_dir, output_name=output_name)

        if return_dict:
            artifact_path = next((path for path in saved_files if Path(path).suffix.lower() == ".ply"), None)
            artifact_path = artifact_path or (saved_files[0] if saved_files else None)
            return {
                "model_id": self.model_id,
                "artifact_path": artifact_path,
                "point_cloud_path": artifact_path if artifact_path and Path(artifact_path).suffix.lower() == ".ply" else None,
                "saved_files": saved_files,
                "result": result,
                "numpy_data": result.numpy_data,
                "camera_params": result.camera_params,
            }
        return result
