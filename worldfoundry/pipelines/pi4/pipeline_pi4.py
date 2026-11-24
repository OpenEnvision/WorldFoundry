# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pi4 (pi^4) 4D scene geometry and dense motion pipeline."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.runtime import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_PI4_REPO_ID,
    Pi4Representation,
)
from worldfoundry.core.io import artifact_root_path
from worldfoundry.core.media.artifacts import render_point_cloud

from ...operators.pi4_operator import Pi4Operator
from ..pipeline_utils import PipelineABC


def _default_output_dir(name: str = "pi4_output") -> str:
    """Resolve a stable default output directory instead of writing to the CWD."""
    return str(artifact_root_path() / name)


def _apply_camera_delta(c2w: np.ndarray, delta: Sequence[float]) -> np.ndarray:
    """Apply a ``[dx, dy, dz, theta_x, theta_z]`` delta to a 4x4 camera-to-world matrix."""
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


def _build_camera_range(camera_params: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute camera parameter range from a list of camera-to-world pose records."""
    count = len(camera_params)
    if count == 0:
        return {}

    translations = np.array([np.array(cam["camera_to_world"])[:3, 3] for cam in camera_params])
    return {
        "available_view_indices": list(range(count)),
        "default_view_index": 0,
        "num_views": count,
        "translation_min": translations.min(axis=0).tolist(),
        "translation_max": translations.max(axis=0).tolist(),
    }


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write colored point cloud vertices to binary PLY (via ``plyfile`` when installed)."""
    vertices = np.zeros(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    try:
        from plyfile import PlyData, PlyElement

        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
    except ImportError:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with path.open("wb") as stream:
            stream.write(header)
            vertices.tofile(stream)


class Pi4Result:
    """Result container for Pi4 4D geometry and dense motion outputs."""

    def __init__(
        self,
        depth_images: list[Image.Image],
        numpy_data: dict[str, np.ndarray],
        camera_params: list[dict[str, Any]],
        camera_range: dict[str, Any],
        input_images: list[np.ndarray] | None = None,
        data_type: str = "image",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.depth_images = depth_images
        self.numpy_data = numpy_data
        self.camera_params = camera_params
        self.camera_range = camera_range
        self.input_images = input_images
        self.data_type = data_type
        self.metadata = dict(metadata or {})

    @property
    def has_tracking(self) -> bool:
        """Whether this result includes 4D dense tracking and moving-object arrays."""
        return "tracks_3d" in self.numpy_data

    def __len__(self) -> int:
        return len(self.depth_images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "depth_image": self.depth_images[idx] if idx < len(self.depth_images) else None,
            "camera_params": self.camera_params[idx] if idx < len(self.camera_params) else None,
        }

    def save(self, output_dir: str | Path | None = None) -> list[str]:
        """Export PLY point cloud, raw NumPy arrays (including 4D tracking), depth, RGB, poses, and metadata."""
        resolved_dir = Path(output_dir or _default_output_dir())
        resolved_dir.mkdir(parents=True, exist_ok=True)
        saved_files: list[str] = []

        # Point cloud (PLY)
        ply_dir = resolved_dir / "point_cloud"
        ply_dir.mkdir(parents=True, exist_ok=True)
        if "points" in self.numpy_data and "masks" in self.numpy_data and self.input_images is not None:
            points_b0 = self.numpy_data["points"][0]
            masks_b0 = self.numpy_data["masks"][0].astype(bool)
            colors = np.stack(self.input_images, axis=0)

            pts = points_b0[masks_b0].astype(np.float32)
            col = (colors[masks_b0] * 255).clip(0, 255).astype(np.uint8)
            ply_path = ply_dir / "result.ply"
            _write_binary_ply(ply_path, pts, col)
            saved_files.append(str(ply_path))

        # Raw numpy data (geometry + optional 4D tracking arrays)
        raw_dir = resolved_dir / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for key, value in self.numpy_data.items():
            if isinstance(value, np.ndarray):
                npy_path = raw_dir / f"{key}.npy"
                np.save(npy_path, value)
                saved_files.append(str(npy_path))

        # Depth map visualizations
        depth_dir = resolved_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(self.depth_images):
            depth_path = depth_dir / f"depth_{index:04d}.png"
            image.save(depth_path)
            saved_files.append(str(depth_path))

        # Input RGB frames
        if self.input_images is not None and len(self.input_images) > 0:
            rgb_dir = resolved_dir / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for index, img_arr in enumerate(self.input_images):
                img_uint8 = (img_arr * 255).clip(0, 255).astype(np.uint8)
                rgb_path = rgb_dir / f"frame_{index:04d}.png"
                Image.fromarray(img_uint8).save(rgb_path)
                saved_files.append(str(rgb_path))

        # Camera poses
        poses_dir = resolved_dir / "camera_poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        for index, cam in enumerate(self.camera_params):
            pose_path = poses_dir / f"pose_{index:04d}.json"
            pose_path.write_text(json.dumps(cam, indent=2), encoding="utf-8")
            saved_files.append(str(pose_path))

        # meta.json with camera_range and 4D metadata
        meta_payload: dict[str, Any] = {
            "camera_range": self.camera_range,
            "has_tracking": self.has_tracking,
            **self.metadata,
        }
        if "metric_scale" in self.numpy_data and self.numpy_data["metric_scale"].size > 0:
            meta_payload.setdefault("metric_scale", float(self.numpy_data["metric_scale"].reshape(-1)[0]))
        if "query_frame" in self.numpy_data and self.numpy_data["query_frame"].size > 0:
            meta_payload.setdefault("query_frame", int(self.numpy_data["query_frame"].reshape(-1)[0]))

        meta_path = resolved_dir / "meta.json"
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        saved_files.append(str(meta_path))

        return saved_files


class Pi4Pipeline(PipelineABC):
    """Pipeline for Pi4 4D metric geometry reconstruction, dense tracking, and rendering."""

    MODEL_ID = "pi4"

    def __init__(
        self,
        representation_model: Pi4Representation | None = None,
        reasoning_model: Any | None = None,
        synthesis_model: Any | None = None,
        operator: Pi4Operator | None = None,
        model_id: str | None = None,
    ) -> None:
        self.model_id = model_id or self.MODEL_ID
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.operator = operator or Pi4Operator()
        self._cached_result: Pi4Result | None = None
        self._current_camera: np.ndarray | None = None

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # model-specific loader arguments, passed by keyword
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str | None = None,
        weight_dtype: str | None = None,
        representation_path: str | Path | None = None,
        pretrained_model_path: str | Path | None = None,
        revision: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> Pi4Pipeline:
        """Load the Pi4 pipeline from a local checkpoint file/directory or Hugging Face repo."""
        del weight_dtype
        options: dict[str, Any] = {}
        checkpoint_source = (
            pretrained_model_path
            if pretrained_model_path is not None
            else (model_path if model_path is not None else representation_path)
        )
        if isinstance(checkpoint_source, Mapping):
            options.update(checkpoint_source)
        elif checkpoint_source is not None:
            options["pretrained_model_path"] = str(checkpoint_source)
        if required_components:
            options.update(required_components)
        options.update(kwargs)

        resolved_path = (
            options.pop("pretrained_model_path", None)
            or options.pop("checkpoint_path", None)
            or options.pop("representation_path", None)
            or options.pop("model_path", None)
            or options.pop("primary", None)
            or options.pop("repo_root", None)
            or DEFAULT_PI4_REPO_ID
        )
        resolved_revision = options.pop("revision", revision)
        cleaned_options = cls._strip_framework_loading_options(options)
        representation_model = Pi4Representation.from_pretrained(
            pretrained_model_path=resolved_path,
            device=device,
            revision=resolved_revision,
            **cleaned_options,
        )
        return cls(representation_model=representation_model, model_id=model_id or cls.MODEL_ID)

    def reset_for_evaluation(self) -> None:
        """Clear cached request/navigation state while keeping model weights resident."""
        self._cached_result = None
        self._current_camera = None
        self.operator.current_interaction = []

    def _run_inference(
        self,
        visual_input: Any,
        *,
        query_frame: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        max_size: int | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        interval: int | None = None,
    ) -> Pi4Result:
        """Run single-shot Pi4 geometry (and optional 4D tracking) inference."""
        if self.representation_model is None:
            raise RuntimeError("Representation model not loaded. Use from_pretrained() first.")

        images_data = self.operator.process_perception(
            visual_input,
            stride=stride,
            max_frames=max_frames,
            interval=interval,
        )
        device = self.representation_model.device
        imgs_tensor = self.operator.images_to_tensor(images_data, device=device, max_size=max_size)
        resized_images = [imgs_tensor[0, i].permute(1, 2, 0).cpu().numpy() for i in range(imgs_tensor.shape[1])]

        data: dict[str, Any] = {
            "images": imgs_tensor,
            "query_frame": query_frame,
            "conf_threshold": conf_threshold,
        }
        results = self.representation_model.get_representation(data)

        depth_images: list[Image.Image] = []
        depth_maps = results.get("depth_map")
        if depth_maps is not None:
            depth_b0 = depth_maps[0]
            if depth_b0.ndim == 2:
                depth_b0 = depth_b0[np.newaxis, ...]
            for idx in range(depth_b0.shape[0]):
                depth_slice = depth_b0[idx].astype(np.float64)
                d_min, d_max = float(np.nanmin(depth_slice)), float(np.nanmax(depth_slice))
                d_norm = np.nan_to_num((depth_slice - d_min) / (d_max - d_min + 1e-8), nan=0.0)
                d_uint8 = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
                depth_images.append(Image.fromarray(d_uint8, mode="L"))

        camera_params: list[dict[str, Any]] = []
        cam_poses = results.get("camera_poses")
        if cam_poses is not None:
            for idx in range(cam_poses[0].shape[0]):
                camera_params.append({"camera_to_world": cam_poses[0][idx].tolist()})

        camera_range = _build_camera_range(camera_params)
        height, width = imgs_tensor.shape[-2:]
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "num_frames": int(imgs_tensor.shape[1]),
            "image_size": [int(width), int(height)],
            "stride": int(stride),
            "max_frames": max_frames,
            "conf_threshold": float(conf_threshold),
            "coordinates": "metric_world",
            "camera_convention": "camera_to_world",
        }
        if query_frame is not None:
            metadata["query_frame"] = int(query_frame)

        result = Pi4Result(
            depth_images=depth_images,
            numpy_data=results,
            camera_params=camera_params,
            camera_range=camera_range,
            input_images=resized_images,
            data_type="video" if len(resized_images) > 1 else "image",
            metadata=metadata,
        )
        self._cached_result = result
        if camera_params:
            self._current_camera = np.array(camera_params[0]["camera_to_world"], dtype=np.float64)
        return result

    def render_view(
        self,
        result: Pi4Result | None = None,
        camera_view: int | Sequence[float] | None = None,
        camera_to_world: np.ndarray | None = None,
    ) -> Image.Image:
        """Render a novel or original view from the cached reconstructed point cloud."""
        res = result or self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run inference first via __call__().")

        if "points" not in res.numpy_data or "masks" not in res.numpy_data:
            raise RuntimeError("Result does not contain point cloud data.")
        if not res.input_images:
            raise RuntimeError("No input images in result for coloring.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(res.input_images, axis=0)

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

        height = pts_all.shape[1] if pts_all.ndim >= 3 else 480
        width = pts_all.shape[2] if pts_all.ndim >= 4 else 640
        return render_point_cloud(pts, cols, c2w, height, width)

    def _render_trajectory(self, n_interp: int = 15, fps: int = 15, **kwargs: Any) -> list[Image.Image]:
        """Render an interpolated camera trajectory across reconstructed views."""
        del fps, kwargs
        res = self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run reconstruction first.")

        pts_all = res.numpy_data["points"][0]
        masks = res.numpy_data["masks"][0].astype(bool)
        colors_all = np.stack(cast(list[np.ndarray], res.input_images), axis=0)
        pts = pts_all[masks].astype(np.float64)
        cols = (colors_all[masks] * 255).clip(0, 255).astype(np.uint8)
        height, width = pts_all.shape[1], pts_all.shape[2]

        c2ws = [np.array(cam["camera_to_world"], dtype=np.float64) for cam in res.camera_params]
        frames: list[Image.Image] = []
        for view_idx in range(len(c2ws) - 1):
            for step in range(n_interp):
                alpha = step / n_interp
                c2w = c2ws[view_idx] * (1 - alpha) + c2ws[view_idx + 1] * alpha
                frames.append(render_point_cloud(pts, cols, c2w, height, width))
        frames.append(render_point_cloud(pts, cols, c2ws[-1], height, width))
        return frames

    @staticmethod
    def _resolve_save_dir(output_dir: str | Path | None, output_path: str | Path | None) -> Path | None:
        if output_dir is not None:
            return Path(output_dir)
        if output_path is not None:
            path = Path(output_path)
            return path.parent if path.suffix else path
        return None

    def __call__(
        self,
        images: Any = None,
        videos: Any = None,
        video: Any = None,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        input_path: str | Path | None = None,
        task_type: str = "reconstruction",
        interactions: list[str] | None = None,
        camera_view: int | Sequence[float] | None = None,
        query_frame: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        max_size: int | None = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        output_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        visualize_ops: bool = True,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Pi4Result | Image.Image | list[Image.Image] | dict[str, Any]:
        """Execute Pi4 reconstruction, 4D tracking, view rendering, or trajectory rendering.

        Returns a structured dictionary when ``return_dict=True`` or ``output_dir`` is provided
        (for CLI/Studio ``WorldFoundryPipelineRunner`` compatibility); otherwise returns
        :class:`Pi4Result` for reconstruction/tracking or ``PIL.Image``/list for rendering tasks.
        """
        del visualize_ops
        op_kwargs = dict(operator_kwargs or {})
        visual_input = (
            videos
            if videos is not None
            else (
                video
                if video is not None
                else (
                    video_path
                    if video_path is not None
                    else (
                        images
                        if images is not None
                        else (
                            image_path
                            if image_path is not None
                            else (input_path if input_path is not None else op_kwargs.get("input_path"))
                        )
                    )
                )
            )
        )

        if task_type in {"reconstruction", "tracking", "4d_tracking"}:
            if visual_input is None:
                raise ValueError("images or input_path is required for Pi4 reconstruction.")
            resolved_query_frame = query_frame
            if resolved_query_frame is None and task_type in {"tracking", "4d_tracking"}:
                resolved_query_frame = 0
            interval = kwargs.get("interval")
            result = self._run_inference(
                visual_input,
                query_frame=resolved_query_frame,
                stride=stride,
                max_frames=max_frames,
                max_size=max_size,
                conf_threshold=conf_threshold,
                interval=interval,
            )
            save_dir = self._resolve_save_dir(output_dir, output_path)
            saved_files: list[str] = []
            if save_dir is not None:
                saved_files = result.save(save_dir)

            if return_dict or output_dir is not None:
                ply_path = next((p for p in saved_files if os.path.splitext(p)[1].lower() == ".ply"), None)
                meta_path = next((p for p in saved_files if os.path.basename(p) == "meta.json"), None)
                artifact_path = ply_path or meta_path or (saved_files[0] if saved_files else None)
                return {
                    "status": "succeeded",
                    "model_id": self.model_id,
                    "task_type": task_type,
                    "artifact_kind": "generated_4d_scene" if result.has_tracking else "generated_3d_asset",
                    "artifact_path": artifact_path,
                    "point_cloud_path": ply_path,
                    "metadata_path": meta_path,
                    "run_dir": str(save_dir) if save_dir is not None else None,
                    "saved_files": saved_files,
                    "result": result,
                    "numpy_data": result.numpy_data,
                    "camera_params": result.camera_params,
                    "camera_range": result.camera_range,
                    "metadata": result.metadata,
                }
            return result

        if task_type == "render_view":
            if interactions is not None:
                n_move = int(kwargs.get("frames_per_interaction", 30))
                n_hold = int(kwargs.get("hold_frames", 10))
                frames: list[Image.Image] = []
                if self._current_camera is None and self._cached_result is not None:
                    self._current_camera = np.array(
                        self._cached_result.camera_params[0]["camera_to_world"], dtype=np.float64
                    )
                camera = self._current_camera
                if camera is None:
                    raise RuntimeError("No result available. Run inference first via __call__().")
                for sig in interactions:
                    hold_img = self.render_view(camera_to_world=camera)
                    frames.extend([hold_img] * n_hold)
                    delta = self.operator.process_interaction_single(sig)
                    sub_delta = [axis_delta / n_move for axis_delta in delta]
                    for _ in range(n_move):
                        camera = _apply_camera_delta(camera, sub_delta)
                        self._current_camera = camera
                        frames.append(self.render_view(camera_to_world=camera))
                hold_img = self.render_view(camera_to_world=camera)
                frames.extend([hold_img] * n_hold)
                return frames
            return self.render_view(camera_view=camera_view)

        if task_type == "render_trajectory":
            return self._render_trajectory(**kwargs)

        raise ValueError(
            f"Unknown task_type: {task_type}. "
            "Choose 'reconstruction', 'tracking', '4d_tracking', 'render_view', or 'render_trajectory'."
        )

    def stream(
        self,
        interaction_signal: str | list[str],
        result: Pi4Result | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Apply navigation interaction to the cached camera and render a new view without re-running Pi4."""
        del kwargs
        res = result or self._cached_result
        if res is None:
            raise RuntimeError("No result available. Run reconstruction first via __call__().")

        if self._current_camera is None:
            if res.camera_params:
                self._current_camera = np.array(res.camera_params[0]["camera_to_world"], dtype=np.float64)
            else:
                raise RuntimeError("No camera parameters available.")

        signals = [interaction_signal] if isinstance(interaction_signal, str) else list(interaction_signal)
        self.operator.get_interaction(signals)
        delta = self.operator.process_interaction()
        self._current_camera = _apply_camera_delta(self._current_camera, delta)
        return self.render_view(result=res, camera_to_world=self._current_camera)


__all__ = ["Pi4Pipeline", "Pi4Result"]
