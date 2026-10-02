from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    DepthAnything3,
)

DEFAULT_DEPTH_ANYTHING3_REPO = "depth-anything/DA3-LARGE-1.1"
DEFAULT_DEPTH_ANYTHING3_SMALL_REPO = "depth-anything/DA3-SMALL"
WORLDBENCH_DA3_PACKAGE_PATH = (
    Path("src")
    / "worldfoundry"
    / "base_models"
    / "three_dimensions"
    / "depth"
    / "depth_anything"
    / "depth_anything_v3"
)


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        matrix = np.eye(4, dtype=ext.dtype)
        matrix[:3, :4] = ext
        return matrix
    raise ValueError(f"Expected extrinsic shape (4, 4) or (3, 4), got {ext.shape}")


def _prediction_to_world_point_clouds(
    depth: np.ndarray,
    intrinsics: Optional[np.ndarray],
    extrinsics: Optional[np.ndarray],
    colors: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if intrinsics is None or extrinsics is None:
        return None

    n_views, height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)

    per_view = []
    all_points = []
    all_colors = []

    for index in range(n_views):
        depth_map = depth[index]
        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(valid_mask):
            per_view.append({"points": np.empty((0, 3), dtype=np.float32), "colors": None})
            continue

        flat_valid = valid_mask.reshape(-1)
        flat_depth = depth_map.reshape(-1)[flat_valid]
        k_inv = np.linalg.inv(intrinsics[index])
        cam_points = (k_inv @ pixels[flat_valid].T) * flat_depth[None, :]
        c2w = np.linalg.inv(_as_homogeneous44(extrinsics[index]))
        world_points = (c2w[:3, :3] @ cam_points + c2w[:3, 3:4]).T.astype(np.float32)

        point_colors = None
        if colors is not None:
            point_colors = colors[index].reshape(-1, 3)[flat_valid]
            point_colors = point_colors.astype(np.uint8)
            all_colors.append(point_colors)

        all_points.append(world_points)
        per_view.append({"points": world_points, "colors": point_colors})

    return {
        "points": np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32),
        "colors": np.concatenate(all_colors, axis=0) if all_colors else None,
        "per_view": per_view,
    }


def _build_depth_visualizations(depth: np.ndarray) -> np.ndarray:
    from worldfoundry.core.media.artifacts import build_depth_visualizations

    return build_depth_visualizations(depth)


class DepthAnything3Representation:
    """Representation wrapper for the vendored Depth Anything 3 runtime."""

    def __init__(
        self,
        model: Optional[DepthAnything3] = None,
        device: Optional[str] = None,
        default_process_res: int = 504,
        default_process_res_method: str = "upper_bound_resize",
        default_ref_view_strategy: str = "saddle_balanced",
        default_align_to_input_ext_scale: bool = True,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model
        self.default_process_res = int(default_process_res)
        self.default_process_res_method = str(default_process_res_method)
        self.default_ref_view_strategy = str(default_ref_view_strategy)
        self.default_align_to_input_ext_scale = bool(default_align_to_input_ext_scale)

        if self.model is not None:
            self.model = self.model.to(self.device).eval()

    @staticmethod
    def _resolve_model_source(pretrained_model_path: Optional[str]) -> str:
        if pretrained_model_path is None:
            return DEFAULT_DEPTH_ANYTHING3_REPO

        candidate = Path(pretrained_model_path).expanduser()
        if candidate.exists():
            if candidate.is_file():
                raise ValueError(
                    "DepthAnything3 expects a local weights directory or HuggingFace repo ID, "
                    f"but received a file path: {candidate}"
                )

            weight_markers = {
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            }
            if any((candidate / marker).is_file() for marker in weight_markers):
                return str(candidate.resolve())

            code_checkout_markers = (
                candidate / "src" / "depth_anything_3",
                candidate / WORLDBENCH_DA3_PACKAGE_PATH,
            )
            if any(marker.exists() for marker in code_checkout_markers):
                raise ValueError(
                    "Received a Depth Anything 3 code checkout rather than a weights directory. "
                    "Pass a local model directory containing `config.json`/`model.safetensors`, "
                    "or a HuggingFace repo ID such as `depth-anything/DA3-LARGE-1.1`."
                )
            return str(candidate.resolve())

        return pretrained_model_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[str] = None,
        device: Optional[str] = None,
        default_process_res: int = 504,
        default_process_res_method: str = "upper_bound_resize",
        default_ref_view_strategy: str = "saddle_balanced",
        default_align_to_input_ext_scale: bool = True,
        **kwargs,
    ) -> "DepthAnything3Representation":
        del kwargs
        model_source = cls._resolve_model_source(pretrained_model_path)
        model = DepthAnything3.from_pretrained(model_source)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(target_device).eval()
        return cls(
            model=model,
            device=str(target_device),
            default_process_res=default_process_res,
            default_process_res_method=default_process_res_method,
            default_ref_view_strategy=default_ref_view_strategy,
            default_align_to_input_ext_scale=default_align_to_input_ext_scale,
        )

    def get_representation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("DepthAnything3 model not loaded. Use from_pretrained() first.")

        images = data.get("images")
        if not images:
            raise ValueError("DepthAnything3Representation requires `images`.")

        prediction = self.model.inference(
            image=images,
            extrinsics=data.get("extrinsics"),
            intrinsics=data.get("intrinsics"),
            align_to_input_ext_scale=data.get(
                "align_to_input_ext_scale",
                self.default_align_to_input_ext_scale,
            ),
            infer_gs=data.get("infer_gs", False),
            use_ray_pose=data.get("use_ray_pose", False),
            ref_view_strategy=data.get(
                "ref_view_strategy",
                self.default_ref_view_strategy,
            ),
            render_exts=data.get("render_exts"),
            render_ixts=data.get("render_ixts"),
            render_hw=data.get("render_hw"),
            process_res=data.get("process_res", self.default_process_res),
            process_res_method=data.get(
                "process_res_method",
                self.default_process_res_method,
            ),
            export_dir=data.get("export_dir"),
            export_format=data.get("export_format", "mini_npz"),
            export_feat_layers=data.get("export_feat_layers"),
            conf_thresh_percentile=data.get("conf_thresh_percentile", 40.0),
            num_max_points=data.get("num_max_points", 1_000_000),
            show_cameras=data.get("show_cameras", True),
            feat_vis_fps=data.get("feat_vis_fps", 15),
            export_kwargs=data.get("export_kwargs"),
        )

        point_cloud = _prediction_to_world_point_clouds(
            depth=prediction.depth,
            intrinsics=prediction.intrinsics,
            extrinsics=prediction.extrinsics,
            colors=prediction.processed_images,
        )
        depth_visualizations = _build_depth_visualizations(prediction.depth)

        # Depth Anything V3 writes its viewer-ready outputs below
        # ``<export_dir>/exports`` but the upstream API returns only the in-memory
        # prediction.  Surface the files explicitly so Studio does not stop after
        # registering the 2D depth previews and omit the NPZ/GLB/PLY artifact.
        export_artifacts = []
        export_dir = data.get("export_dir")
        if export_dir:
            exports_root = Path(export_dir) / "exports"
            if exports_root.is_dir():
                export_artifacts = [
                    str(path)
                    for path in sorted(exports_root.rglob("*"))
                    if path.is_file()
                ]

        return {
            "prediction": prediction,
            "depth": prediction.depth,
            "confidence": prediction.conf,
            "sky": prediction.sky,
            "extrinsics": prediction.extrinsics,
            "intrinsics": prediction.intrinsics,
            "processed_images": prediction.processed_images,
            "depth_visualizations": depth_visualizations,
            "point_cloud": point_cloud,
            "gaussians": prediction.gaussians,
            "is_metric": bool(prediction.is_metric),
            "scale_factor": prediction.scale_factor,
            "artifact_paths": export_artifacts,
        }
