# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pi4 representation runtime wrapper mirroring ``Pi3XRepresentation``.

Wraps :class:`~worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.pi4.Pi4`
(ported from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``, CC BY-NC 4.0)
and normalizes outputs into NumPy arrays in consistent predicted metric world coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models.pi4 import Pi4
from .postprocess import geometry_confidence

DEFAULT_PI4_REPO_ID = "yyfz233/Pi4"
DEFAULT_PI4_REVISION = "59f307317fb630c9c5a5aa635038d98ed106aaf5"
DEFAULT_CONF_THRESHOLD = 0.05


class Pi4Representation:
    """Pi4 representation model for 4D scene geometry, camera poses, and dense motion."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_type = "pi4"

        if self.model is not None:
            self.model = self.model.to(self.device).eval()

        if self.device.startswith("cuda") and torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | Path | None = None,
        device: str | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> Pi4Representation:
        """Load a pretrained Pi4 model from a local file/directory or Hugging Face repository.

        Args:
            pretrained_model_path: Path to ``model.safetensors``, directory containing it,
                or Hugging Face repo ID (defaults to ``yyfz233/Pi4``).
            device: Target execution device (defaults to ``"cuda"`` when available, else ``"cpu"``).
            revision: Optional Hugging Face repository revision/commit SHA.

        Returns:
            An initialized :class:`Pi4Representation` instance on ``device``.
        """
        del kwargs
        source = pretrained_model_path or DEFAULT_PI4_REPO_ID
        model = Pi4.from_pretrained(source, revision=revision)
        return cls(model=model, device=device)

    @staticmethod
    def _transform_to_metric_world(
        local_points: torch.Tensor,
        camera_poses: torch.Tensor,
        metric_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert native local pointmaps and camera poses into predicted metric world coordinates.

        Applies ``m * (R @ P + t)`` to local points and scales camera translations by ``m`` so
        ``points``, ``camera_poses``, and ``tracks_3d`` share one consistent metric world frame.
        """
        world_points = torch.einsum("btij,bthwj->bthwi", camera_poses[..., :3, :3], local_points)
        world_points = (world_points + camera_poses[..., :3, 3][:, :, None, None, :]) * metric_scale[
            :, None, None, None, None
        ]
        metric_cameras = camera_poses.clone()
        metric_cameras[..., :3, 3] = camera_poses[..., :3, 3] * metric_scale[:, None, None]
        return world_points, metric_cameras

    @staticmethod
    def _build_quality_masks(
        local_points: torch.Tensor,
        confidence: torch.Tensor,
        conf_threshold: float,
    ) -> torch.Tensor:
        """Build boolean validity masks via ``geometry_confidence`` (depth-normal edge + threshold)."""
        batch_size, frames, height, width, _ = local_points.shape
        masks = torch.zeros((batch_size, frames, height, width), dtype=torch.bool, device="cpu")
        local_cpu = local_points.detach().float().cpu()
        conf_cpu = confidence.detach().float().cpu()
        for batch_idx in range(batch_size):
            for frame_idx in range(frames):
                frame_points = local_cpu[batch_idx, frame_idx]
                filtered_conf = geometry_confidence(frame_points, conf_cpu[batch_idx, frame_idx])
                finite = torch.isfinite(frame_points).all(dim=-1)
                masks[batch_idx, frame_idx] = finite & (filtered_conf >= conf_threshold)
        return masks

    def get_representation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run Pi4 inference and return NumPy arrays for geometry and optional 4D tracking.

        Args:
            data: Input mapping containing:
                - ``images``: Floating-point RGB tensor of shape ``[B, T, 3, H, W]`` in ``[0, 1]``.
                - ``query_frame`` (optional): Integer frame index (or ``[B]`` tensor) enabling
                  dense 2D/3D tracking and moving-object segmentation.
                - ``conf_threshold`` (optional): Geometry confidence threshold (default ``0.05``).

        Returns:
            Dictionary of NumPy arrays:
                - ``points``: ``[B, T, H, W, 3]`` float32 metric world coordinates.
                - ``local_points``: ``[B, T, H, W, 3]`` float32 native camera-space pointmaps.
                - ``camera_poses``: ``[B, T, 4, 4]`` float32 metric camera-to-world poses.
                - ``metric_scale``: ``[B]`` float32 sequence-level scale factor.
                - ``confidence``: ``[B, T, H, W]`` float32 geometry confidence probabilities.
                - ``depth_map``: ``[B, T, H, W]`` float32 camera-space Z depth (``local_points[..., 2]``).
                - ``masks``: ``[B, T, H, W]`` bool masks after ``geometry_confidence`` filtering.
                When ``query_frame`` is provided, additionally includes ``tracks_2d``,
                ``tracks_3d``, ``visibility``, ``tracking_confidence``, ``dynamic_mask``,
                and ``query_frame``.
        """
        if self.model is None:
            raise RuntimeError("Pi4 model not loaded. Use from_pretrained() first.")

        imgs = data["images"]
        if not isinstance(imgs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for 'images', got {type(imgs)}")
        imgs = imgs.to(self.device)

        query_frame = data.get("query_frame")
        conf_threshold = float(data.get("conf_threshold", DEFAULT_CONF_THRESHOLD))

        with torch.inference_mode():
            prediction = self.model(imgs, query_frame=query_frame)

        local_points = prediction["local_points"].detach().float()
        camera_poses = prediction["camera_poses"].detach().float()
        metric_scale = prediction["metric_scale"].detach().float()
        confidence = prediction["confidence"].detach().float()

        world_points, metric_cameras = self._transform_to_metric_world(local_points, camera_poses, metric_scale)
        masks = self._build_quality_masks(local_points, confidence, conf_threshold=conf_threshold)

        results: dict[str, Any] = {
            "points": world_points.cpu().numpy(),
            "local_points": local_points.cpu().numpy(),
            "camera_poses": metric_cameras.cpu().numpy(),
            "metric_scale": metric_scale.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "depth_map": local_points[..., 2].cpu().numpy(),
            "masks": masks.numpy(),
        }

        for key in ("tracks_2d", "tracks_3d", "visibility", "tracking_confidence", "dynamic_mask"):
            if key in prediction:
                results[key] = prediction[key].detach().float().cpu().numpy()
        if "query_frame" in prediction:
            results["query_frame"] = prediction["query_frame"].detach().cpu().numpy().astype(np.int64)

        return results


__all__ = ["DEFAULT_CONF_THRESHOLD", "DEFAULT_PI4_REPO_ID", "DEFAULT_PI4_REVISION", "Pi4Representation"]
