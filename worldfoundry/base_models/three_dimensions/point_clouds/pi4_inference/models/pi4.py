# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pi4 (pi^4) joint 4D scene geometry and motion model assembly.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/pi4.py``). Upstream code and weights are licensed under CC BY-NC 4.0.

Preserves the exact 1,319-tensor ``yyfz233/Pi4/model.safetensors`` state_dict layout
while enabling both CUDA (BF16 autocast, with optional ``flex_gemm`` acceleration)
and pure-PyTorch CPU/CUDA execution via the sparse 3D fallback in :mod:`.refiner`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import GeometryDecoder, MotionDecoder
from .encoder import Encoder
from .heads import CameraHead, MetricHead, MotionMask, PointHead, Tracker
from .layers import patch_positions
from .refiner import DepthRefiner


class Pi4(nn.Module):
    """Geometry and dense 4D motion from RGB images with a fixed inference architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.geometry_decoder = GeometryDecoder()
        self.point_head = PointHead()
        self.camera_head = CameraHead()
        self.metric_head = MetricHead()
        self.depth_refiner = DepthRefiner()
        self.motion_decoder = MotionDecoder()
        self.tracker = Tracker()
        self.motion_mask = MotionMask()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = "yyfz233/Pi4",
        *,
        revision: str | None = None,
    ) -> Pi4:
        """Load Pi4 strictly from a local Safetensors file, directory, or Hugging Face repo."""
        from safetensors.torch import load_file

        location = Path(model_id)
        if location.is_file():
            checkpoint: str | Path = location
        elif location.is_dir():
            checkpoint = location / "model.safetensors"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"model.safetensors not found in {location}")
        else:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(repo_id=str(model_id), filename="model.safetensors", revision=revision)
        state = load_file(str(checkpoint), device="cpu")
        with torch.device("meta"):
            model = cls()
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        query_frame: int | torch.Tensor | list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run joint geometry (and optional 4D tracking/motion) inference.

        Args:
            images: Floating-point RGB tensor of shape ``[B, T, 3, H, W]`` in ``[0, 1]``,
                with ``H`` and ``W`` positive multiples of 16. Runs on CUDA (BF16 autocast
                for transformer/motion stages, FP32 for geometry heads/refiner) or CPU.
            query_frame: Optional integer frame index (scalar or ``[B]`` tensor) activating
                dense 2D/3D tracking and moving-object segmentation. Requires ``T >= 2``.

        Returns:
            Dictionary containing ``local_points``, ``camera_poses``, ``metric_scale``,
            ``confidence``, and (when ``query_frame`` is provided) ``tracks_2d``,
            ``tracks_3d``, ``visibility``, ``tracking_confidence``, ``dynamic_mask``,
            and ``query_frame``.
        """
        if images.ndim != 5 or images.shape[2] != 3 or not images.is_floating_point():
            raise ValueError("images must be floating-point RGB [B,T,3,H,W] in [0,1]")
        batch_size, frames, _, height, width = images.shape
        if min(batch_size, frames, height, width) < 1 or height % 16 or width % 16:
            raise ValueError("image dimensions must be positive multiples of 16")
        queries: torch.Tensor | None = None
        if query_frame is not None:
            if frames < 2:
                raise ValueError("tracking requires at least two images")
            queries = torch.as_tensor(query_frame, device=images.device)
            if queries.dtype == torch.bool or queries.is_floating_point():
                raise ValueError("query_frame must contain integer frame indices")
            if queries.ndim == 0:
                queries = queries.expand(batch_size)
            if queries.shape != (batch_size,) or (queries < 0).any() or (queries >= frames).any():
                raise ValueError("query_frame must be one valid index per batch element")
            queries = queries.long()

        device_type = images.device.type
        use_bf16_autocast = device_type == "cuda"

        images = images.float()
        mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(1, 1, 3, 1, 1)
        deviation = images.new_tensor((0.229, 0.224, 0.225)).reshape(1, 1, 3, 1, 1)
        normalized = (images - mean) / deviation
        positions = patch_positions(batch_size * frames, height // 16, width // 16, images.device)
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16_autocast):
            tokens = self.encoder(normalized.flatten(0, 1))
            geometry, initial = self.geometry_decoder(tokens, positions, batch_size, frames)
        del tokens
        points: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        with torch.amp.autocast(device_type=device_type, enabled=False):
            for frame_tokens in geometry[:, 5:].split(1):
                rays, log_depth, confidence, feature = self.point_head(frame_tokens, height, width)
                coarse_depth = log_depth.exp().clamp_min(1e-6).log()
                points.append(self.depth_refiner(rays, coarse_depth, feature))
                confidences.append(confidence)
            local_points = torch.cat(points).reshape(batch_size, frames, height, width, 3)
            confidence_logits = torch.cat(confidences).reshape(batch_size, frames, height, width)
            cameras = self.camera_head(geometry, positions).reshape(batch_size, frames, 4, 4)
            metric_scale = self.metric_head(geometry[:, 5:], positions[:, 5:], batch_size)
        del points, confidences
        prediction: dict[str, Any] = {
            "local_points": local_points,
            "camera_poses": cameras,
            "metric_scale": metric_scale,
            "confidence": confidence_logits.sigmoid(),
        }
        del confidence_logits
        if queries is not None:
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16_autocast):
                motion = self.motion_decoder(initial, geometry, positions, batch_size, frames)[:, 5:]
                del initial, geometry
                tracks, visibility, track_confidence = self.tracker(motion, normalized, queries)
                mask = self.motion_mask(motion, normalized.flatten(0, 1))
            prediction.update(
                tracks_2d=tracks,
                visibility=visibility.sigmoid(),
                tracking_confidence=track_confidence.sigmoid(),
                dynamic_mask=mask.reshape(batch_size, frames, height, width).sigmoid(),
                query_frame=queries,
            )
            prediction["tracks_3d"] = self.lift_tracks(tracks, local_points, cameras, metric_scale)
        return prediction

    @staticmethod
    def lift_tracks(
        tracks: torch.Tensor,
        local_points: torch.Tensor,
        camera_poses: torch.Tensor,
        metric_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Lift 2D pixel correspondences onto reconstructed surfaces in metric world XYZ."""
        batch_size, frames, height, width, _ = local_points.shape
        grid = (tracks + 0.5) * tracks.new_tensor((2 / width, 2 / height)) - 1
        sampled = F.grid_sample(
            local_points.flatten(0, 1).permute(0, 3, 1, 2),
            grid.flatten(0, 1),
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1).reshape(batch_size, frames, height, width, 3)
        world = torch.einsum("btij,bthwj->bthwi", camera_poses[..., :3, :3], sampled)
        world = (world + camera_poses[..., :3, 3][:, :, None, None]) * metric_scale[:, None, None, None, None]
        inside = (
            (tracks[..., 0] >= 0)
            & (tracks[..., 0] <= width - 1)
            & (tracks[..., 1] >= 0)
            & (tracks[..., 1] <= height - 1)
        )
        return world.masked_fill(~inside[..., None], float("nan"))
