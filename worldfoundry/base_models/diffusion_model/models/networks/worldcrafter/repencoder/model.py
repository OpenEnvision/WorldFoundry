# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .camera import (
    DEFAULT_NEAR_ZERO_BASELINE_M,
    RepEncoderCameraConditioning,
    prepare_repencoder_camera_conditioning,
)
from .checkpoint import load_checkpoint_state, read_checkpoint_metadata
from .config import RepEncoderConfig
from .input_layer import InputLayer
from .low_rank import LowRankConv2d, LowRankLinear, inject_low_rank_branches
from .output_layer import OutputLayer
from .repfeature import RepFeature
from .vggt import DinoTail, RepresentationBackbone


class RepEncoder(nn.Module):
    def __init__(
        self,
        config: RepEncoderConfig | None = None,
        *,
        compute_dtype: str | torch.dtype = torch.bfloat16,
        target_microbatch: int = 4,
        near_zero_baseline_m: float = DEFAULT_NEAR_ZERO_BASELINE_M,
    ) -> None:
        super().__init__()
        self.config = RepEncoderConfig() if config is None else config
        self.input_layer = InputLayer(config=self.config)
        self.dino_tail = DinoTail(config=self.config)
        self.vggt = RepresentationBackbone(config=self.config)
        self.repfeature = RepFeature(config=self.config)
        self.output_layer = OutputLayer(config=self.config)
        self.low_rank_branch_count = inject_low_rank_branches(self)
        self._apply_checkpoint_dtype_ownership()
        self.target_microbatch = int(target_microbatch)
        if self.target_microbatch <= 0:
            raise ValueError("target_microbatch must be positive")
        self.near_zero_baseline_m = float(near_zero_baseline_m)
        if self.near_zero_baseline_m < 0:
            raise ValueError("near_zero_baseline_m must be non-negative")
        self.compute_dtype = self._resolve_compute_dtype(compute_dtype)
        self.report: dict[str, Any] = {}

    def _apply_checkpoint_dtype_ownership(self) -> None:
        self.dino_tail.to(dtype=torch.bfloat16)
        self.vggt.to(dtype=torch.bfloat16)
        self.repfeature.to(dtype=torch.bfloat16)
        self.dino_tail.latent_cls_token.data = (
            self.dino_tail.latent_cls_token.data.float()
        )
        self.dino_tail.latent_register_tokens.data = (
            self.dino_tail.latent_register_tokens.data.float()
        )
        for module in self.modules():
            if isinstance(module, (LowRankLinear, LowRankConv2d)):
                module.low_rank_a.data = module.low_rank_a.data.float()
                module.low_rank_b.data = module.low_rank_b.data.float()
        self.input_layer.float()
        self.output_layer.float()

    @staticmethod
    def _resolve_compute_dtype(value: str | torch.dtype) -> torch.dtype:
        dtype = (
            value
            if isinstance(value, torch.dtype)
            else {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }.get(str(value).lower())
        )
        if dtype != torch.bfloat16:
            raise ValueError(
                "RepEncoder v2 has one exact inference contract: bfloat16 autocast "
                "with checkpoint-owned mixed parameter dtypes"
            )
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        device: torch.device | str = "cpu",
        compute_dtype: str | torch.dtype = torch.bfloat16,
        target_microbatch: int = 4,
        near_zero_baseline_m: float = DEFAULT_NEAR_ZERO_BASELINE_M,
    ) -> "RepEncoder":
        model_path, config, manifest, actual_sha256 = read_checkpoint_metadata(
            path, expected_sha256=expected_sha256
        )
        model = cls(
            config,
            compute_dtype=compute_dtype,
            target_microbatch=target_microbatch,
            near_zero_baseline_m=near_zero_baseline_m,
        )
        load_checkpoint_state(model, model_path, device="cpu")
        model.requires_grad_(False)
        model.eval()
        model.to(device=torch.device(device))
        model.report = {
            "format": "worldcrafter_repencoder_runtime_v1",
            "model_path": str(model_path),
            "model_sha256": actual_sha256,
            "manifest": manifest,
            "compute_dtype": str(model.compute_dtype).removeprefix("torch."),
        }
        return model

    def prepare_camera_conditioning(
        self,
        source_c2w_metric: torch.Tensor,
        target_c2w_metric: torch.Tensor,
        *,
        near_zero_baseline_m: float | None = None,
    ) -> RepEncoderCameraConditioning:
        epsilon = (
            self.near_zero_baseline_m
            if near_zero_baseline_m is None
            else float(near_zero_baseline_m)
        )
        return prepare_repencoder_camera_conditioning(
            source_c2w_metric,
            target_c2w_metric,
            near_zero_baseline_m=epsilon,
            device=self.device,
        )

    def forward_conditioned(
        self,
        source_latents: torch.Tensor,
        source_camera_tokens: torch.Tensor,
        target_rays: torch.Tensor,
    ) -> torch.Tensor:
        expected_source = (
            source_latents.shape[0],
            self.config.source_views,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )
        if tuple(source_latents.shape) != expected_source:
            raise ValueError(f"source_latents must be {expected_source}")
        batch = source_latents.shape[0]
        if tuple(source_camera_tokens.shape) != (batch, self.config.source_views, 11):
            raise ValueError("source_camera_tokens must be [B,9,11]")
        if tuple(target_rays.shape) != (
            batch,
            self.config.target_views,
            6,
            self.config.ray_height,
            self.config.ray_width,
        ):
            raise ValueError("target_rays must be [B,4,6,384,640]")
        if {source_latents.device, source_camera_tokens.device, target_rays.device} != {
            self.device
        }:
            raise ValueError("all RepEncoder inputs must be on the model device")

        if self.device.type != "cuda":
            raise RuntimeError(
                "RepEncoder exact inference requires CUDA BF16 autocast; "
                "CPU forward is not a supported runtime"
            )
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        with autocast_context:
            input_features = self.input_layer(source_latents)
            dino_features = self.dino_tail(input_features)
            scene = self.vggt(dino_features, source_camera_tokens)
            target_features = self.repfeature(
                scene,
                target_rays,
                target_microbatch=self.target_microbatch,
            )
            memory4 = self.output_layer(target_features)
        return memory4

    def forward(
        self,
        source_latents: torch.Tensor,
        source_c2w_metric: torch.Tensor,
        target_c2w_metric: torch.Tensor,
        near_zero_baseline_m: float | None = None,
    ) -> torch.Tensor:
        conditioning = self.prepare_camera_conditioning(
            source_c2w_metric,
            target_c2w_metric,
            near_zero_baseline_m=near_zero_baseline_m,
        )
        return self.forward_conditioned(
            source_latents,
            conditioning.source_camera_tokens,
            conditioning.target_rays,
        )


__all__ = ["RepEncoder"]
