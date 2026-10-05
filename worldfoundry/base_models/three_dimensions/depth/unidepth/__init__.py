# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for base_models -> three_dimensions -> depth -> unidepth -> __init__.py functionality."""

import os
from pathlib import Path
from typing import Literal

import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType
from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .models.unidepthv2.unidepthv2 import UniDepthV2

UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def resolve_unidepth_source(type: Literal["s", "b", "l"] = "l", model_path: str | None = None) -> str:
    """Resolve the shared model asset, then a local HF snapshot or Hub model."""
    if type not in {"s", "b", "l"}:
        raise ValueError(f"Unsupported UniDepth encoder: {type}")
    explicit = model_path or os.environ.get("WORLDFOUNDRY_UNIDEPTH_MODEL")
    if type == "l" and not explicit:
        explicit = os.environ.get("WORLDFOUNDRY_UNIDEPTH_V2_VITL14_MODEL_DIR")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            return str(path)
        if _offline() or path.is_absolute():
            raise FileNotFoundError(f"UniDepth model directory missing: {path}")
        return explicit
    if type == "l":
        from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

        status = BASE_MODEL_CAPABILITIES["unidepth_v2_vitl14"].assets[0].check()
        if status["ready"]:
            return status["matched_path"]
    from huggingface_hub import snapshot_download

    return snapshot_download(
        f"lpiccinelli/unidepth-v2-vit{type}14",
        local_files_only=_offline(),
        allow_patterns=["config.json", "model.safetensors"],
        revision=os.environ.get("WORLDFOUNDRY_UNIDEPTH_REVISION", UNIDEPTH_REVISION) if type == "l" else None,
    )


def load_local_unidepth_v2(source: str | Path) -> UniDepthV2:
    """Load the pinned HF V2 snapshot with its matching upstream decoder."""
    import json

    from safetensors.torch import load_file

    root = Path(source)
    config = json.loads((root / "config.json").read_text())
    if "shape_constraints" not in config.get("data", {}):
        raise ValueError(
            f"UniDepth checkpoint configuration at {root} is incompatible with the "
            "bundled V2 runtime. Load config.json and model.safetensors from "
            f"lpiccinelli/unidepth-v2-vitl14 revision {UNIDEPTH_REVISION}."
        )
    config.setdefault("training", {})
    model = UniDepthV2(config)
    state = load_file(str(root / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.resolution_level = 9
    return model


class UniDepth2Model(DepthEstimationModel):
    """Uni depth model implementation."""

    def __init__(self, type: Literal["s", "b", "l"] = "l", model_path: str | None = None, device: str = "cuda") -> None:
        """Init.

        Args:
            type: The type.
            model_path: The model path.

        Returns:
            The return value.
        """
        super().__init__()
        source = resolve_unidepth_source(type, model_path)
        if not Path(source).is_dir():
            from huggingface_hub import snapshot_download

            source = snapshot_download(
                source, local_files_only=_offline(), allow_patterns=["config.json", "model.safetensors"]
            )
        self.model = load_local_unidepth_v2(source)
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.to(device).eval()

    @property
    def depth_type(self) -> DepthType:
        """Depth type.

        Returns:
            The return value.
        """
        return DepthType.MODEL_METRIC_DEPTH

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        """Estimate.

        Args:
            src: The src.

        Returns:
            The return value.
        """
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        assert src.camera_type == CameraType.PINHOLE, "UniDepth only supports pinhole cameras"
        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        rgb = torch.clamp(rgb.moveaxis(-1, 1) * 255.0, max=255.0).byte()
        K = torch.tensor(
            [
                [focal_length, 0, rgb.shape[-1] / 2],
                [0, focal_length, rgb.shape[-2] / 2],
                [0, 0, 1],
            ],
            device=rgb.device,
        ).float()
        camera = K[None].repeat(rgb.shape[0], 1, 1)

        predictions = self.model.infer(rgb, camera)
        pred_depth = predictions["depth"].squeeze(1)
        confidence = predictions["confidence"].squeeze(1)
        points = predictions["points"].moveaxis(1, -1)

        if not batch_dim:
            pred_depth, confidence, points = pred_depth[0], confidence[0], points[0]

        return DepthEstimationResult(
            metric_depth=pred_depth,
            confidence=confidence,
            points=points,
        )
