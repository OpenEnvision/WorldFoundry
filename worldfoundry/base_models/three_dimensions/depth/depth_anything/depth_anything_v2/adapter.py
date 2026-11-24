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

import numpy as np
import torch

from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.misc import unpack_optional

from ...base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType
from .dpt import DepthAnythingV2


class DepthAnythingDepthModel(DepthEstimationModel):
    """
    https://github.com/DepthAnything/Depth-Anything-V2
    """

    def __init__(self, model: str = "vitl", ckpt: str = "default", weights_path: str | None = None) -> None:
        super().__init__()

        self.model_config = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }[model]
        url_name = {"vits": "Small", "vitb": "Base", "vitl": "Large", "vitg": "Giant"}[model]

        if ckpt == "default":
            self.is_metric = False
            self.max_depth = None
            self.ckpt_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{url_name}/resolve/main/depth_anything_v2_{model}.pth?download=true"

        elif ckpt == "metric-indoor":
            self.is_metric = True
            self.max_depth = 20.0
            self.ckpt_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-{url_name}/resolve/main/depth_anything_v2_metric_hypersim_{model}.pth?download=true"

        elif ckpt == "metric-outdoor":
            self.is_metric = True
            self.max_depth = 80.0
            self.ckpt_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-{url_name}/resolve/main/depth_anything_v2_metric_vkitti_{model}.pth?download=true"

        else:
            raise ValueError("Invalid checkpoint name.")

        self.model = DepthAnythingV2(**self.model_config, max_depth=self.max_depth)
        state_dict = (
            torch.load(weights_path, map_location="cpu")
            if weights_path
            else torch.hub.load_state_dict_from_url(self.ckpt_url, map_location="cpu")
        )
        self.model.load_state_dict(state_dict)
        self.model.cuda().eval()

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH if self.is_metric else DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        assert rgb.size(0) == 1, "Only batch size 1 is supported."

        raw_rgb = rgb[0].detach().cpu().numpy()
        if np.issubdtype(raw_rgb.dtype, np.floating):
            raw_rgb = np.clip(raw_rgb, 0.0, 1.0) * 255.0
        raw_bgr = np.clip(raw_rgb, 0, 255).astype(np.uint8)[..., ::-1].copy()
        depth = self.model.infer_image(raw_bgr)
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth)
        depth = depth.to(device=rgb.device, dtype=torch.float32)

        if batch_dim:
            depth = depth[None]

        if self.is_metric:
            return DepthEstimationResult(metric_depth=depth)

        else:
            return DepthEstimationResult(relative_inv_depth=depth)


__all__ = ["DepthAnythingDepthModel"]
