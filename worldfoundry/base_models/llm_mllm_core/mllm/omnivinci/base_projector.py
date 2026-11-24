# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class DownSampleBlock(nn.Module):
    """Downsample 2D feature maps by rearranging into 2x2 blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x: torch.Tensor) -> torch.Tensor:
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = "mlp_downsample", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    """Released 2x2 spatial packing followed by the vision-to-Qwen MLP."""

    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig) -> None:
        super().__init__(mm_projector_cfg)
        if mm_projector_cfg.mm_projector_type != "mlp_downsample":
            raise ValueError(f"Unsupported OmniVinci vision projector: {mm_projector_cfg.mm_projector_type}")
        self.layers = nn.Sequential(
            DownSampleBlock(),
            nn.LayerNorm(config.mm_hidden_size * 4),
            nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
