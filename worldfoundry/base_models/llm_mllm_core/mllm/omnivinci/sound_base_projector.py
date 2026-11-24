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


class SoundMultimodalProjectorConfig(PretrainedConfig):
    model_type = "sound_mm_projector"

    def __init__(self, sound_mm_projector_type: str = "mlp", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sound_mm_projector_type = sound_mm_projector_type


class SoundMultimodalProjector(PreTrainedModel):
    """Project Qwen2Audio features into the released Qwen2 hidden dimension."""

    config_class = SoundMultimodalProjectorConfig

    def __init__(self, sound_mm_projector_cfg: SoundMultimodalProjectorConfig, config: PretrainedConfig) -> None:
        super().__init__(sound_mm_projector_cfg)
        if sound_mm_projector_cfg.sound_mm_projector_type != "mlp":
            raise ValueError(f"Unsupported OmniVinci sound projector: {sound_mm_projector_cfg.sound_mm_projector_type}")
        self.layers = nn.Sequential(
            nn.Linear(config.sound_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
