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

import torch
from torch import nn
from transformers import PretrainedConfig, SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class VisionTower(nn.Module):
    def __init__(self, vision_tower: str, args: PretrainedConfig) -> None:
        super().__init__()

        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

    def feature_select(self, image_forward_outs: BaseModelOutputWithPooling) -> torch.Tensor:
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features


class VisionTowerDynamicS2(VisionTower):
    def __init__(self, vision_tower: str, args: PretrainedConfig) -> None:
        super().__init__(vision_tower, args)

        self.scales = list(map(int, args.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = args.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(args, "s2_resize_output_to_scale_idx", 0)

    def forward_feature(self, images: torch.Tensor) -> torch.Tensor:
        image_forward_outs = self.vision_tower(
            images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert type(images) is not list
        image_features = self.forward_feature(images)

        return image_features


class SiglipVisionTowerDynamicS2(VisionTowerDynamicS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        if isinstance(config.model_dtype, str):
            model_dtype = getattr(torch, config.model_dtype.removeprefix("torch."))
        else:
            model_dtype = config.model_dtype

        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=model_dtype,
            local_files_only=True,
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path, local_files_only=True)
        # Each Dynamic-S2 tile uses the first scale's image resolution.
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[0]
