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


import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from worldfoundry.core.io import resolve_data_path

from .eagle2_hg_model.configuration_eagle_chat import Eagle2ChatConfig
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor, ModelSpecificValues, reshape_model_embeddings
from .eagle2_hg_model.modeling_eagle_chat import Eagle2ChatModel

# The Eagle2 model files of the release are bundled with the WorldFoundry runtime configurations.
DEFAULT_EAGLE_MODEL_NAME = str(resolve_data_path("models/runtime/configs/gr00t-n1/eagle2_hg_model"))


def get_embeddings(
    self,
    reproject_vision: bool,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    output_hidden_states=None,
    skip_llm=False,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    assert pixel_values is not None
    vit_embeds = visual_features if visual_features is not None else self.extract_feature(pixel_values)

    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    assert selected.sum() != 0

    embeds_to_scatter = vit_embeds.reshape(-1, C).to(input_embeds.device, input_embeds.dtype)
    input_embeds[selected] = embeds_to_scatter
    input_embeds = input_embeds.reshape(B, N, C)

    # return hidden_states
    embeddings = self.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    embeddings = embeddings.hidden_states[-1]

    if reproject_vision:
        embeddings = embeddings.reshape(B * N, C)
        embeddings[selected] = embeds_to_scatter
        embeddings = embeddings.reshape(B, N, C)
    return embeddings


class EagleBackbone(nn.Module):
    def __init__(
        self,
        select_layer: int = 12,
        model_name: str = DEFAULT_EAGLE_MODEL_NAME,
        # tune_llm and tune_visual come from released checkpoint configurations and only affect training.
        tune_llm: bool = False,
        tune_visual: bool = False,
        reproject_vision: bool = False,
        scale_image_resolution: int = 1,
        processor_cfg: dict | None = None,
        projector_dim: int = -1,
        allow_reshape_visual: bool = True,
        remove_llm: bool = False,
        load_pretrained_det_eagle_path=None,
        use_local_eagle_hg_model: bool = True,
    ):
        super().__init__()
        self.reproject_vision = reproject_vision

        # use local eagle model
        if use_local_eagle_hg_model:
            model_name = DEFAULT_EAGLE_MODEL_NAME

        config = Eagle2ChatConfig.from_pretrained(model_name)
        self.model = Eagle2ChatModel._from_config(config)
        self.model.neftune_alpha = None

        if hasattr(self.model.vision_model, "vision_model") and hasattr(self.model.vision_model.vision_model, "head"):
            self.model.vision_model.vision_model.head = torch.nn.Identity()

        # remove parts of the LLM
        self.model.language_model.lm_head = torch.nn.Identity()
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        # initialize processor
        processor = EagleProcessor(
            model_path=processor_cfg["model_path"],
            max_input_tiles=processor_cfg["max_input_tiles"],
            model_spec=ModelSpecificValues(**processor_cfg["model_spec"]),
        )
        self.model.img_context_token_id = processor.get_img_context_token()

        assert self.model.template == processor.model_spec.template
        assert self.model.num_image_token == processor.model_spec.num_image_token

        if projector_dim != -1:
            self.linear = torch.nn.Linear(projector_dim, 1536)
        else:
            self.linear = torch.nn.Identity()

        if allow_reshape_visual and scale_image_resolution != 1:
            reshape_model_embeddings(self.model, scale_image_resolution)

        if load_pretrained_det_eagle_path is not None:
            print(f"loading eagle model weight from: {load_pretrained_det_eagle_path}")
            self.model.load_state_dict(torch.load(load_pretrained_det_eagle_path))

        if (
            hasattr(self.model, "vision_model")
            and hasattr(self.model.vision_model, "vision_model")
            and hasattr(self.model.vision_model.vision_model, "vision_towers")
            and len(self.model.vision_model.vision_model.vision_towers) > 1
        ):
            vision_towers = self.model.vision_model.vision_model.vision_towers

            if (
                hasattr(vision_towers[0], "vision_tower")
                and hasattr(vision_towers[0].vision_tower, "vision_model")
                and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
            ):
                vision_towers[0].vision_tower.vision_model.encoder.gradient_checkpointing = False
                vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

            if hasattr(vision_towers[1], "vision_tower"):
                vision_towers[1].vision_tower.head = torch.nn.Identity()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        embeddings = get_embeddings(
            self.model,
            self.reproject_vision,
            pixel_values=vl_input["pixel_values"],
            input_ids=vl_input["input_ids"],
            attention_mask=vl_input["attention_mask"],
        )

        embeddings = self.linear(embeddings)

        attention_mask = vl_input["attention_mask"]
        return BatchFeature(
            data={
                "backbone_features": embeddings,
                "backbone_attention_mask": attention_mask,
            }
        )  # [B, T2, hidden_size]
