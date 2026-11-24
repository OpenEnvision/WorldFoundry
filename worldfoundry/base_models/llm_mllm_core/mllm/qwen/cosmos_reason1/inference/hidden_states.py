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

"""Reason1 hidden-state extraction for Gamma-World inference."""

import torch
from torch.distributed._tensor import DTensor

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen import QwenModel
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen2 import (
    get_rope_index as get_rope_index_v2,
)
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen2_5 import (
    get_rope_index as get_rope_index_v2_5,
)


class QwenVLBaseModel(QwenModel):
    """Qwen-VL backbone returning hidden states without a language-model head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the Qwen backbone outputs required by the video conditioner."""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # This is a trick to handle TP for LLM but no TP for vision encoder, we need to convert DTensor to regular tensor later
            is_inputs_embeds_dtensor = isinstance(inputs_embeds, DTensor)  # This is True for TP>1, False for TP=1
            if is_inputs_embeds_dtensor:
                target_device_mesh = inputs_embeds.device_mesh
                target_placements = inputs_embeds.placements
                inputs_embeds = inputs_embeds.full_tensor()

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if is_inputs_embeds_dtensor:
                inputs_embeds = (
                    DTensor.from_local(inputs_embeds, device_mesh=target_device_mesh)
                    .redistribute(placements=target_placements)
                    .to_local()
                )
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                if self.config.model_type == "qwen2_5_vl":
                    position_ids, rope_deltas = get_rope_index_v2_5(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                elif self.config.model_type == "qwen2_vl":
                    position_ids, rope_deltas = get_rope_index_v2(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask,
                    )
                elif self.config.model_type == "qwen2_5":
                    position_ids = None
                    rope_deltas = None
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(  # Qwen2_5_VLModel
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:  # noqa: B006 - Keeps the upstream default object and signature.
        """Extract hidden states for one token batch."""
        assert "pixel_values" not in data_batch, "pixel_values should not be in data_batch, use images instead"
        pixel_values = data_batch.get("images", None)
        image_grid_thw = data_batch.get("image_grid_thw", None)
        pixel_values_videos = data_batch.get("videos", None)
        video_grid_thw = data_batch.get("video_grid_thw", None)
        attention_mask = data_batch.get("padding_mask", None)

        if image_grid_thw is not None:
            assert len(image_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            image_grid_thw = image_grid_thw[0]  # 1, N_img, 3 -> N_img, 3
        if video_grid_thw is not None:
            assert len(video_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            video_grid_thw = video_grid_thw[0]  # 1, N_video, 3 -> N_video, 3
        return self._forward(
            input_ids=tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
