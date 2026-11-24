#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen2MoeModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.llava_arch import (
    LlavaMetaForCausalLM,
    LlavaMetaModel,
)

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenMoeConfig(Qwen2MoeConfig):
    model_type = "llava_qwen_moe"


class LlavaQwenMoeModel(LlavaMetaModel, Qwen2MoeModel):
    config_class = LlavaQwenMoeConfig

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__(config)


class LlavaQwenMoeForCausalLM(Qwen2MoeForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenMoeConfig

    def __init__(self, config):
        # super(Qwen2MoeForCausalLM, self).__init__(config)
        Qwen2MoeForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_moe"
        config.rope_scaling = None

        self.model = LlavaQwenMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        image_sizes: list[list[int]] | None = None,
        return_dict: bool | None = None,
        modalities: list[str] | None = ["image"],  # noqa: B006 - Keeps the upstream default object and signature.
        dpo_forward: bool | None = False,
        cache_position=None,
    ) -> tuple | CausalLMOutputWithPast:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes
                )
            )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        modalities: list[str] | None = ["image"],  # noqa: B006 - Keeps the upstream default object and signature.
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen_moe", LlavaQwenMoeConfig)
AutoModelForCausalLM.register(LlavaQwenMoeConfig, LlavaQwenMoeForCausalLM)
