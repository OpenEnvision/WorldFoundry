#    Copyright 2023 Haotian Liu
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
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, MptConfig, MptForCausalLM, MptModel

from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava.model.llava_arch import (
    LlavaMetaForCausalLM,
    LlavaMetaModel,
)


class LlavaMptConfig(MptConfig):
    model_type = "llava_mpt"


class LlavaMptModel(LlavaMetaModel, MptModel):
    config_class = LlavaMptConfig

    def __init__(self, config: MptConfig):
        config.hidden_size = config.d_model
        super().__init__(config)

    def embed_tokens(self, x):
        return self.wte(x)


class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MptForCausalLM, self).__init__(config)

        config.model_type = "llava_mpt"
        config.rope_scaling = None
        self.generation_config = GenerationConfig(
            temperature=0.0,
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
        )

        self.transformer = LlavaMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMptModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position=None,
        images=None,
    ):
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        _inputs["images"] = images
        return _inputs


AutoConfig.register("llava_mpt", LlavaMptConfig)
AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM)
