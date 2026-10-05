"""Checkpoint-compatible HunyuanVideo language-model wrappers.

:class:`HunyuanVideoLLMEncoder` and
:class:`HunyuanVideoMLLMEncoder` adapt Hugging Face causal
LMs so original Hunyuan prompt encoding can call them as
encoder-only towers.

Used by :mod:`.original` / :mod:`.i2v`.  HunyuanVideo 1.5
LLM loading lives under :mod:`.h15_text`.

This is a weight-compatible wrapper, not a ConditionEncoder
by itself.
"""

from transformers import LlamaModel, LlamaConfig, DynamicCache, LlavaForConditionalGeneration
from copy import deepcopy
import torch


class HunyuanVideoLLMEncoder(LlamaModel):
    """Hunyuan video llm encoder implementation."""

    def __init__(self, config: LlamaConfig):
        """Init.

        Args:
            config: The config.
        """
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        """Enable auto offload."""
        self.auto_offload = True

    def forward(self, input_ids, attention_mask, hidden_state_skip_layer=2):
        """Forward.

        Args:
            input_ids: The input ids.
            attention_mask: The attention mask.
            hidden_state_skip_layer: The hidden state skip layer.
        """
        embed_tokens = deepcopy(self.embed_tokens).to(input_ids.device) if self.auto_offload else self.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        past_key_values = DynamicCache()

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        rotary_emb = deepcopy(self.rotary_emb).to(input_ids.device) if self.auto_offload else self.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_id, decoder_layer in enumerate(self.layers):
            if self.auto_offload:
                decoder_layer = deepcopy(decoder_layer).to(hidden_states.device)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if layer_id + hidden_state_skip_layer + 1 >= len(self.layers):
                break

        return hidden_states


class HunyuanVideoMLLMEncoder(LlavaForConditionalGeneration):
    """Hunyuan video mllm encoder implementation."""

    def __init__(self, config):
        """Init.

        Args:
            config: The config.
        """
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        """Enable auto offload."""
        self.auto_offload = True

    # TODO: implement the low VRAM inference for MLLM.
    def forward(self, input_ids, pixel_values, attention_mask, hidden_state_skip_layer=2):
        """Forward.

        Args:
            input_ids: The input ids.
            pixel_values: The pixel values.
            attention_mask: The attention mask.
            hidden_state_skip_layer: The hidden state skip layer.
        """
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True,
                                  pixel_values=pixel_values)
        hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
        return hidden_state
