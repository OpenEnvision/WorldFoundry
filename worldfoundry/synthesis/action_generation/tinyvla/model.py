"""Inference-only LLaVA-Pythia action model used by TinyVLA."""

from __future__ import annotations

import contextlib

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GPTNeoXModel, GPTNeoXPreTrainedModel

from .act_head import build_act_head
from .config import LlavaPythiaConfig
from .diffusion_head import ConditionalUnet1D
from .multimodal import LlavaMetaForCausalLM, LlavaMetaModel


class LLavaPythiaModel(LlavaMetaModel, GPTNeoXModel):
    config_class = LlavaPythiaConfig

    def __init__(self, config) -> None:
        super().__init__(config)


class LlavaPythiaForCausalLM(GPTNeoXPreTrainedModel, LlavaMetaForCausalLM):
    """GPT-NeoX/Vision encoder with ACT or diffusion action decoding."""

    config_class = LlavaPythiaConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.gpt_neox = LLavaPythiaModel(config)
        self.head_type = config.action_head_type
        self.visual_concat = config.concat
        self.action_dim = int(config.action_dim)

        if self.head_type == "act":
            action_spec = dict(config.act["act"])
            self.embed_out = build_act_head(action_spec, state_dim=int(config.state_dim))
            middle_dim = int(max(config.hidden_size, action_spec["hidden_dim"]) / 2)
            self.proj_to_action = nn.Sequential(
                nn.Linear(config.hidden_size, middle_dim),
                nn.LayerNorm(middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, action_spec["hidden_dim"]),
                nn.LayerNorm(action_spec["hidden_dim"]),
            )
        elif self.head_type == "droid_diffusion":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            self.proj_to_action = nn.Identity()
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=100,
                    beta_schedule="squaredcos_cap_v2",
                    clip_sample=True,
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type="epsilon",
                )
            finally:
                torch.set_default_dtype(previous_dtype)
            self.embed_out = ConditionalUnet1D(
                input_dim=int(config.action_dim),
                global_cond_dim=int(config.hidden_size),
                state_dim=int(config.state_dim),
            )
            self.num_queries = int(config.chunk_size)
            self.num_inference_timesteps = 10
        else:
            raise ValueError(f"unsupported TinyVLA action head: {self.head_type!r}")
        self.post_init()

    def get_model(self):
        return self.gpt_neox

    def get_channel_proj(self, x):
        return self.channel_proj(x)

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings) -> None:
        self.embed_out = new_embeddings

    def encode_images(self, images, proj: bool = True):
        features = self.get_model().get_vision_tower()(images)
        return self.get_model().mm_projector(features) if proj else features

    def get_mm_projector(self, image_features):
        return self.get_model().mm_projector(image_features)

    def get_image_fusion_embedding(
        self,
        visual_concat=None,
        images=None,
        images_r=None,
        images_top=None,
        states=None,
    ):
        del states
        if "channel_cat" in visual_concat:
            raise ValueError("channel_cat TinyVLA checkpoints are not supported")
        features = self.encode_images(images)
        if images_r is None:
            return features
        if visual_concat != "token_cat":
            raise ValueError(f"unsupported TinyVLA visual concatenation: {visual_concat!r}")
        result = [features, self.encode_images(images_r)]
        if images_top is not None:
            result.append(self.encode_images(images_top))
        return torch.cat(result, dim=1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
        actions=None,
        states=None,
        images_r=None,
        images_top=None,
        is_pad=None,
        eval: bool = False,
    ):
        del inputs_embeds, actions, is_pad
        if not eval:
            raise RuntimeError("the in-tree TinyVLA model retains inference code only; pass eval=True")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, attention_mask, past_key_values, model_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            images_r=images_r,
            images_top=images_top,
            visual_concat=self.visual_concat,
            states=states,
        )
        outputs = self.get_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=model_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        if self.head_type == "act":
            hidden_states = self.proj_to_action(hidden_states)
            action, _, _, _, _ = self.embed_out(
                qpos=states,
                hidden_states=hidden_states,
                env_state=None,
            )
            return action
        return self._sample_diffusion_action(hidden_states, states)

    def _sample_diffusion_action(self, hidden_states, states):
        batch_size = hidden_states.shape[0]
        action = torch.randn(
            (batch_size, self.num_queries, self.action_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_timesteps,
            device=hidden_states.device,
        )
        for timestep in self.noise_scheduler.timesteps:
            noise = self.embed_out(action, timestep, global_cond=hidden_states, states=states)
            action = self.noise_scheduler.step(noise, timestep, action).prev_sample
        return action


with contextlib.suppress(ValueError):
    AutoConfig.register(LlavaPythiaConfig.model_type, LlavaPythiaConfig)
with contextlib.suppress(ValueError):
    AutoModelForCausalLM.register(LlavaPythiaConfig, LlavaPythiaForCausalLM)


__all__ = ["LLavaPythiaModel", "LlavaPythiaForCausalLM"]
