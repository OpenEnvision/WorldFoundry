# Inference-only DB-CogACT source retained in-tree.
from typing import List, Optional

import torch
import torch.nn as nn

from .action_builder import build_action_model
from .base import (ActionOutputForCausalLM,
                                          CausalLMOutputDexbotic,
                                          DexboticConfig, DexboticForCausalLM,
                                          DexboticVLMModel)


class CogActConfig(DexboticConfig):
    model_type = "dexbotic_cogact"
    action_model_type: Optional[str] = None
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None


class CogActModel(DexboticVLMModel):
    def __init__(self, config: CogActConfig):
        super().__init__(config)
        if config.action_model_type is not None:
            self.action_head = self._build_action_head_module(config)

    def _build_action_head_module(self, config: CogActConfig):
        if getattr(self, 'action_head', None) is not None:
            return self.action_head
        self.action_head = build_action_model(config)
        return self.action_head

    @property
    def action_head_module(self) -> nn.Module:
        return self.action_head

    @property
    def action_head_prefix(self) -> str:
        return 'action_head'

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        self.mm_projector = self._build_mm_projector_module(self.config)
        self.action_head = self._build_action_head_module(self.config)


class CogACTForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = CogActConfig
    _tied_weights_keys = {}

    def _real_init(self, config: CogActConfig):
        self.model = CogActModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> CausalLMOutputDexbotic:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            cache_position
        ) = self.model._prepare_inputs_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            cache_position,
            images
        )
        outputs = self.model.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]

        if not return_dict:
            return last_hidden_state

        return CausalLMOutputDexbotic(
            logits=last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

        )

    @torch.no_grad()
    def inference_action(self, input_ids, image_tensor, inference_args=None, **kwargs):
        inference_args = inference_args or {}
        cfg_scale = inference_args.get('cfg_scale', 1.5)
        num_ddim_steps = inference_args.get('num_ddim_steps', 10)
        action_norms = inference_args.get('action_norms')
        seed = inference_args.get('seed')

        out_features = self.__call__(
            input_ids=input_ids,
            images=image_tensor,
            use_cache=True)

        cognition_features = out_features.logits[:, -1, :].unsqueeze(1)  # [B, 1, D]
        B = cognition_features.size(0)

        generator = None
        if seed is not None:
            # Use a device-local generator so a request seed is reproducible
            # without mutating the process-wide RNG used by other models.
            generator = torch.Generator(device=cognition_features.device)
            generator.manual_seed(int(seed))
        noise = torch.randn(
            B,
            self.config.chunk_size,
            self.config.action_dim,
            device=cognition_features.device,
            dtype=cognition_features.dtype,
            generator=generator)  # [B T D]

        if cfg_scale > 1.0:
            noise = torch.cat([noise, noise], 0)

            uncondition = self.model.action_head.net.z_embedder.uncondition  # [1, D]
            uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)  # [B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.model.action_head.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.model.action_head.net.forward

        if (
            self.model.action_head.ddim_diffusion is None
            or self.model.action_head.ddim_diffusion.num_timesteps != num_ddim_steps
        ):
            self.model.action_head.create_ddim(ddim_step=num_ddim_steps)

        samples = self.model.action_head.ddim_diffusion.ddim_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=cognition_features.device,
            eta=0.0)
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)

        actions = self._denorm(samples[0].cpu().numpy(), action_norms).tolist()
        return actions
