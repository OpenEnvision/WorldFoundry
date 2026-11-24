"""
This file is based on work from open-pi-zero (https://github.com/allenzren/open-pi-zero),
licensed under the MIT License.

Modifications:
   Copyright (c) 2025 Galaxea AI.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Inference wrapper around the joint G0Plus mixtures and flow-matching sampler.

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""
import logging  # noqa: E402 - Follows a module-level setup call, as upstream.
from pathlib import Path  # noqa: E402 - Follows a module-level setup call, as upstream.

import torch  # noqa: E402 - Follows a module-level setup call, as upstream.
from safetensors import safe_open  # noqa: E402 - Follows a module-level setup call, as upstream.
from torch import nn  # noqa: E402 - Follows a module-level setup call, as upstream.

from .joint_model import JointModel  # noqa: E402 - Follows a module-level setup call, as upstream.
from .modules import (  # noqa: E402 - Follows a module-level setup call, as upstream.
    ActionDecoder,
    ActionEncoder,
    SinusoidalPosEmb,
)
from .siglip import (  # noqa: E402 - Follows a module-level setup call, as upstream.
    PaliGemmaMultiModalProjector,
    SiglipVisionModel,
)

logger = logging.getLogger(__name__)


class GalaxeaZero(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.fill_padded_with_token = cfg.fill_padded_with_token

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.cond_steps = cfg.cond_steps
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.total_num_tokens = self.max_image_text_tokens + self.num_proprio_tokens + self.num_action_tokens
        self.num_input_images = cfg.num_input_images

        self.image_text_hidden_size = cfg.joint.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.joint.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.joint.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters
        self.embed_tokens_key_prefix = cfg.embed_token_key_prefix

        # Vision
        self.vision_tower = SiglipVisionModel(cfg.vision)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(cfg.vision_projector)
        self.vision_tower_key_prefix = cfg.vision.key_prefix
        self.multi_modal_projector_key_prefix = cfg.vision_projector.key_prefix

        # Mixtures
        self.joint_model = JointModel(cfg.joint)
        self.joint_model_key_prefix = cfg.joint.key_prefix

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(cfg.time_hidden_size)
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(self.action_hidden_size)

        # proprio encoder
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = ActionDecoder(
            self.action_hidden_size,
            self.action_dim,
            num_layers=cfg.action_decoder_layers,
        )

    def load_pretrained_weights(self, model_path: str | Path | None = None) -> None:
        """vision, projector, lm from paligemma"""
        root = Path(model_path or self.cfg.pretrained_model_path)
        safetensors_files = sorted(root.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No staged PaliGemma safetensors found in {root}")
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if self.embed_tokens_key_prefix in k:
                new_key = k.replace(f"{self.embed_tokens_key_prefix}.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if self.vision_tower_key_prefix in k:
                new_key = k.replace(f"{self.vision_tower_key_prefix}.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if self.multi_modal_projector_key_prefix in k:
                new_key = k.replace(f"{self.multi_modal_projector_key_prefix}.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(multi_modal_projector_state_dict, strict=True)
        logger.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in joint_model_state_dict.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if self.joint_model_key_prefix in k:
                new_key = k.replace(f"{self.joint_model_key_prefix}.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        load_result = self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        if load_result.missing_keys:
            logger.warning(f"Missing keys when loading pre-trained weights: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys when loading pre-trained weights: {load_result.unexpected_keys}")
        logger.info("Loaded pre-trained weights for lm part of the joint model")

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        device = attention_mask.device
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = 0  # proprio/action attend to image/text
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = 0  # proprio attend to itself
        causal_mask[:, action_start:, proprio_start:] = 0  # action attend to itself and proprio

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1, device=device).repeat(bsz, 1)
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1, device=device).repeat(bsz, 1)
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
            device=device,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz, seq_len = input_ids.shape
        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        # NOTE: embed_tokens will use autocast dtype automatically
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # pixel_values: [Batch_Size, T, C, H, W], T can be # of history frames or more cameras
        pixel_values = pixel_values.reshape(bsz * pixel_values.shape[1], *pixel_values.shape[2:])

        # [Batch_Size*T, Channels, Height, Width] -> [Batch_Size*T, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values)
        # [Batch_Size*T, Num_Patches, Embed_Dim] -> [Batch_Size, T*Num_Patches, Embed_Dim]
        selected_image_feature = selected_image_feature.reshape(bsz, -1, selected_image_feature.shape[-1])
        # [Batch_Size, T*Num_Patches, Embed_Dim] -> [Batch_Size, T*Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # AMP Best Practice: Use scaled_image_features.dtype to ensure consistency
        # In autocast context, all intermediate tensors will have the same dtype
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim),
            self.pad_token_id,
            dtype=scaled_image_features.dtype,  # Use image features dtype for consistency
            device=device,
        )

        # [Batch_Size, Seq_Len]
        image_mask = input_ids == self.image_token_index  # [Batch_Size, Seq_Len]
        if (  # noqa: SIM108 - Branches carry per-case comments.
            self.fill_padded_with_token
        ):  # In the final embedding, the padded ones are filled with the token of pad_token_id
            text_mask = ~image_mask
        else:  # Or with pad_token_id itself
            text_mask = ~image_mask & (input_ids != self.pad_token_id)

        # Ensure dtype consistency when assigning
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[i, :num_image_tokens]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        pixel_values: torch.FloatTensor,
        proprios: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        device = pixel_values.device
        bsz = pixel_values.size(0)

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # AMP Best Practice: Use dtype from model's output for consistency
        dtype = inputs_embeds.dtype

        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = self.split_full_mask_into_submasks(causal_mask)

        kv_caches = self.joint_model.init_mixture_caches()

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn((bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype)

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _i in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action
