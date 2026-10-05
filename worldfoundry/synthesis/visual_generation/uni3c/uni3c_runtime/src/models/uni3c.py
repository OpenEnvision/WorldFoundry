# Copyright 2025 The RealisDance-DiT Team. All rights reserved.
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
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import CacheMixin, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor2_0,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from src.models.pcd_controller import MaskCamEmbed, WanXControlNet
from src.xfuser_compat import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    long_context_attention,
)

try:
    from diffusers.models.controlnet import zero_module
except ModuleNotFoundError:
    # Diffusers 0.35+ moved ControlNet modules under ``models.controlnets``.
    from diffusers.models.controlnets.controlnet import zero_module

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AttnProcessorSP:
    """
    This processor will be used when enabling sequential parallelism.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessorSP requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        if get_sequence_parallel_world_size() > 1:
            # convert [batch, num_head, length, channel] -> [batch, length, num_head, channel]
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

            # convert to half
            def half(x):
                return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
            original_dtype = query.dtype
            query, key, value = half(query), half(key), half(value)

            # do attention
            hidden_states = long_context_attention(
                None, query=query, key=key, value=value
            )
            # convert back
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(original_dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class ShiftedWanRotaryPosEmbed(WanRotaryPosEmbed):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor,
        shift_f: bool,
        shift_h: bool,
        shift_w: bool,
        shift_f_size: int = 81,
    ) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        ori_freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = ori_freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = ori_freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = ori_freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        cond_batch_size, _, cond_num_frames, cond_height, cond_width = cond_states.shape
        assert cond_batch_size == batch_size
        cond_ppf, cond_pph, cond_ppw = cond_num_frames // p_t, cond_height // p_h, cond_width // p_w

        if shift_f:
            # This solution is ugly. We will design new RoPE for condition insertion in the future.
            cond_freqs_f = ori_freqs[0][shift_f_size:shift_f_size + cond_ppf].view(
                cond_ppf, 1, 1, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        else:
            cond_freqs_f = ori_freqs[0][:cond_ppf].view(
                cond_ppf, 1, 1, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        if shift_h:
            cond_freqs_h = ori_freqs[1][pph:pph + cond_pph].view(
                1, cond_pph, 1, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        else:
            cond_freqs_h = ori_freqs[1][:cond_pph].view(
                1, cond_pph, 1, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        if shift_w:
            cond_freqs_w = ori_freqs[2][ppw:ppw + cond_ppw].view(
                1, 1, cond_ppw, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        else:
            cond_freqs_w = ori_freqs[2][:cond_ppw].view(
                1, 1, cond_ppw, -1).expand(cond_ppf, cond_pph, cond_ppw, -1)
        cond_freqs = torch.cat(
            [cond_freqs_f, cond_freqs_h, cond_freqs_w], dim=-1).reshape(1, 1, cond_ppf * cond_pph * cond_ppw, -1)

        final_freqs = torch.cat((freqs, cond_freqs), dim=2)  # cat along sequence length

        return final_freqs


@dataclass
class RealisDanceDiTOutput(BaseOutput):
    sample: "torch.Tensor"


class RealisDanceDiT(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
        A Transformer model for video-like data used in the Wan model.

        Args:
            patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
            num_attention_heads (`int`, defaults to `40`):
                Fixed length for text embeddings.
            attention_head_dim (`int`, defaults to `128`):
                The number of channels in each head.
            in_channels (`int`, defaults to `16`):
                The number of channels in the input.
            out_channels (`int`, defaults to `16`):
                The number of channels in the output.
            text_dim (`int`, defaults to `512`):
                Input dimension for text embeddings.
            freq_dim (`int`, defaults to `256`):
                Dimension for sinusoidal time embeddings.
            ffn_dim (`int`, defaults to `13824`):
                Intermediate dimension in feed-forward network.
            num_layers (`int`, defaults to `40`):
                The number of layers of transformer blocks to use.
            window_size (`Tuple[int]`, defaults to `(-1, -1)`):
                Window size for local attention (-1 indicates global attention).
            cross_attn_norm (`bool`, defaults to `True`):
                Enable cross-attention normalization.
            qk_norm (`bool`, defaults to `True`):
                Enable query/key normalization.
            eps (`float`, defaults to `1e-6`):
                Epsilon value for normalization layers.
            add_img_emb (`bool`, defaults to `False`):
                Whether to use img_emb.
            added_kv_proj_dim (`int`, *optional*, defaults to `None`):
                The number of channels to use for the added key and value projections. If `None`, no projection is used.

            # Adiitional Args for RealisDance-DiT
            add_cond_in_dim (`int`, defaults to 16 * num_cond):
                Input dimension for pose condition embeddings.
            attn_cond_in_dim (`int`, defaults to 16):
                Input dimension for reference image embeddings.
            shift_f (`bool`, defaults to `True`):
                Enable shifted RoPE for frame dimension.
            shift_h (`bool`, defaults to `True`):
                Enable shifted RoPE for height dimension.
            shift_w (`bool`, defaults to `True`):
                Enable shifted RoPE for width dimension.
        """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        add_cond_in_dim: int = 32,
        attn_cond_in_dim: int = 16,
        shift_f: bool = True,
        shift_h: bool = True,
        shift_w: bool = True,
        controlnet_cfg=None
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.controlnet_cfg = controlnet_cfg

        # 1. Patch & position embedding
        self.shift_f = shift_f
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.rope = ShiftedWanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )
        self.add_conv_in = nn.Conv3d(
            add_cond_in_dim, inner_dim,
            kernel_size=patch_size, stride=patch_size)
        self.add_proj = zero_module(nn.Linear(inner_dim, inner_dim))
        self.attn_conv_in = nn.Conv3d(
            attn_cond_in_dim, inner_dim,
            kernel_size=patch_size, stride=patch_size)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim ** 0.5)

        self.gradient_checkpointing = False
        self.sp_degree = 1

    def build_controlnet(self, model_path, logger=None):
        # controlnet
        self.controlnet_patch_embedding = nn.Conv3d(
            self.in_channels, self.controlnet_cfg.conv_out_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.controlnet_mask_embedding = MaskCamEmbed(self.controlnet_cfg)
        self.controlnet = WanXControlNet(self.controlnet_cfg)
        self.controlnet_rope = WanRotaryPosEmbed(self.controlnet_cfg.dim // self.controlnet_cfg.num_heads,
                                                 self.patch_size, self.rope_max_seq_len)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Uni3C controlnet is not staged locally: {model_path}. "
                "Runtime downloads are disabled."
            )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        # print("Missing keys:", missing_keys)
        if logger is not None:
            logger.info(f"Unexpected keys: {unexpected_keys}")
        else:
            print("Unexpected keys:", unexpected_keys)

    def set_sp_degree(self, sp_degree: int):
        self.sp_degree = int(sp_degree)
        if self.sp_degree > 1:
            for block in self.blocks:
                block.attn1.set_processor(AttnProcessorSP())  # only apply AttnProcessorSP to self-attn
        else:
            for block in self.blocks:
                block.attn1.set_processor(WanAttnProcessor2_0())

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        add_cond: Optional[torch.Tensor] = None,
        attn_cond: Optional[torch.Tensor] = None,
        render_latent=None,
        render_mask=None,
        camera_embedding=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        ### process controlnet inputs ###
        render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)
        controlnet_rotary_emb = self.controlnet_rope(render_latent)
        controlnet_inputs = self.controlnet_patch_embedding(render_latent)
        controlnet_inputs = controlnet_inputs.flatten(2).transpose(1, 2)

        # additional inputs (mask, camera embedding)
        if camera_embedding is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        else:
            add_inputs = render_mask
        add_inputs = self.controlnet_mask_embedding(add_inputs)
        controlnet_inputs = controlnet_inputs + add_inputs
        ### process controlnet inputs over ###

        rotary_emb = self.rope(hidden_states, attn_cond, self.shift_f, self.shift_h, self.shift_w)

        hidden_states = self.patch_embedding(hidden_states)
        add_cond = self.add_conv_in(add_cond)
        attn_cond = self.attn_conv_in(attn_cond)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        add_cond = add_cond.flatten(2).transpose(1, 2)
        attn_cond = attn_cond.flatten(2).transpose(1, 2)

        hidden_states = hidden_states + self.add_proj(add_cond)
        hidden_states_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, attn_cond], dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        ### controlnet encoding ###
        if self.sp_degree > 1:
            assert controlnet_inputs.shape[1] % self.sp_degree == 0
            controlnet_inputs = torch.chunk(controlnet_inputs, self.sp_degree, dim=1)[get_sequence_parallel_rank()]
            controlnet_rotary_emb = torch.chunk(controlnet_rotary_emb, self.sp_degree, dim=2)[get_sequence_parallel_rank()]

        with torch.autocast("cuda", dtype=self.dtype, enabled=True):
            controlnet_states = self.controlnet(hidden_states=controlnet_inputs,
                                                temb=temb,
                                                rotary_emb=controlnet_rotary_emb)
        if self.sp_degree > 1:
            controlnet_states = [get_sp_group().all_gather(cs, dim=1) for cs in controlnet_states]
        ### controlnet encoding over ###

        # 4. Transformer blocks
        # for sp split
        if self.sp_degree > 1:
            original_seq_len = hidden_states.shape[1]
            if original_seq_len % self.sp_degree != 0:
                # TODO: We should use attention mask to prevent processing padding tokens.
                # TODO: But currently, xFuserLongContextAttention does not support attention mask.
                padding_num = self.sp_degree - original_seq_len % self.sp_degree
                hidden_states = torch.cat(
                    [hidden_states, hidden_states.new_zeros(
                        hidden_states.shape[0], padding_num, hidden_states.shape[2])], dim=1)
                rotary_emb = torch.cat(
                    [rotary_emb, rotary_emb.new_zeros(
                        rotary_emb.shape[0], rotary_emb.shape[1], padding_num, rotary_emb.shape[-1])], dim=2)
            hidden_states = torch.chunk(hidden_states, self.sp_degree, dim=1)[get_sequence_parallel_rank()]
            rotary_emb = torch.chunk(rotary_emb, self.sp_degree, dim=2)[get_sequence_parallel_rank()]

        def _block_forward(x):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for i, block in enumerate(self.blocks):
                    x = self._gradient_checkpointing_func(
                        block, x, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                    # adding control features
                    if i < len(controlnet_states):
                        if self.sp_degree > 1:
                            x = get_sp_group().all_gather(x, dim=1)
                        x, extra_x = x[:, :hidden_states_len], x[:, hidden_states_len:]
                        x += controlnet_states[i]
                        x = torch.cat((x, extra_x), dim=1)
                        if self.sp_degree > 1:
                            x = torch.chunk(x, self.sp_degree, dim=1)[get_sequence_parallel_rank()]

            else:
                for i, block in enumerate(self.blocks):
                    x = block(x, encoder_hidden_states, timestep_proj, rotary_emb)
                    # adding control features
                    if i < len(controlnet_states):
                        if self.sp_degree > 1:
                            x = get_sp_group().all_gather(x, dim=1)
                        x, extra_x = x[:, :hidden_states_len], x[:, hidden_states_len:]
                        x += controlnet_states[i]
                        x = torch.cat((x, extra_x), dim=1)
                        if self.sp_degree > 1:
                            x = torch.chunk(x, self.sp_degree, dim=1)[get_sequence_parallel_rank()]
            return x


        hidden_states = _block_forward(hidden_states)

        # for sp gather
        if self.sp_degree > 1:
            hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :hidden_states_len]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return RealisDanceDiTOutput(sample=output)
