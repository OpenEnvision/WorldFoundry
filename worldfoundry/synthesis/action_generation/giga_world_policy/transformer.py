# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.image_embedding import WanImageEmbedding

from .action_projectors import EmbodimentSpecificActionDecoder, EmbodimentSpecificActionEncoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors: ClassVar[list] = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class WanRotaryPosEmbed1D(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freq_cos, freq_sin = get_1d_rotary_pos_embed(
            attention_head_dim,
            max_seq_len,
            theta,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=freqs_dtype,
        )
        self.register_buffer("freqs_cos", freq_cos, persistent=False)
        self.register_buffer("freqs_sin", freq_sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, num_heads, head_dim)
        seq_len = hidden_states.shape[1]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(2)
        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, self_attention_mask, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class CasualWorldActionTransformer(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
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
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns: ClassVar[list] = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules: ClassVar[list] = ["WanTransformerBlock"]
    _keep_in_fp32_modules: ClassVar[list] = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected: ClassVar[list] = ["norm_added_q"]
    _repeated_blocks: ClassVar[list] = ["WanTransformerBlock"]
    _cp_plan: ClassVar[dict] = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        in_action_channels: int = 32,
        out_action_channels: int = 32,
        num_embodiments: int = 4,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

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
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # MLP, encode actions into a latent space [B, T, 14] -> [B, T, emb_dim]
        self.action_rope = WanRotaryPosEmbed1D(attention_head_dim, rope_max_seq_len)
        self.action_encoder = EmbodimentSpecificActionEncoder(
            action_dim=in_action_channels,
            inner_dim=inner_dim,
            num_embodiments=num_embodiments,
        )
        self.action_decoder = EmbodimentSpecificActionDecoder(
            inner_dim=inner_dim,
            action_dim=out_action_channels,
            num_embodiments=num_embodiments,
        )

    def _encode_action_tokens(self, action: torch.Tensor, emb_ids: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(self.action_encoder, "uses_emb_ids", False):
            return self.action_encoder(action, emb_ids=emb_ids)
        return self.action_encoder(action)

    def _decode_action_tokens(self, action_states: torch.Tensor, emb_ids: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(self.action_decoder, "uses_emb_ids", False):
            return self.action_decoder(action_states, emb_ids=emb_ids)
        return self.action_decoder(action_states)

    def _embed_conditions(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the timestep and the text and image conditions.

        Returns:
            ``(temb, timestep_proj, encoder_hidden_states)``: ``timestep_proj`` is split into 6 modulation
            chunks along the last-but-one dimension, and image tokens are prepended to the text tokens.
        """
        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        return temb, timestep_proj, encoder_hidden_states

    def _modulate_output(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply the output norm with the timestep shift and scale."""
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # With multi-GPU inference through accelerate, shift and scale live on the first device while
        # hidden_states ends up on the last one.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        return (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

    def forward(self, *args, **kwargs):
        action_only = kwargs.pop("action_only", False)
        no_video_tokens = kwargs.pop("no_video_tokens", False)
        if self.training:
            if action_only:
                if no_video_tokens:
                    return self._forward_train_action_only_no_video(*args, **kwargs)
                return self._forward_train_action_only(*args, **kwargs)
            return self._forward_train(*args, **kwargs)
        if action_only:
            return self._forward_inference_action_only(*args, **kwargs)
        return self._forward_inference(*args, **kwargs)

    def _forward_train_action_only_no_video(
        self,
        ref_latents: torch.Tensor = None,
        noisy_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        action: torch.Tensor | None = None,
        emb_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        action_states = self._encode_action_tokens(action, emb_ids=emb_ids)
        state_states = self._encode_action_tokens(state, emb_ids=emb_ids)
        num_action_tokens = action_states.shape[1]
        num_state_tokens = state_states.shape[1]
        extra_states = torch.cat([state_states, action_states], dim=1)
        batch_size = extra_states.shape[0]

        num_ref_tokens = 0
        ref_states = None
        ref_rotary_emb = None
        if ref_latents is not None:
            ref_rotary_emb = self.rope(ref_latents)
            ref_states = self.patch_embedding(ref_latents)
            ref_states = ref_states.flatten(2).transpose(1, 2)
            num_ref_tokens = ref_states.shape[1]

        hidden_states = (
            torch.cat([state_states, ref_states, action_states], dim=1)
            if ref_states is not None
            else torch.cat([state_states, action_states], dim=1)
        )

        extra_rotary_emb = self.action_rope(extra_states)
        if ref_rotary_emb is not None:
            rotary_emb = (
                torch.cat(
                    [
                        extra_rotary_emb[0][:, :num_state_tokens],
                        ref_rotary_emb[0][:, :num_ref_tokens],
                        extra_rotary_emb[0][:, num_state_tokens:],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        extra_rotary_emb[1][:, :num_state_tokens],
                        ref_rotary_emb[1][:, :num_ref_tokens],
                        extra_rotary_emb[1][:, num_state_tokens:],
                    ],
                    dim=1,
                ),
            )
        else:
            rotary_emb = extra_rotary_emb

        temb, timestep_proj, encoder_hidden_states = self._embed_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        L = hidden_states.shape[1]
        device = hidden_states.device
        mask = torch.zeros((L, L), device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        mask[:s_r_end, s_r_end:] = float("-inf")
        self_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_attention_heads, L, L)
        self_attention_mask = self_attention_mask.to(dtype=hidden_states.dtype)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )

        hidden_states = self._modulate_output(hidden_states, temb)

        pred_action_states = hidden_states[
            :, num_state_tokens + num_ref_tokens : num_state_tokens + num_ref_tokens + num_action_tokens
        ]
        action_pred = self._decode_action_tokens(pred_action_states, emb_ids=emb_ids)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return action_pred

        return {"action_pred": action_pred}

    def _forward_train_action_only(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        action: torch.Tensor | None = None,
        emb_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        action_states = self._encode_action_tokens(action, emb_ids=emb_ids)
        state_states = self._encode_action_tokens(state, emb_ids=emb_ids)
        num_action_tokens = action_states.shape[1]
        num_state_tokens = state_states.shape[1]
        num_ref_tokens = post_patch_width * post_patch_height
        num_noisy_tokens = (post_patch_num_frames - 1) * post_patch_width * post_patch_height
        extra_states = torch.cat([state_states, action_states], dim=1)

        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states = torch.cat(
            [
                extra_states[:, :num_state_tokens],
                hidden_states[:, :num_ref_tokens],
                extra_states[:, num_state_tokens:],
                hidden_states[:, num_ref_tokens:],
            ],
            dim=1,
        )
        rotary_emb = (
            torch.cat(
                [
                    extra_rotary_emb[0][:, :num_state_tokens],
                    rotary_emb[0][:, :num_ref_tokens],
                    extra_rotary_emb[0][:, num_state_tokens:],
                    rotary_emb[0][:, num_ref_tokens:],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    extra_rotary_emb[1][:, :num_state_tokens],
                    rotary_emb[1][:, :num_ref_tokens],
                    extra_rotary_emb[1][:, num_state_tokens:],
                    rotary_emb[1][:, num_ref_tokens:],
                ],
                dim=1,
            ),
        )

        temb, timestep_proj, encoder_hidden_states = self._embed_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        L = hidden_states.shape[1]
        device = hidden_states.device
        mask = torch.zeros((L, L), device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        action_end = s_r_end + num_action_tokens
        mask[s_r_end:action_end, action_end:] = float("-inf")
        mask[:s_r_end, s_r_end:] = float("-inf")
        self_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_attention_heads, L, L)
        self_attention_mask = self_attention_mask.to(dtype=hidden_states.dtype)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )

        hidden_states = self._modulate_output(hidden_states, temb)

        action_states = hidden_states[:, -(num_action_tokens + num_noisy_tokens) : -num_noisy_tokens]
        action_pred = self._decode_action_tokens(action_states, emb_ids=emb_ids)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return action_pred

        return {"action_pred": action_pred}

    def _forward_inference(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        action: torch.Tensor | None = None,
        emb_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # 1. LoRA scale, handled as in training.
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # 2. Video dimensions; noisy_latents is the input of the current denoising step.
        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 3. Robot state and action tokens; at inference the action is zero noise or the previous sampling step's result.
        action_states = self._encode_action_tokens(action, emb_ids=emb_ids)
        state_states = self._encode_action_tokens(state, emb_ids=emb_ids)
        num_action_tokens = action_states.shape[1]
        num_state_tokens = state_states.shape[1]
        num_ref_tokens = post_patch_width * post_patch_height
        _num_noisy_tokens = (post_patch_num_frames - 1) * post_patch_width * post_patch_height
        extra_states = torch.cat([state_states, action_states], dim=1)

        # 4. RoPE is computed per stream here and interleaved in the token order below.
        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        # 5. Patchify the video.
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 6. Token order: [State(1), Ref_Video(spatial), Action(T), Noisy_Video(spatial * (T-1))].
        hidden_states = torch.cat(
            [
                extra_states[:, :num_state_tokens],
                hidden_states[:, :num_ref_tokens],
                extra_states[:, num_state_tokens:],
                hidden_states[:, num_ref_tokens:],
            ],
            dim=1,
        )
        # RoPE
        rotary_emb = (
            torch.cat(
                [
                    extra_rotary_emb[0][:, :num_state_tokens],
                    rotary_emb[0][:, :num_ref_tokens],
                    extra_rotary_emb[0][:, num_state_tokens:],
                    rotary_emb[0][:, num_ref_tokens:],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    extra_rotary_emb[1][:, :num_state_tokens],
                    rotary_emb[1][:, :num_ref_tokens],
                    extra_rotary_emb[1][:, num_state_tokens:],
                    rotary_emb[1][:, num_ref_tokens:],
                ],
                dim=1,
            ),
        )

        # 7. Timestep and text conditioning.
        temb, timestep_proj, encoder_hidden_states = self._embed_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        # 8. Causal attention mask.
        L = hidden_states.shape[1]
        device = hidden_states.device
        mask = torch.zeros((L, L), device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        action_end = s_r_end + num_action_tokens

        # Earlier segments cannot attend to later ones.
        mask[s_r_end:action_end, action_end:] = float("-inf")
        mask[:s_r_end, s_r_end:] = float("-inf")

        self_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_attention_heads, L, L)
        self_attention_mask = self_attention_mask.to(dtype=hidden_states.dtype)

        # 9. Transformer blocks.
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask)

        # 10. Output norm and modulation.
        hidden_states = self._modulate_output(hidden_states, temb)

        # 11. Output heads: action prediction.
        pred_action_states = hidden_states[:, s_r_end:action_end]
        action_pred = self._decode_action_tokens(pred_action_states, emb_ids=emb_ids)

        # Video latents: the reference part followed by the denoised future frames.
        video_hidden = torch.cat([hidden_states[:, num_state_tokens:s_r_end], hidden_states[:, action_end:]], dim=1)
        video_output = self.proj_out(video_hidden)

        # Unpatchify to [B, C, T, H, W].
        video_output = video_output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_output = video_output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, action_pred

        return Transformer2DModelOutput(sample=output)  # Use return_dict=False to also get the action prediction.

    def _forward_inference_action_only(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        action: torch.Tensor | None = None,
        emb_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # 1. LoRA scale, handled as in training.
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # 2. Video dimensions; noisy_latents is the input of the current denoising step.
        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 3. Robot state and action tokens; at inference the action is zero noise or the previous sampling step's result.
        action_states = self._encode_action_tokens(action, emb_ids=emb_ids)
        state_states = self._encode_action_tokens(state, emb_ids=emb_ids)
        num_action_tokens = action_states.shape[1]
        num_state_tokens = state_states.shape[1]
        num_ref_tokens = post_patch_width * post_patch_height
        _num_noisy_tokens = (post_patch_num_frames - 1) * post_patch_width * post_patch_height
        extra_states = torch.cat([state_states, action_states], dim=1)

        # 4. RoPE is computed per stream here and interleaved in the token order below.
        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        # 5. Patchify the video.
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 6. Token order: [State(1), Ref_Video(spatial), Action(T), Noisy_Video(spatial * (T-1))].
        hidden_states = torch.cat(
            [
                extra_states[:, :num_state_tokens],
                hidden_states[:, :num_ref_tokens],
                extra_states[:, num_state_tokens:],
                # hidden_states[:, num_ref_tokens:]
            ],
            dim=1,
        )
        # RoPE
        rotary_emb = (
            torch.cat(
                [
                    extra_rotary_emb[0][:, :num_state_tokens],
                    rotary_emb[0][:, :num_ref_tokens],
                    extra_rotary_emb[0][:, num_state_tokens:],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    extra_rotary_emb[1][:, :num_state_tokens],
                    rotary_emb[1][:, :num_ref_tokens],
                    extra_rotary_emb[1][:, num_state_tokens:],
                ],
                dim=1,
            ),
        )

        timestep = timestep[:, : hidden_states.shape[1]]
        # 7. Timestep and text conditioning.
        temb, timestep_proj, encoder_hidden_states = self._embed_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        # 8. Causal attention mask.
        L = hidden_states.shape[1]
        device = hidden_states.device
        mask = torch.zeros((L, L), device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        action_end = s_r_end + num_action_tokens

        # State and reference tokens cannot attend to action tokens.
        # mask[s_r_end:action_end, action_end:] = float("-inf")
        mask[:s_r_end, s_r_end:] = float("-inf")

        self_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_attention_heads, L, L)
        self_attention_mask = self_attention_mask.to(dtype=hidden_states.dtype)

        # 9. Transformer blocks.
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask)
        # 10. Output norm and modulation.
        hidden_states = self._modulate_output(hidden_states, temb)

        # 11. Output heads: action prediction.
        pred_action_states = hidden_states[:, s_r_end:action_end]
        action_pred = self._decode_action_tokens(pred_action_states, emb_ids=emb_ids)

        # breakpoint()
        # # Video latents: the reference part followed by the denoised future frames.
        # video_hidden = torch.cat([hidden_states[:, num_state_tokens:s_r_end], hidden_states[:, action_end:]], dim=1)
        # video_output = self.proj_out(video_hidden)

        # # Unpatchify to [B, C, T, H, W].
        # video_output = video_output.reshape(
        #     batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        # )
        # video_output = video_output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # output = video_output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return action_pred

        return Transformer2DModelOutput(sample=None)  # Use return_dict=False to get the action prediction.

    def _forward_train(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        action: torch.Tensor | None = None,
        emb_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        action_states = self._encode_action_tokens(action, emb_ids=emb_ids)  # (1, 4)
        state_states = self._encode_action_tokens(state, emb_ids=emb_ids)
        num_action_tokens = action_states.shape[1]  # [B, 48, D]
        num_state_tokens = state_states.shape[1]  # [B, 1, D]
        num_ref_tokens = post_patch_width * post_patch_height
        num_noisy_tokens = (post_patch_num_frames - 1) * post_patch_width * post_patch_height
        extra_states = torch.cat([state_states, action_states], dim=1)  # [B, 49, D]

        rotary_emb = self.rope(hidden_states)  # (B, 288, 1, D)
        extra_rotary_emb = self.action_rope(extra_states)  # (B, 49, 1, D)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        _num_hidden_tokens = hidden_states.shape[1]

        # state, ref, action, noisy tokens
        hidden_states = torch.cat(
            [
                extra_states[:, :num_state_tokens],
                hidden_states[:, :num_ref_tokens],
                extra_states[:, num_state_tokens:],
                hidden_states[:, num_ref_tokens:],
            ],
            dim=1,
        )
        rotary_emb = (
            torch.cat(
                [
                    extra_rotary_emb[0][:, :num_state_tokens],
                    rotary_emb[0][:, :num_ref_tokens],
                    extra_rotary_emb[0][:, num_state_tokens:],
                    rotary_emb[0][:, num_ref_tokens:],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    extra_rotary_emb[1][:, :num_state_tokens],
                    rotary_emb[1][:, :num_ref_tokens],
                    extra_rotary_emb[1][:, num_state_tokens:],
                    rotary_emb[1][:, num_ref_tokens:],
                ],
                dim=1,
            ),
        )

        temb, timestep_proj, encoder_hidden_states = self._embed_conditions(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )

        L = hidden_states.shape[1]
        device = hidden_states.device
        mask = torch.zeros((L, L), device=device)
        s_r_end = num_state_tokens + num_ref_tokens
        action_end = s_r_end + num_action_tokens
        mask[s_r_end:action_end, action_end:] = float("-inf")
        mask[:s_r_end, s_r_end:] = float("-inf")
        self_attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_attention_heads, L, L)
        self_attention_mask = self_attention_mask.to(dtype=hidden_states.dtype)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attention_mask
                )

        # 5. Output norm, projection & unpatchify
        hidden_states = self._modulate_output(hidden_states, temb)
        action_states = hidden_states[:, -(num_action_tokens + num_noisy_tokens) : -num_noisy_tokens]
        hidden_states = torch.cat(
            [hidden_states[:, num_state_tokens:s_r_end], hidden_states[:, -num_noisy_tokens:]], dim=1
        )
        action_pred = self._decode_action_tokens(action_states, emb_ids=emb_ids)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, action_pred

        return Transformer2DModelOutput(sample=output)
