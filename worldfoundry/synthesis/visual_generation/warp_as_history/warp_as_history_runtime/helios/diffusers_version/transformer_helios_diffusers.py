from __future__ import annotations

# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
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
from functools import lru_cache
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import logging

from worldfoundry.base_models.diffusion_model.models.networks.helios.components import (
    HeliosAttention as CanonicalHeliosAttention,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.components import (
    HeliosOutputNorm as HeliosOutputNorm,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.components import (
    HeliosTimeTextEmbedding as HeliosTimeTextEmbedding,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.components import (
    _get_qkv_projections as _get_qkv_projections,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.components import (
    apply_rotary_emb_transposed as apply_rotary_emb_transposed,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.history import (
    pool_history_visible_mask as pool_history_visible_mask,
)
from worldfoundry.base_models.diffusion_model.models.networks.helios.history import (
    resolve_history_keep_mask as resolve_history_keep_mask,
)
from worldfoundry.core.nn.blocks.volume import center_down_sample_3d as center_down_sample_3d
from worldfoundry.core.nn.blocks.volume import pad_for_3d_conv as pad_for_3d_conv

try:
    from diffusers.utils import apply_lora_scale
except ImportError:

    def apply_lora_scale(_scale_key):
        def decorator(fn):
            return fn

        return decorator


from diffusers.utils.torch_utils import maybe_allow_in_graph

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def filter_history_tokens_by_mask(
    hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    keep_mask: torch.Tensor | None,
    threshold: float = 0.5,
):
    keep = resolve_history_keep_mask(keep_mask, threshold=threshold)
    if keep is None:
        return hidden_states, rotary_emb
    if bool(keep.all()):
        return hidden_states, rotary_emb
    if not bool(keep.any()):
        return hidden_states[:, :0], rotary_emb[:, :0]
    return hidden_states[:, keep, :], rotary_emb[:, keep, :]


def replace_history_tokens_by_mask(
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor | None,
    invisible_token: torch.Tensor | None,
    threshold: float = 0.5,
):
    keep = resolve_history_keep_mask(keep_mask, threshold=threshold)
    if keep is None or bool(keep.all()):
        return hidden_states
    if invisible_token is None:
        raise ValueError("history invisible token mode requires transformer.history_invisible_token to be initialized.")
    replace = (~keep).view(1, -1, 1)
    token = invisible_token.to(device=hidden_states.device, dtype=hidden_states.dtype)
    token = token.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
    return torch.where(replace, token, hidden_states)


class HeliosAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HeliosAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: HeliosAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        original_context_length: int | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

        if not attn.is_cross_attention and attn.is_amplify_history:
            history_seq_len = hidden_states.shape[1] - original_context_length

            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(attn.history_key_scale) * (attn.max_scale - 1.0)
                if attn.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                key = torch.cat([key[:, :history_seq_len] * scale_key, key[:, history_seq_len:]], dim=1)

        history_key_boost_mask = getattr(attn, "history_key_boost_mask", None)
        history_key_boost_scale = float(getattr(attn, "history_key_boost_scale", 1.0) or 1.0)
        if not attn.is_cross_attention and history_key_boost_mask is not None and history_key_boost_scale != 1.0:
            boost_mask = history_key_boost_mask.to(device=key.device, dtype=torch.bool)
            if boost_mask.ndim != 1:
                raise ValueError(f"history_key_boost_mask must be 1D, got {tuple(boost_mask.shape)}.")
            if boost_mask.shape[0] != key.shape[1]:
                raise ValueError(
                    "history_key_boost_mask length must match self-attention key sequence length, "
                    f"got mask={boost_mask.shape[0]} key={key.shape[1]}."
                )
            if bool(boost_mask.any()):
                key = torch.where(
                    boost_mask.view(1, -1, 1, 1),
                    key * history_key_boost_scale,
                    key,
                )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            # Reference: https://github.com/huggingface/diffusers/pull/12909
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class HeliosAttention(CanonicalHeliosAttention):
    _default_processor_cls = HeliosAttnProcessor
    _available_processors: ClassVar[list] = [HeliosAttnProcessor]


class HeliosRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta
        self.register_buffer("freqs_base_t", self._get_freqs_base(self.DT), persistent=False)
        self.register_buffer("freqs_base_y", self._get_freqs_base(self.DY), persistent=False)
        self.register_buffer("freqs_base_x", self._get_freqs_base(self.DX), persistent=False)

    def _get_freqs_base(self, dim):
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    def get_frequency_batched(self, freqs_base, pos):
        freqs = torch.einsum("d,bthw->dbthw", freqs_base, pos)
        freqs = freqs.repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    @lru_cache(maxsize=32)  # noqa: B019 - Keeps the upstream per-instance method cache.
    def _get_spatial_meshgrid(self, height, width, device_str):
        device = torch.device(device_str)
        grid_y_coords = torch.arange(height, device=device, dtype=torch.float32)
        grid_x_coords = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y_coords, grid_x_coords, indexing="ij")
        return grid_y, grid_x

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        batch_size = frame_indices.shape[0]
        num_frames = frame_indices.shape[1]

        frame_indices = frame_indices.to(device=device, dtype=torch.float32)
        grid_y, grid_x = self._get_spatial_meshgrid(height, width, str(device))

        grid_t = frame_indices[:, :, None, None].expand(batch_size, num_frames, height, width)
        grid_y_batch = grid_y[None, None, :, :].expand(batch_size, num_frames, -1, -1)
        grid_x_batch = grid_x[None, None, :, :].expand(batch_size, num_frames, -1, -1)

        freqs_cos_t, freqs_sin_t = self.get_frequency_batched(self.freqs_base_t, grid_t)
        freqs_cos_y, freqs_sin_y = self.get_frequency_batched(self.freqs_base_y, grid_y_batch)
        freqs_cos_x, freqs_sin_x = self.get_frequency_batched(self.freqs_base_x, grid_x_batch)

        result = torch.cat([freqs_cos_t, freqs_cos_y, freqs_cos_x, freqs_sin_t, freqs_sin_y, freqs_sin_x], dim=0)

        return result.permute(1, 0, 2, 3, 4)

    def forward_with_positions(self, frame_positions, y_positions, x_positions, device):
        frame_positions = frame_positions.to(device=device, dtype=torch.float32)
        y_positions = y_positions.to(device=device, dtype=torch.float32)
        x_positions = x_positions.to(device=device, dtype=torch.float32)

        freqs_cos_t, freqs_sin_t = self.get_frequency_batched(self.freqs_base_t, frame_positions)
        freqs_cos_y, freqs_sin_y = self.get_frequency_batched(self.freqs_base_y, y_positions)
        freqs_cos_x, freqs_sin_x = self.get_frequency_batched(self.freqs_base_x, x_positions)

        result = torch.cat([freqs_cos_t, freqs_cos_y, freqs_cos_x, freqs_sin_t, freqs_sin_y, freqs_sin_x], dim=0)
        return result.permute(1, 0, 2, 3, 4)


@maybe_allow_in_graph
class HeliosTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        guidance_cross_attn: bool = False,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head]
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=HeliosAttnProcessor(),
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
        )

        # 2. Cross-attention
        self.attn2 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=HeliosAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # 4. Guidance cross-attention
        self.guidance_cross_attn = guidance_cross_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        original_context_length: int | None = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
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
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            None,
            rotary_emb,
            original_context_length,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        if self.guidance_cross_attn:
            history_seq_len = hidden_states.shape[1] - original_context_length

            history_hidden_states, hidden_states = torch.split(
                hidden_states, [history_seq_len, original_context_length], dim=1
            )
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                None,
                None,
                original_context_length,
            )
            hidden_states = hidden_states + attn_output
            hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
        else:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                None,
                None,
                original_context_length,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class HeliosTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Helios model.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
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
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
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
    _skip_layerwise_casting_patterns: ClassVar[list] = [
        "patch_embedding",
        "patch_short",
        "patch_mid",
        "patch_long",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules: ClassVar[list] = ["HeliosTransformerBlock", "HeliosOutputNorm"]
    _keep_in_fp32_modules: ClassVar[list] = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
        "history_key_scale",
    ]
    _keys_to_ignore_on_load_unexpected: ClassVar[list] = ["norm_added_q"]
    _repeated_blocks: ClassVar[list] = ["HeliosTransformerBlock"]
    _cp_plan: ClassVar[dict] = {
        # Input split at attn level and ffn level.
        "blocks.*.attn1": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "rotary_emb": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.attn2": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*.ffn": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # Output gather at attn level and ffn level.
        **{f"blocks.{i}.attn1": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.attn2": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
        **{f"blocks.{i}.ffn": ContextParallelOutput(gather_dim=1, expected_dims=3) for i in range(40)},
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
        added_kv_proj_dim: int | None = None,
        rope_dim: tuple[int, ...] = (44, 42, 42),
        rope_theta: float = 10000.0,
        guidance_cross_attn: bool = True,
        zero_history_timestep: bool = True,
        has_multi_term_memory_patch: bool = True,
        is_amplify_history: bool = False,
        history_scale_mode: str = "per_head",  # [scalar, per_head]
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = HeliosRotaryPosEmbed(rope_dim=rope_dim, theta=rope_theta)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Initial Multi Term Memory Patch
        self.zero_history_timestep = zero_history_timestep
        self.inner_dim = inner_dim
        if has_multi_term_memory_patch:
            self.patch_short = nn.Conv3d(in_channels, self.inner_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_mid = nn.Conv3d(
                in_channels,
                self.inner_dim,
                kernel_size=tuple(2 * p for p in patch_size),
                stride=tuple(2 * p for p in patch_size),
            )
            self.patch_long = nn.Conv3d(
                in_channels,
                self.inner_dim,
                kernel_size=tuple(4 * p for p in patch_size),
                stride=tuple(4 * p for p in patch_size),
            )

        # 3. Condition embeddings
        self.condition_embedder = HeliosTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                HeliosTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    guidance_cross_attn=guidance_cross_attn,
                    is_amplify_history=is_amplify_history,
                    history_scale_mode=history_scale_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = HeliosOutputNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))

        self.gradient_checkpointing = False

    def enable_target_channel_fusion(self):
        if getattr(self, "target_channel_fusion_mlp", None) is not None:
            return
        self.target_channel_fusion_mlp = nn.Sequential(
            nn.Linear(self.inner_dim * 2, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        nn.init.xavier_uniform_(self.target_channel_fusion_mlp[0].weight)
        nn.init.zeros_(self.target_channel_fusion_mlp[0].bias)
        nn.init.zeros_(self.target_channel_fusion_mlp[2].weight)
        nn.init.zeros_(self.target_channel_fusion_mlp[2].bias)

    def fuse_target_channel_condition(self, hidden_states, condition_hidden_states):
        fusion_mlp = getattr(self, "target_channel_fusion_mlp", None)
        if fusion_mlp is None:
            raise ValueError("target_channel_fusion_latents were provided before enabling target_channel_fusion_mlp.")
        if hidden_states.shape != condition_hidden_states.shape:
            raise ValueError(
                "target channel fusion expects condition and target patch tokens to have the same shape, "
                f"got target={tuple(hidden_states.shape)} condition={tuple(condition_hidden_states.shape)}"
            )
        fusion_dtype = next(fusion_mlp.parameters()).dtype
        fusion_input = torch.cat([hidden_states, condition_hidden_states.to(hidden_states)], dim=-1).to(fusion_dtype)
        fused_delta = fusion_mlp(fusion_input)
        return hidden_states + fused_delta.to(hidden_states)

    def _maybe_apply_learned_rope_delta(
        self,
        rotary_emb: torch.Tensor,
        frame_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        attention_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        if not attention_kwargs or "learned_rope_delta" not in attention_kwargs:
            return rotary_emb
        delta = attention_kwargs["learned_rope_delta"]
        if delta is None:
            return rotary_emb
        delta = delta.to(device=device, dtype=torch.float32)
        if (
            delta.ndim != 5
            or delta.shape[1] != 3
            or delta.shape[-3:]
            != (
                frame_indices.shape[1],
                height,
                width,
            )
        ):
            return rotary_emb

        batch_size = frame_indices.shape[0]
        if delta.shape[0] == 1 and batch_size != 1:
            delta = delta.expand(batch_size, -1, -1, -1, -1)
        elif delta.shape[0] != batch_size:
            return rotary_emb

        grid_y, grid_x = self.rope._get_spatial_meshgrid(height, width, str(device))
        frame_pos = frame_indices.to(device=device, dtype=torch.float32)[:, :, None, None].expand(
            batch_size, frame_indices.shape[1], height, width
        )
        pos_y = grid_y[None, None, :, :].expand(batch_size, frame_indices.shape[1], -1, -1)
        pos_x = grid_x[None, None, :, :].expand(batch_size, frame_indices.shape[1], -1, -1)
        return self.rope.forward_with_positions(
            frame_pos + delta[:, 0],
            pos_y + delta[:, 1],
            pos_x + delta[:, 2],
            device,
        )

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # ------------ Stage 1 ------------
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        history_visible_mask_short=None,
        history_visible_mask_mid=None,
        history_visible_mask_long=None,
        target_channel_fusion_latents=None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        history_threshold = 0.5
        history_invisible_token_mode = "none"
        history_key_boost_mode = "none"
        history_key_boost_scale = 1.0
        if attention_kwargs:
            history_threshold = float(attention_kwargs.get("history_visible_token_threshold", 0.5) or 0.5)
            history_invisible_token_mode = str(attention_kwargs.get("history_invisible_token_mode", "none") or "none")
            history_key_boost_mode = str(attention_kwargs.get("history_key_boost_mode", "none") or "none")
            history_key_boost_scale = float(attention_kwargs.get("history_key_boost_scale", 1.0) or 1.0)
        history_key_boost_enabled = (
            history_key_boost_mode
            in {
                "short_last_latent",
                "warp_last_latent",
            }
            and history_key_boost_scale != 1.0
        )
        # 1. Input
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        # 2. Process noisy latents
        hidden_states = self.patch_embedding(
            hidden_states.to(device=self.patch_embedding.weight.device, dtype=self.patch_embedding.weight.dtype)
        )
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape

        if indices_hidden_states is None:
            indices_hidden_states = torch.arange(0, post_patch_num_frames).unsqueeze(0).expand(batch_size, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if target_channel_fusion_latents is not None:
            condition_hidden_states = self.patch_embedding(
                target_channel_fusion_latents.to(
                    device=self.patch_embedding.weight.device, dtype=self.patch_embedding.weight.dtype
                )
            )
            condition_hidden_states = condition_hidden_states.flatten(2).transpose(1, 2)
            hidden_states = self.fuse_target_channel_condition(hidden_states, condition_hidden_states)
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = self._maybe_apply_learned_rope_delta(
            rotary_emb,
            indices_hidden_states,
            post_patch_height,
            post_patch_width,
            hidden_states.device,
            attention_kwargs,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]
        history_key_boost_mask = torch.zeros(original_context_length, device=hidden_states.device, dtype=torch.bool)

        # 3. Process short history latents
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            boost_mask_short = None
            if history_key_boost_enabled:
                _, _, T1, _, _ = latents_history_short.shape
                boost_mask_short = torch.zeros(
                    T1,
                    H1,
                    W1,
                    device=latents_history_short.device,
                    dtype=torch.bool,
                )
                boost_mask_short[-1] = True
                boost_mask_short = boost_mask_short.flatten()
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_history_short = rotary_emb_history_short.flatten(2).transpose(1, 2)
            keep_mask_short = pool_history_visible_mask(history_visible_mask_short, (1, 2, 2))
            if boost_mask_short is not None:
                keep_short = resolve_history_keep_mask(keep_mask_short, threshold=history_threshold)
                if keep_short is not None:
                    boost_mask_short = boost_mask_short[keep_short.to(device=boost_mask_short.device)]
            if history_invisible_token_mode == "global":
                latents_history_short = replace_history_tokens_by_mask(
                    latents_history_short,
                    keep_mask_short,
                    getattr(self, "history_invisible_token", None),
                    threshold=history_threshold,
                )
            else:
                latents_history_short, rotary_emb_history_short = filter_history_tokens_by_mask(
                    latents_history_short,
                    rotary_emb_history_short,
                    keep_mask_short,
                    threshold=history_threshold,
                )

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_short, rotary_emb], dim=1)
            if boost_mask_short is None:
                boost_mask_short = torch.zeros(
                    latents_history_short.shape[1],
                    device=history_key_boost_mask.device,
                    dtype=torch.bool,
                )
            history_key_boost_mask = torch.cat(
                [boost_mask_short.to(device=history_key_boost_mask.device), history_key_boost_mask],
                dim=0,
            )

        # 4. Process mid history latents
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_history_mid = pad_for_3d_conv(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = center_down_sample_3d(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = rotary_emb_history_mid.flatten(2).transpose(1, 2)
            keep_mask_mid = pool_history_visible_mask(history_visible_mask_mid, (2, 4, 4))
            if history_invisible_token_mode == "global":
                latents_history_mid = replace_history_tokens_by_mask(
                    latents_history_mid,
                    keep_mask_mid,
                    getattr(self, "history_invisible_token", None),
                    threshold=history_threshold,
                )
            else:
                latents_history_mid, rotary_emb_history_mid = filter_history_tokens_by_mask(
                    latents_history_mid,
                    rotary_emb_history_mid,
                    keep_mask_mid,
                    threshold=history_threshold,
                )

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_mid, rotary_emb], dim=1)
            history_key_boost_mask = torch.cat(
                [
                    torch.zeros(
                        latents_history_mid.shape[1],
                        device=history_key_boost_mask.device,
                        dtype=torch.bool,
                    ),
                    history_key_boost_mask,
                ],
                dim=0,
            )

        # 5. Process long history latents
        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.patch_long(latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rotary_emb_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rotary_emb_history_long = pad_for_3d_conv(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = center_down_sample_3d(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = rotary_emb_history_long.flatten(2).transpose(1, 2)
            keep_mask_long = pool_history_visible_mask(history_visible_mask_long, (4, 8, 8))
            if history_invisible_token_mode == "global":
                latents_history_long = replace_history_tokens_by_mask(
                    latents_history_long,
                    keep_mask_long,
                    getattr(self, "history_invisible_token", None),
                    threshold=history_threshold,
                )
            else:
                latents_history_long, rotary_emb_history_long = filter_history_tokens_by_mask(
                    latents_history_long,
                    rotary_emb_history_long,
                    keep_mask_long,
                    threshold=history_threshold,
                )

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_long, rotary_emb], dim=1)
            history_key_boost_mask = torch.cat(
                [
                    torch.zeros(
                        latents_history_long.shape[1],
                        device=history_key_boost_mask.device,
                        dtype=torch.bool,
                    ),
                    history_key_boost_mask,
                ],
                dim=0,
            )

        history_context_length = hidden_states.shape[1] - original_context_length

        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(timestep, encoder_hidden_states)
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)

        # 6. Transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()
        if history_key_boost_enabled and bool(history_key_boost_mask.any()):
            for block in self.blocks:
                block.attn1.history_key_boost_mask = history_key_boost_mask
                block.attn1.history_key_boost_scale = float(history_key_boost_scale)
        else:
            for block in self.blocks:
                block.attn1.history_key_boost_mask = None
                block.attn1.history_key_boost_scale = 1.0
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )

        # 7. Normalization
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
