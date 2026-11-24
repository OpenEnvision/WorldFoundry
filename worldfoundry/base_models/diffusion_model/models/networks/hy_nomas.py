# Copyright 2026 The Hy-Nomas Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Tencent-Hunyuan/Hy-Nomas (911cea6458a89c2663513ff7099561ec190dd491)
# and jiaxiangc/diffusers (12722ef11acf8e335f96cfd4faea17de78cd8909).
"""Checkpoint-compatible segment-causal Hy-Nomas transformer; inference only."""

from __future__ import annotations

import itertools
import math
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.core.attention import attention_forward
from worldfoundry.core.nn import ConditioningProjection, DiTModulation, RMSNorm, TransformerMLP


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    backend: str | None = None,
    parallel_config: object = None,
) -> torch.Tensor:
    return attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        backend=backend or "torch",
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
    )


def _apply_rotary_emb(hidden_states: Any, rotary_emb: Any) -> Any:
    cosine, sine = rotary_emb
    real, imaginary = hidden_states.float().unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(-2)
    return (hidden_states.float() * cosine[None, :, None] + rotated * sine[None, :, None]).to(hidden_states.dtype)


class HyNomasLayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )


class HyNomasRotaryPosEmbed(nn.Module):
    def __init__(self, axes_dim: Any, theta: Any) -> None:
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def forward(
        self, num_frames: Any, height: Any, width: Any, device: Any, frame_offset: Any = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        computation_device = "cpu" if frame_offset else device
        positions = torch.meshgrid(
            torch.arange(frame_offset, frame_offset + num_frames, device=computation_device, dtype=torch.float32),
            torch.arange(height, device=computation_device, dtype=torch.float32),
            torch.arange(width, device=computation_device, dtype=torch.float32),
            indexing="ij",
        )
        cosines, sines = ([], [])
        for dim, position in zip(self.axes_dim, positions, strict=False):
            frequencies = 1.0 / self.theta ** (torch.arange(0, dim, 2, device=computation_device).float() / dim)
            angles = torch.outer(position.flatten(), frequencies)
            cosines.append(angles.cos().repeat_interleave(2, dim=-1))
            sines.append(angles.sin().repeat_interleave(2, dim=-1))
        return (torch.cat(cosines, dim=-1).to(device), torch.cat(sines, dim=-1).to(device))


class HyNomasTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: Any) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.frequencies = tuple(
            torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device="cpu") / 128).tolist()
        )

    def forward(self, timestep: Any, dtype: Any) -> torch.Tensor:
        frequencies = torch.tensor(self.frequencies, dtype=torch.float32, device=timestep.device)
        angles = timestep[:, None].float() * frequencies[None]
        hidden_states = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(hidden_states.to(dtype))


class HyNomasTextAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: Any, hidden_states: Any, attention_mask: Any = None) -> Any:
        query, key, value = attn.self_attn_qkv(hidden_states).unflatten(-1, (3, attn.heads, -1)).unbind(-3)
        hidden_states = _attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.self_attn_proj(hidden_states.flatten(-2))


class HyNomasTextAttention(nn.Module):
    _default_processor_cls = HyNomasTextAttnProcessor
    _supports_qkv_fusion = False

    def __init__(self, hidden_size: Any, heads: Any) -> None:
        super().__init__()
        self.heads = heads
        self.self_attn_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.processor = self._default_processor_cls()

    def forward(self, hidden_states: Any, attention_mask: Any = None) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask)


class HyNomasTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size: Any, heads: Any) -> None:
        super().__init__()
        self.norm1 = HyNomasLayerNorm(hidden_size, eps=1e-06)
        self.attn = HyNomasTextAttention(hidden_size, heads)
        self.norm2 = HyNomasLayerNorm(hidden_size, eps=1e-06)
        self.mlp = TransformerMLP(hidden_size, hidden_size * 4, act_layer=nn.SiLU)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, hidden_states: Any, temb: Any, attention_mask: Any) -> torch.Tensor:
        attention_gate, mlp_gate = self.adaLN_modulation(temb).chunk(2, dim=-1)
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states).to(hidden_states.dtype), attention_mask) * attention_gate[:, None]
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * mlp_gate[:, None]
        return hidden_states


class HyNomasIndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size: Any, heads: Any) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([HyNomasTokenRefinerBlock(hidden_size, heads) for _ in range(2)])

    def forward(self, hidden_states: Any, temb: Any, attention_mask: Any) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask.clone().bool()
            attention_mask[:, 0] = True
            attention_mask = attention_mask[:, None, None, :]
        for block in self.blocks:
            hidden_states = block(hidden_states, temb, attention_mask)
        return hidden_states


class HyNomasTokenRefiner(nn.Module):
    def __init__(self, in_channels: Any, hidden_size: Any, heads: Any) -> None:
        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = HyNomasTimestepEmbedder(hidden_size)
        self.c_embedder = ConditioningProjection(in_channels, hidden_size, nn.SiLU)
        self.individual_token_refiner = HyNomasIndividualTokenRefiner(hidden_size, heads)

    def forward(self, hidden_states: Any, timestep: Any, attention_mask: Any = None) -> torch.Tensor:
        temb = self.t_embedder(timestep, hidden_states.dtype)
        if attention_mask is None:
            context = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            context = (hidden_states * mask).sum(dim=1, dtype=torch.float32) / mask.sum(
                dim=1, dtype=torch.float32
            ).clamp_min(1)
        temb = temb + self.c_embedder(context.to(hidden_states.dtype))
        hidden_states = self.input_embedder(hidden_states)
        return self.individual_token_refiner(hidden_states, temb, attention_mask)


class HyNomasAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any,
        rotary_emb: Any,
        encoder_attention_mask: Any = None,
        kv_cache: Any = None,
        update_cache: Any = False,
        action_text_offsets: Any = None,
        action_token_windows: Any = None,
        cache_action_start: Any = None,
    ) -> Any:
        query = attn.img_attn_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.img_attn_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.img_attn_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = _apply_rotary_emb(attn.img_attn_q_norm(query).to(value.dtype), rotary_emb)
        key = _apply_rotary_emb(attn.img_attn_k_norm(key).to(value.dtype), rotary_emb)
        encoder_query = attn.txt_attn_q(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_key = attn.txt_attn_k(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_value = attn.txt_attn_v(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        encoder_query = attn.txt_attn_q_norm(encoder_query).to(encoder_value.dtype)
        encoder_key = attn.txt_attn_k_norm(encoder_key).to(encoder_value.dtype)
        query = torch.cat((query, encoder_query), dim=1)
        current_key = torch.cat((key, encoder_key), dim=1)
        current_value = torch.cat((value, encoder_value), dim=1)
        batch_size, video_length = hidden_states.shape[:2]
        text_length = encoder_hidden_states.shape[1]
        text_mask = encoder_attention_mask
        if text_mask is None:
            text_mask = torch.ones(batch_size, text_length, dtype=torch.bool, device=query.device)
        current_mask = torch.cat(
            (torch.ones(batch_size, video_length, dtype=torch.bool, device=query.device), text_mask.bool()), dim=1
        )
        history_length = 0
        key, value, key_mask = (current_key, current_value, current_mask)
        store = None if kv_cache is None else kv_cache.get("store")
        if store is not None and store.length:
            history_key, history_value = store.materialize()
            history_length = history_key.shape[1]
            key = torch.cat((history_key, current_key), dim=1)
            value = torch.cat((history_value, current_value), dim=1)
            key_mask = torch.cat((kv_cache["mask"], current_mask), dim=1)
        elif kv_cache is not None and "key" in kv_cache:
            history_length = kv_cache["key"].shape[1]
            key = torch.cat((kv_cache["key"], current_key), dim=1)
            value = torch.cat((kv_cache["value"], current_value), dim=1)
            key_mask = torch.cat((kv_cache["mask"], current_mask), dim=1)
        output = self._compute_attention(
            query,
            key,
            value,
            key_mask,
            text_mask,
            history_length,
            video_length,
            text_length,
            encoder_attention_mask,
            action_text_offsets,
            action_token_windows,
        )
        output = output * current_mask[:, :, None, None].to(output.dtype)
        if update_cache:
            cache_mask = current_mask.clone()
            if cache_action_start is not None:
                cache_mask[:, video_length + cache_action_start :] = False
            elif action_text_offsets is not None:
                for batch_index, offsets in enumerate(action_text_offsets):
                    if offsets is not None:
                        cache_mask[batch_index, video_length + offsets[0] :] = False
            if history_length:
                cache_mask = torch.cat((kv_cache["mask"], cache_mask), dim=1)
            if store is not None:
                store.append(current_key, current_value)
                kv_cache["mask"] = cache_mask
            else:
                kv_cache.update(key=key.detach(), value=value.detach(), mask=cache_mask)
        output = output.flatten(-2)
        hidden_states, encoder_hidden_states = output.split((video_length, text_length), dim=1)
        return (attn.img_attn_proj(hidden_states), attn.txt_attn_proj(encoder_hidden_states))

    def _compute_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        key_mask: Any,
        text_mask: Any,
        history_length: Any,
        video_length: Any,
        text_length: Any,
        encoder_attention_mask: Any,
        action_text_offsets: Any,
        action_token_windows: Any,
    ) -> Any:
        if action_text_offsets is None:
            attention_mask = (
                key_mask[:, None, None, :] if encoder_attention_mask is not None or history_length else None
            )
            output = _attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            outputs = []
            for batch_index, offsets in enumerate(action_text_offsets):
                sample_output = torch.zeros_like(query[batch_index : batch_index + 1])
                sample_text_mask = text_mask[batch_index].bool()
                text_positions = sample_text_mask.cumsum(dim=0) - 1
                valid_text = sample_text_mask.sum()
                if offsets is None:
                    groups = [(0, video_length + text_length, -1)]
                else:
                    groups = list(action_token_windows)
                    groups.append((video_length, video_length + min(offsets[0], text_length), -1))
                    groups.extend(
                        (
                            (video_length + min(start, text_length), video_length + min(end, text_length), index)
                            for index, (start, end) in enumerate(itertools.pairwise(offsets))
                        )
                    )
                for start, end, action_index in groups:
                    if end <= start:
                        continue
                    visible_text = sample_text_mask
                    if offsets is not None:
                        visible_text = text_positions < offsets[0]
                        if action_index >= 0:
                            visible_text = visible_text | (text_positions >= offsets[action_index]) & (
                                text_positions < offsets[action_index + 1]
                            )
                        visible_text = visible_text & sample_text_mask
                    attention_mask = torch.cat((key_mask[batch_index, : history_length + video_length], visible_text))[
                        None, None, None, :
                    ]
                    group_output = _attention(
                        query[batch_index : batch_index + 1, start:end],
                        key[batch_index : batch_index + 1],
                        value[batch_index : batch_index + 1],
                        attn_mask=attention_mask,
                        backend=self._attention_backend,
                        parallel_config=self._parallel_config,
                    )
                    if offsets is not None:
                        query_mask = torch.arange(start, end, device=query.device) < video_length + valid_text
                        group_output = group_output * query_mask[None, :, None, None].to(group_output.dtype)
                    sample_output[:, start:end] = group_output
                outputs.append(sample_output)
            output = torch.cat(outputs, dim=0)
        return output


class HyNomasAttention(nn.Module):
    _default_processor_cls = HyNomasAttnProcessor
    _supports_qkv_fusion = False

    def __init__(self, hidden_size: Any, heads: Any) -> None:
        super().__init__()
        self.heads = heads
        for prefix in ("img", "txt"):
            for projection in ("q", "k", "v", "proj"):
                setattr(self, f"{prefix}_attn_{projection}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"{prefix}_attn_q_norm", RMSNorm(hidden_size // heads, eps=1e-06))
            setattr(self, f"{prefix}_attn_k_norm", RMSNorm(hidden_size // heads, eps=1e-06))
        self.processor = self._default_processor_cls()

    def forward(
        self, hidden_states: Any, encoder_hidden_states: Any, rotary_emb: Any, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, hidden_states, encoder_hidden_states, rotary_emb, **kwargs)


class HyNomasTransformerBlock(nn.Module):
    def __init__(self, hidden_size: Any, heads: Any, mlp_ratio: Any) -> None:
        super().__init__()
        self.img_mod = DiTModulation(hidden_size, factor=6, zero_init=False)
        self.img_norm1 = HyNomasLayerNorm(hidden_size, eps=1e-06, elementwise_affine=False)
        self.img_norm2 = HyNomasLayerNorm(hidden_size, eps=1e-06, elementwise_affine=False)
        self.img_mlp = TransformerMLP(
            hidden_size, int(hidden_size * mlp_ratio), act_layer=partial(nn.GELU, approximate="tanh")
        )
        self.txt_mod = DiTModulation(hidden_size, factor=6, zero_init=False)
        self.txt_norm1 = HyNomasLayerNorm(hidden_size, eps=1e-06, elementwise_affine=False)
        self.txt_norm2 = HyNomasLayerNorm(hidden_size, eps=1e-06, elementwise_affine=False)
        self.txt_mlp = TransformerMLP(
            hidden_size, int(hidden_size * mlp_ratio), act_layer=partial(nn.GELU, approximate="tanh")
        )
        self.attn = HyNomasAttention(hidden_size, heads)

    def forward(
        self, hidden_states: Any, encoder_hidden_states: Any, temb: Any, rotary_emb: Any, attention_kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = self.img_mod(temb).chunk(6, dim=-1)
        text_shift, text_scale, text_gate, text_mlp_shift, text_mlp_scale, text_mlp_gate = self.txt_mod(temb).chunk(
            6, dim=-1
        )
        norm_hidden_states = self.img_norm1(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        norm_encoder_hidden_states = (
            self.txt_norm1(encoder_hidden_states) * (1 + text_scale[:, None]) + text_shift[:, None]
        )
        attention_output, encoder_attention_output = self.attn(
            norm_hidden_states.to(hidden_states.dtype),
            norm_encoder_hidden_states.to(encoder_hidden_states.dtype),
            rotary_emb,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attention_output * gate[:, None]
        hidden_states = (
            hidden_states
            + self.img_mlp(
                (self.img_norm2(hidden_states) * (1 + mlp_scale[:, None]) + mlp_shift[:, None]).to(hidden_states.dtype)
            )
            * mlp_gate[:, None]
        )
        encoder_hidden_states = encoder_hidden_states + encoder_attention_output * text_gate[:, None]
        encoder_hidden_states = (
            encoder_hidden_states
            + self.txt_mlp(
                (self.txt_norm2(encoder_hidden_states) * (1 + text_mlp_scale[:, None]) + text_mlp_shift[:, None]).to(
                    encoder_hidden_states.dtype
                )
            )
            * text_mlp_gate[:, None]
        )
        return (hidden_states, encoder_hidden_states)


class HyNomasCameraAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        camera_hidden_states: Any,
        rotary_emb: Any,
        camera_attention_mask: Any = None,
    ) -> Any:
        batch_size, num_frames = camera_hidden_states.shape[:2]
        query = attn.img_attn_q(hidden_states).unflatten(-1, (attn.heads, -1))
        camera_hidden_states = attn.keyboard_embed(camera_hidden_states)
        key, value = attn.keyboard_attn_kv(camera_hidden_states).unflatten(-1, (2, attn.heads, -1)).unbind(-3)
        query = attn.key_attn_q_norm(query).to(value.dtype)
        key = attn.key_attn_k_norm(key).to(value.dtype)
        tokens_per_frame = hidden_states.shape[1] // num_frames
        query_rotary_emb = tuple(component.repeat_interleave(tokens_per_frame, dim=0) for component in rotary_emb)
        query = _apply_rotary_emb(query, query_rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)
        output = self._compute_attention(query, key, value, num_frames, tokens_per_frame)
        output = output.flatten(-2).reshape(batch_size, -1, attn.camera_attention_dim)
        output = attn.proj_keyboard(output)
        if camera_attention_mask is not None:
            output = output * camera_attention_mask.reshape(batch_size, 1, 1).to(output.dtype)
        return hidden_states + output

    def _compute_attention(self, query: Any, key: Any, value: Any, num_frames: Any, tokens_per_frame: Any) -> Any:
        query = query.unflatten(1, (num_frames, tokens_per_frame)).flatten(0, 1)
        key = key.flatten(0, 1).unsqueeze(1)
        value = value.flatten(0, 1).unsqueeze(1)
        output = _attention(query, key, value, backend=self._attention_backend, parallel_config=self._parallel_config)
        return output


class HyNomasCameraAttention(nn.Module):
    _default_processor_cls = HyNomasCameraAttnProcessor
    _supports_qkv_fusion = False

    def __init__(self, hidden_size: Any, camera_embed_dim: Any, camera_attention_dim: Any, heads: Any) -> None:
        super().__init__()
        self.heads = heads
        self.camera_attention_dim = camera_attention_dim
        self.keyboard_embed = nn.Sequential(
            nn.Linear(64, camera_embed_dim), nn.SiLU(), nn.Linear(camera_embed_dim, camera_embed_dim)
        )
        self.key_attn_q_norm = RMSNorm(camera_attention_dim // heads, eps=1e-06)
        self.key_attn_k_norm = RMSNorm(camera_attention_dim // heads, eps=1e-06)
        self.img_attn_q = nn.Linear(hidden_size, camera_attention_dim, bias=False)
        self.keyboard_attn_kv = nn.Linear(camera_embed_dim, 2 * camera_attention_dim, bias=False)
        self.proj_keyboard = nn.Linear(camera_attention_dim, hidden_size, bias=False)
        self.processor = self._default_processor_cls()

    def forward(
        self, hidden_states: Any, camera_hidden_states: Any, rotary_emb: Any, camera_attention_mask: Any = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, camera_hidden_states, rotary_emb, camera_attention_mask)


class HyNomasPatchEmbed(nn.Module):
    def __init__(self, in_channels: Any, hidden_size: Any) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=1, stride=1)

    def forward(self, hidden_states: Any) -> torch.Tensor:
        return self.proj(hidden_states).flatten(2).transpose(1, 2).contiguous()


class HyNomasFinalLayer(nn.Module):
    def __init__(self, hidden_size: Any, out_channels: Any) -> None:
        super().__init__()
        self.norm_final = HyNomasLayerNorm(hidden_size, eps=1e-06, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, hidden_states: Any, temb: Any) -> torch.Tensor:
        dtype = hidden_states.dtype
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=-1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.linear(hidden_states.to(dtype))


class HyNomasTransformer3DModel(nn.Module):
    """The Hy-Nomas segment autoregressive video transformer with frame-local camera conditioning.

    Args:
        in_channels (`int`, defaults to `32`): Number of latent channels.
        out_channels (`int`, defaults to `32`): Number of output channels.
        num_layers (`int`, defaults to `54`): Number of double stream blocks.
        num_attention_heads (`int`, defaults to `16`): Number of attention heads.
        attention_head_dim (`int`, defaults to `128`): Dimension of an attention head.
        text_embed_dim (`int`, defaults to `3584`): Dimension of text encoder features.
        mlp_ratio (`float`, defaults to `4.0`): MLP expansion ratio.
        rope_axes_dim (`tuple[int]`, defaults to `(16, 56, 56)`): Temporal and spatial RoPE dimensions.
        rope_theta (`float`, defaults to `256.0`): RoPE frequency base.
        camera_embed_dim (`int`, defaults to `128`): Camera embedding dimension.
        camera_attention_dim (`int`, defaults to `1024`): Camera attention projection dimension.
        camera_num_attention_heads (`int`, defaults to `16`): Number of camera attention heads.
        camera_rope_axes_dim (`tuple[int]`, defaults to `(8, 28, 28)`): Camera RoPE dimensions.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 54,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        text_embed_dim: int = 3584,
        mlp_ratio: float = 4.0,
        rope_axes_dim: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 256.0,
        camera_embed_dim: int = 128,
        camera_attention_dim: int = 1024,
        camera_num_attention_heads: int = 16,
        camera_rope_axes_dim: tuple[int, int, int] = (8, 28, 28),
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            text_embed_dim=text_embed_dim,
            mlp_ratio=mlp_ratio,
            rope_axes_dim=rope_axes_dim,
            rope_theta=rope_theta,
            camera_embed_dim=camera_embed_dim,
            camera_attention_dim=camera_attention_dim,
            camera_num_attention_heads=camera_num_attention_heads,
            camera_rope_axes_dim=camera_rope_axes_dim,
        )
        if sum(rope_axes_dim) != attention_head_dim or any(dim % 2 for dim in rope_axes_dim):
            raise ValueError("Even rope_axes_dim entries must sum to attention_head_dim.")
        if camera_attention_dim != camera_num_attention_heads * sum(camera_rope_axes_dim):
            raise ValueError("camera_attention_dim must match camera_num_attention_heads * sum(camera_rope_axes_dim).")
        hidden_size = num_attention_heads * attention_head_dim
        self.img_in = HyNomasPatchEmbed(in_channels, hidden_size)
        self.txt_in = HyNomasTokenRefiner(text_embed_dim, hidden_size, num_attention_heads)
        self.time_in = HyNomasTimestepEmbedder(hidden_size)
        self.rotary_embed = HyNomasRotaryPosEmbed(rope_axes_dim, rope_theta)
        self.camera_rotary_embed = HyNomasRotaryPosEmbed(camera_rope_axes_dim, rope_theta)
        self.double_blocks = nn.ModuleList(
            [HyNomasTransformerBlock(hidden_size, num_attention_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.action_modules = nn.ModuleList(
            [
                HyNomasCameraAttention(hidden_size, camera_embed_dim, camera_attention_dim, camera_num_attention_heads)
                for _ in range(num_layers)
            ]
        )
        self.extrinsics_embedding_camctrl = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer1 = nn.Linear(64, 64)
        self.c2ws_hidden_states_layer2 = nn.Linear(64, 64)
        self.final_layer = HyNomasFinalLayer(hidden_size, out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        kv_cache: list[dict[str, Any]] | None = None,
        update_cache: bool = False,
        frame_offset: int = 0,
        camera_extrinsics: torch.Tensor | None = None,
        camera_attention_mask: torch.Tensor | None = None,
        action_text_offsets: list[list[int] | None] | None = None,
        action_frame_windows: list[tuple[int, int, int]] | None = None,
        cache_action_start: int | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace | tuple[torch.Tensor]:
        """Predict flow velocity for one segment.

        Args:
            hidden_states (`torch.Tensor`):
                Noisy video latents of shape `(batch, channels, frames, height, width)`.
            encoder_hidden_states (`torch.Tensor`):
                Text features of shape `(batch, text_tokens, text_embed_dim)`.
            timestep (`torch.Tensor`):
                Diffusion timesteps of shape `(batch,)` in the reference model's 0–1000 scale.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Boolean text validity mask of shape `(batch, text_tokens)`.
            kv_cache (`list[dict[str, torch.Tensor]]`, *optional*):
                Caller-owned history, with one dictionary per transformer block. Empty dictionaries start a new
                session. Populated entries contain `key`, `value`, and `mask` tensors. The model stores no session
                state.
            update_cache (`bool`, defaults to `False`):
                Append detached keys, values, and a visibility mask to `kv_cache` after attention.
            frame_offset (`int`, defaults to `0`):
                Number of previously generated latent frames, used for video rotary positions.
            camera_extrinsics (`torch.Tensor`, *optional*):
                Normalized poses of shape `(batch, pixel_frames, 4, 4)`. Four consecutive poses form one latent frame;
                the first pose is repeated at the front when necessary.
            camera_attention_mask (`torch.Tensor`, *optional*):
                Per-sample camera gate of shape `(batch,)`.
            action_text_offsets (`list[list[int] | None]`, *optional*):
                Head and action text boundaries for each sample; `None` entries use unrestricted current text.
            action_frame_windows (`list[tuple[int, int, int]]`, *optional*):
                `(start_frame, end_frame, action_index)` intervals covering this segment, with `-1` denoting a gap.
            cache_action_start (`int`, *optional*):
                Text boundary after which tokens become invisible to later segments, independently of action windows.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.modeling_outputs.SimpleNamespace`] or a tuple.

        Returns:
            [`~models.modeling_outputs.SimpleNamespace`] or `tuple[torch.Tensor]`:
                Predicted flow velocity, with the same video geometry as `hidden_states`.
        """
        batch_size, _, num_frames, height, width = hidden_states.shape
        if kv_cache is not None and len(kv_cache) != len(self.double_blocks):
            raise ValueError("kv_cache must contain one dictionary per transformer block.")
        if update_cache and kv_cache is None:
            raise ValueError("update_cache=True requires a caller-owned kv_cache.")
        if action_text_offsets is not None and (len(action_text_offsets) != batch_size or action_frame_windows is None):
            raise ValueError("Action offsets require one entry per sample and action_frame_windows.")
        if cache_action_start is not None and (not 0 <= cache_action_start <= encoder_hidden_states.shape[1]):
            raise ValueError("cache_action_start must be a valid text boundary or None.")
        if kv_cache is not None:
            cache_lengths = []
            for layer_cache in kv_cache:
                if not layer_cache:
                    cache_lengths.append(0)
                    continue
                if "store" in layer_cache:
                    cache_lengths.append(layer_cache["store"].length)
                    continue
                if not {"key", "value", "mask"}.issubset(layer_cache):
                    raise ValueError("Each nonempty cache entry must contain key, value, and mask tensors.")
                key, value, mask = (layer_cache["key"], layer_cache["value"], layer_cache["mask"])
                expected_shape = (
                    batch_size,
                    key.shape[1],
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                )
                if key.shape != expected_shape or value.shape != expected_shape or mask.shape != expected_shape[:2]:
                    raise ValueError("kv_cache tensor shapes must match the model and input batch size.")
                cache_lengths.append(key.shape[1])
            if len(set(cache_lengths)) != 1:
                raise ValueError("All kv_cache layers must contain the same number of tokens.")
        rotary_emb = self.rotary_embed(num_frames, height, width, hidden_states.device, frame_offset)
        hidden_states = self.img_in(hidden_states)
        temb = self.time_in(timestep, hidden_states.dtype)
        encoder_hidden_states = self.txt_in(encoder_hidden_states, timestep, encoder_attention_mask)
        camera_hidden_states = None
        if camera_extrinsics is not None:
            camera_extrinsics = camera_extrinsics.to(device=hidden_states.device, dtype=hidden_states.dtype)
            padding = -camera_extrinsics.shape[1] % 4
            if padding:
                camera_extrinsics = torch.cat(
                    (camera_extrinsics[:, :1].repeat(1, padding, 1, 1), camera_extrinsics), dim=1
                )
            if camera_extrinsics.shape[1] != num_frames * 4:
                raise ValueError("Camera poses must align with the segment's latent frames after front padding.")
            camera_hidden_states = self.extrinsics_embedding_camctrl(
                camera_extrinsics.reshape(batch_size, num_frames, 64)
            )
            camera_hidden_states = camera_hidden_states + self.c2ws_hidden_states_layer2(
                F.silu(self.c2ws_hidden_states_layer1(camera_hidden_states))
            ).to(camera_hidden_states.device)
            camera_rotary_emb = self.camera_rotary_embed(num_frames, 1, 1, hidden_states.device)
        action_token_windows = None
        if action_frame_windows is not None:
            action_token_windows = [
                (start * height * width, end * height * width, index) for start, end, index in action_frame_windows
            ]
        for index, block in enumerate(self.double_blocks):
            attention_kwargs = {
                "encoder_attention_mask": encoder_attention_mask,
                "kv_cache": kv_cache[index] if kv_cache is not None else None,
                "update_cache": update_cache,
                "action_text_offsets": action_text_offsets,
                "action_token_windows": action_token_windows,
                "cache_action_start": cache_action_start,
            }
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary_emb, attention_kwargs=attention_kwargs
            )
            if camera_hidden_states is not None:
                hidden_states = self.action_modules[index](
                    hidden_states, camera_hidden_states, camera_rotary_emb, camera_attention_mask
                )
        hidden_states = self.final_layer(hidden_states, temb)
        output = hidden_states.transpose(1, 2).reshape(batch_size, self.config.out_channels, num_frames, height, width)
        if not return_dict:
            return (output,)
        return SimpleNamespace(sample=output)
