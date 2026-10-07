# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# WorldPlay2 adaptations are licensed under CC-BY-NC-4.0.
"""WorldPlay2 additions to the shared packed Wan2.2 transformer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
    DiTBlock,
    Head,
    WanModel,
    apply_wan_qk_norm_rope,
)
from worldfoundry.core.nn import sinusoidal_embedding_1d

from .compressor import CausalMemCompressModel


class WorldPlay2AttentionProcessor:
    """Attend to request-owned ``[sink | compressed history | latest]`` KV."""

    def __init__(self, block_index: int) -> None:
        self.block_index = block_index

    def __call__(self, attention, x, freqs, **kwargs):
        q, k = apply_wan_qk_norm_rope(
            attention, attention.q(x), attention.k(x), freqs,
            precision=kwargs.get("_worldfoundry_rope_precision", "fp64"),
        )
        v = attention.v(x)
        state = kwargs.get("_worldfoundry_attention_context")
        if state is None:
            return attention.o(attention.attn(q, k, v))
        cache = state["cache"][self.block_index]
        if not state.get("write", False):
            if cache["k_vision"] is not None:
                k = torch.cat((cache["k_vision"].to(k), k), dim=1)
                v = torch.cat((cache["v_vision"].to(v), v), dim=1)
            return attention.o(attention.attn(q, k, v))

        prefix = int(state.get("prefix_tokens", 0))
        if prefix:
            k = torch.cat((cache["k_vision"][:, :prefix].to(k), k), dim=1)
            v = torch.cat((cache["v_vision"][:, :prefix].to(v), v), dim=1)
        sink = int(state.get("sink_tokens", 0)) if not prefix else 0
        compressed = int(state["new_compressed_tokens"])
        group = int(state["compressed_chunk_tokens"])
        mask = torch.zeros(x.shape[1], k.shape[1], device=x.device, dtype=torch.bool)
        if sink:
            mask[:sink, :sink] = True
        for start in range(0, compressed, group):
            end = min(start + group, compressed)
            mask[sink + start:sink + end, :prefix + sink + end] = True
        mask[sink + compressed:, :] = True
        heads = attention.num_heads
        q, k_heads, v_heads = (
            value.unflatten(-1, (heads, -1)).transpose(1, 2)
            for value in (q, k, v)
        )
        output = F.scaled_dot_product_attention(q, k_heads, v_heads, attn_mask=mask)
        cache["k_vision"], cache["v_vision"] = k.detach(), v.detach()
        return attention.o(output.transpose(1, 2).flatten(2))


class WorldPlay2Block(DiTBlock):
    """Inject the released action MLP between shared cross-attention and FFN."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_action_mlp = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim),
        )

    def forward(self, x, context=None, t_mod=None, freqs=None, *, vec_action=None, **kwargs):
        x, modifiers = super().forward(
            x, context, t_mod, freqs, return_partial=True, **kwargs,
        )
        if vec_action is not None:
            action_dtype = self.img_action_mlp[0].weight.dtype
            x = x + self.img_action_mlp(torch.cat((x, vec_action.to(x)), dim=-1).to(action_dtype))
        return super().forward(x, run_remaining=True, modifiers=modifiers)


class FixedPDDHead(Head):
    """The two compact displacement/endpoint projections in each Fast expert."""

    def __init__(self, dim, out_dim, patch_size, eps) -> None:
        super().__init__(dim, out_dim, patch_size, eps)
        del self.head
        output_width = 2 * out_dim * math.prod(patch_size)
        self.block_weight = nn.Parameter(torch.empty(2, output_width, dim))
        self.block_bias = nn.Parameter(torch.empty(2, output_width))

    def forward(self, x, t_mod, *, block_index: int) -> torch.Tensor:
        if block_index not in (0, 1):
            raise ValueError("WorldPlay2 PDD block index must be 0 or 1")
        with torch.autocast(device_type=x.device.type, enabled=False):
            shift, scale = (self.modulation.float().unsqueeze(0) + t_mod.float().unsqueeze(2)).unbind(2)
            value = self.norm(x.float()) * (1 + scale) + shift
            value = F.linear(
                value.to(self.block_weight.dtype), self.block_weight[block_index], self.block_bias[block_index],
            )
        first, second = value.chunk(2, dim=-1)
        patch_volume = math.prod(self.patch_size)
        # Unpatchify expects patch dimensions before channels.
        return torch.cat((first.unflatten(-1, (patch_volume, -1)),
                          second.unflatten(-1, (patch_volume, -1))), dim=-1).flatten(-2)


class WorldPlay2Model(WanModel):
    """Reuse Wan patchify, timestep, text, transformer traversal and unpatchify."""

    def __init__(self, *, text_len=512, fixed_pdd=False, compressor_options=None, **kwargs) -> None:
        super().__init__(block_class=WorldPlay2Block, per_token_timestep=True, **kwargs)
        self.text_len = int(text_len)
        self.fixed_pdd = bool(fixed_pdd)
        self.action_in_cont = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, self.dim),
        )
        self.action_in_disc = nn.ModuleList(nn.Embedding(size, self.dim) for size in (3, 3, 2, 2))
        self.memory_compress = CausalMemCompressModel(
            output_dim=self.dim, input_dim=self.in_dim,
            spatial_down=(1, 1, 1, 0, 0, 0), temporal_down=(1, 0, 0, 0, 0, 0),
            **dict(compressor_options or {}),
        )
        for index, block in enumerate(self.blocks):
            block.self_attn.set_processor(WorldPlay2AttentionProcessor(index))
        if fixed_pdd:
            self.head = FixedPDDHead(self.dim, self.out_dim, self.patch_size, kwargs["eps"])

    def encode_actions(self, action: torch.Tensor) -> torch.Tensor:
        action = action.reshape(-1, 6)
        value = self.action_in_cont(action[:, :2].to(self.action_in_cont[0].weight.dtype))
        for index, offset in enumerate((1, 1, 0, 0)):
            value = value + self.action_in_disc[index](action[:, index + 2].long() + offset)
        return value

    def _frame_features(self, action, batch, spatial):
        return self.encode_actions(action).reshape(batch, -1, self.dim).repeat_interleave(spatial, dim=1)

    def _time_features(self, timestep, batch, tokens):
        time_dtype = self.time_embedding[0].weight.dtype
        value = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(time_dtype))
        value = value[:, None].expand(batch, tokens, self.dim)
        return value.float(), self.time_projection(value).unflatten(-1, (6, self.dim)).float()

    def prepare_token_sequence(self, x, freqs, t_mod, t, grid_size, **kwargs):
        del freqs
        frames, height, width = grid_size
        shift = int(kwargs.get("chunk_start", 0))
        memory = kwargs.get("memory")
        if memory is not None:
            shift = int(memory["rope_shift_idx"])
        freqs = self.rotary_frequencies((shift + frames, height, width), device=x.device)[-x.shape[1]:]
        t = t[:, None].expand(-1, x.shape[1], -1).float() if t.ndim == 2 else t.float()
        t_mod = t_mod[:, None].expand(-1, x.shape[1], -1, -1).float() if t_mod.ndim == 3 else t_mod.float()
        target_tokens = x.shape[1]
        if kwargs.get("cache") is None and memory is not None:
            x = torch.cat((memory["tokens"], x), dim=1)
            freqs = torch.cat((memory["freqs"], freqs), dim=0)
            t_mod = torch.cat((memory["e0"], t_mod), dim=1)
            t = torch.cat((memory["e"], t), dim=1)
        return x, freqs, t_mod, t, target_tokens

    def block_forward_kwargs(self, grid_size, **kwargs):
        action = kwargs["actions"]
        batch = int(kwargs.get("batch_size", 1))
        vec_action = self._frame_features(action, batch, grid_size[1] * grid_size[2])
        memory = kwargs.get("memory")
        cache = kwargs.get("cache")
        if cache is None and memory is not None:
            vec_action = torch.cat((memory["vec_action"], vec_action), dim=1)
        return {
            "vec_action": vec_action,
            "_worldfoundry_attention_context": {"cache": cache} if cache is not None else None,
        }

    def finalize_token_sequence(self, x, token_state, **kwargs):
        del kwargs
        return x[:, -token_state:], None

    def project_output(self, x, t, **kwargs):
        t = t[:, -x.shape[1]:]
        if self.fixed_pdd:
            return self.head(x, t, block_index=int(kwargs["pdd_block_index"]))
        return super().project_output(x, t)

    def forward(self, x=None, timestep=None, context=None, *, operation="denoise", **kwargs):
        if operation == "compress":
            return self.compress_memory(**kwargs)
        if operation == "prefill":
            return self.prefill(kwargs["memory"], context, kwargs["cache"])
        return super().forward(x, timestep, context, batch_size=x.shape[0], **kwargs)

    def new_cache(self):
        return [{"k_vision": None, "v_vision": None} for _ in self.blocks]

    def compress_memory(self, *, hr_input, lr_input, lr_actions, temporal_input,
                        temporal_actions, sink_actions=None):
        """Fuse HR compressor features with LR Wan tokens at temporal stride two."""
        batch, _, history_frames, _, _ = hr_input.shape
        lr_frames, lr_height, lr_width = lr_input.shape[2:]
        cmp_height, cmp_width = lr_height // self.patch_size[1], lr_width // self.patch_size[2]
        cmp_spatial = cmp_height * cmp_width
        compressed, _ = self.patchify(lr_input)
        compressed = compressed + self.memory_compress(hr_input)
        compressed_freqs = self.rotary_frequencies(
            (history_frames, cmp_height, cmp_width), device=hr_input.device,
        ).reshape(history_frames, cmp_spatial, 1, -1)[::2][:lr_frames].flatten(0, 1)
        temporal, temporal_grid = self.patchify(temporal_input)
        temporal_frames, temporal_height, temporal_width = temporal_grid
        temporal_freqs = self.rotary_frequencies(
            (history_frames, temporal_height, temporal_width), device=hr_input.device,
        )[-temporal.shape[1]:]
        token_parts, freq_parts, action_parts = [], [], []
        sink_tokens = 0
        if sink_actions is not None and sink_actions.shape[-2]:
            count = sink_actions.shape[-2]
            sink, sink_grid = self.patchify(hr_input[:, :, :count])
            token_parts.append(sink)
            freq_parts.append(self.rotary_frequencies(sink_grid, device=hr_input.device))
            action_parts.append(self._frame_features(sink_actions, batch, sink_grid[1] * sink_grid[2]))
            sink_tokens = sink.shape[1]
        token_parts.extend((compressed, temporal))
        freq_parts.extend((compressed_freqs, temporal_freqs))
        action_parts.extend((self._frame_features(lr_actions, batch, cmp_spatial),
                             self._frame_features(temporal_actions, batch, temporal_height * temporal_width)))
        tokens = torch.cat(token_parts, dim=1)
        time, modulation = self._time_features(hr_input.new_zeros(1), batch, tokens.shape[1])
        return {
            "tokens": tokens, "freqs": torch.cat(freq_parts), "e": time, "e0": modulation,
            "vec_action": torch.cat(action_parts, dim=1), "rope_shift_idx": history_frames,
            "sink_token_count": sink_tokens, "cmp_token_count": compressed.shape[1],
            "cmp_chunk_token_count": 2 * cmp_spatial, "tmp_token_count": temporal.shape[1],
        }

    def prefill(self, memory, context, cache):
        """Append new compressed tokens and replace the previous temporal anchor."""
        sink, compressed, recent = (int(memory[key]) for key in
                                    ("sink_token_count", "cmp_token_count", "tmp_token_count"))
        existing = cache[0]["k_vision"]
        prefix = 0
        cached_compressed = 0
        if existing is not None:
            cached_compressed = existing.shape[1] - sink - recent
            prefix = sink + cached_compressed
            if not 0 <= cached_compressed <= compressed:
                raise ValueError("WorldPlay2 cache does not match the compressed history")
        indices = slice(prefix, sink + compressed + recent) if existing is not None else slice(None)
        x = memory["tokens"][:, indices]
        actions = memory["vec_action"][:, indices]
        modulation = memory["e0"][:, indices]
        freqs = memory["freqs"][indices]
        text = self.prepare_condition_context(context)
        attention_context = {
            "cache": cache, "write": True, "prefix_tokens": prefix,
            "sink_tokens": sink, "new_compressed_tokens": compressed - cached_compressed,
            "compressed_chunk_tokens": int(memory["cmp_chunk_token_count"]),
        }
        for block in self.blocks:
            x = block(x, text, modulation, freqs, vec_action=actions,
                      _worldfoundry_attention_context=attention_context)
        return cache


__all__ = ["WorldPlay2Model", "FixedPDDHead"]
