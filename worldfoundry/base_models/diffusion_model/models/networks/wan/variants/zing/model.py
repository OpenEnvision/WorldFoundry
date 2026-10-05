# Adapted from seedleap/zing-world-model (Apache-2.0).
"""Wan 2.2 backbone with Zing's action and prompt-switch cache protocol.

All generic Wan weights/layers are constructed by reference_22, not copied here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...reference_22 import Head, sinusoidal_embedding_1d
from ...reference_22 import WanAttentionBlock as BaseWanBlock
from ...reference_22 import WanModel as BaseWanModel
from .action import ActionConditioner
from .attention import CompiledSegment, CrossAttention, SelfAttention, compute_rope, make_rope_freqs
from .config import ZingConfig
from .kv_cache import CausalKVCache


def adaln_op(
    x,
    m_shift=None,
    m_scale=None,
    e_shift=None,
    e_scale=None,
    eps=1.0e-6,
    y=None,
    m_gate=None,
    e_gate=None,
    r=None,
    weight=None,
    bias=None,
    cast_norm=False,
):
    if y is not None:
        x = (x.float() + y.float() * (m_gate.float() + e_gate.float())).type_as(x)
    if r is not None:
        x = x + r
    if m_shift is None:
        return x
    h = F.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    )
    shift, scale = m_shift + e_shift, m_scale + e_scale
    if cast_norm:
        h = h.type_as(x)
        return x, h * (1 + scale) + shift
    return x, (h * (1 + scale.float()) + shift.float()).type_as(x)


def adaln(value: torch.Tensor, compile_fusion: bool, *args, **kwargs):
    function = CompiledSegment.get(adaln_op, compile_fusion and value.is_cuda)
    return function(value, *args, **kwargs)


class WanAttentionBlock(BaseWanBlock):
    def __init__(self, config, deterministic):
        super().__init__(
            config.dim,
            config.ffn_dim,
            config.num_heads,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
        )
        self.compile_fusion = config.compile_fusion
        self.self_attn = SelfAttention(
            config.dim, config.num_heads, config.eps, config.qk_norm, config.compile_fusion, deterministic
        )
        self.cross_attn = CrossAttention(
            config.dim, config.num_heads, config.eps, config.qk_norm, config.compile_fusion, deterministic
        )

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        self_history: tuple[torch.Tensor | None, torch.Tensor | None],
        cross_history: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        _, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 0],
            self.modulation[:, 1],
            time_embedding.select(-2, 0),
            time_embedding.select(-2, 1),
            self.norm1.eps,
        )
        attended, key, value = self.self_attn(normalized, rope[0], rope[1], self_history)
        hidden = adaln(
            hidden,
            self.compile_fusion,
            y=attended,
            m_gate=self.modulation[:, 2],
            e_gate=time_embedding.select(-2, 2),
        )
        crossed, new_cross = self.cross_attn(self.norm3(hidden), context, context_lengths, cross_history)
        hidden, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 3],
            self.modulation[:, 4],
            time_embedding.select(-2, 3),
            time_embedding.select(-2, 4),
            self.norm2.eps,
            r=crossed,
        )
        feed_forward = self.ffn(normalized)
        hidden = adaln(
            hidden,
            self.compile_fusion,
            y=feed_forward,
            m_gate=self.modulation[:, 5],
            e_gate=time_embedding.select(-2, 5),
        )
        return hidden, (key, value), new_cross


class WanHead(Head):
    def __init__(self, config):
        super().__init__(config.dim, config.out_dim, config.patch_size, config.eps)
        self.compile_fusion = config.compile_fusion

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        _, normalized = adaln(
            hidden,
            self.compile_fusion,
            self.modulation[:, 0],
            self.modulation[:, 1],
            embedding,
            embedding,
            self.norm.eps,
        )
        return self.head(normalized)


class WanModel(BaseWanModel):
    def __init__(self, config: ZingConfig):
        g = config.generator
        super().__init__(
            model_type="ti2v",
            patch_size=tuple(g.patch_size),
            in_dim=g.in_dim,
            out_dim=g.out_dim,
            dim=g.dim,
            ffn_dim=g.ffn_dim,
            freq_dim=g.freq_dim,
            text_dim=g.text_dim,
            num_heads=g.num_heads,
            num_layers=0,
            qk_norm=g.qk_norm,
            cross_attn_norm=g.cross_attn_norm,
            eps=g.eps,
        )
        self.zing_config = g
        self.rope_max_seq_len = g.rope_max_seq_len
        self.frames_per_block = config.inference.frames_per_block
        self.blocks = nn.ModuleList(
            [WanAttentionBlock(g, config.inference.deterministic_attention) for _ in range(g.num_layers)]
        )
        self.head = WanHead(g)
        self.freqs_t = self.freqs_h = self.freqs_w = None
        self.action_in = ActionConditioner(config.action, g.dim)
        self.action_history_frames = 2 * (config.action.kernel_size - 1)

    def make_kv_cache(self) -> CausalKVCache:
        return CausalKVCache(
            len(self.blocks),
            self.action_history_frames,
            local_attn_size=self.zing_config.local_attn_size,
            sink_size=self.zing_config.sink_size,
            frames_per_block=self.frames_per_block,
        )

    def _rope(self, positions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.freqs_t is None:
            self.freqs_t, self.freqs_h, self.freqs_w = make_rope_freqs(
                self.zing_config.dim, self.zing_config.num_heads, self.rope_max_seq_len
            )
        if self.freqs_t.device != reference.device:
            self.freqs_t = self.freqs_t.to(reference.device)
            self.freqs_h = self.freqs_h.to(reference.device)
            self.freqs_w = self.freqs_w.to(reference.device)
        return compute_rope(positions, self.freqs_t, self.freqs_h, self.freqs_w)

    def _pack_embed(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        grids = []
        for sample in inputs:
            embedded = self.patch_embedding(sample.unsqueeze(0))
            grids.append(torch.tensor(embedded.shape[2:], dtype=torch.long, device=inputs.device))
            tokens.append(embedded.flatten(2).transpose(1, 2).squeeze(0))
        return torch.cat(tokens, dim=0), torch.stack(grids)

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cache: CausalKVCache,
        cache_mode: str | None,
        action: torch.Tensor | None = None,
        prompt_switch: bool = False,
    ) -> torch.Tensor:
        packed, grid_sizes = self._pack_embed(inputs)
        batch = inputs.shape[0]
        sequence = packed.shape[0] // batch
        hidden = packed.view(batch, sequence, -1)
        positions, block_id, action_window = cache.prepare(grid_sizes, inputs.device, action, prompt_switch)
        if action_window is not None:
            hidden = self.action_in.add_action_to_tokens(hidden, action_window, grid_sizes, align_to_end=True)
        frames, height, width = (int(value) for value in grid_sizes[0].tolist())
        spatial_tokens = height * width
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(batch, frames)
        timestep = timestep[:, :frames]
        frame_index = torch.arange(frames, device=inputs.device).repeat_interleave(spatial_tokens)
        embedded_time = self.time_embedding(
            sinusoidal_embedding_1d(self.zing_config.freq_dim, timestep.reshape(-1)).to(hidden.dtype)
        )
        time_projection = self.time_projection(embedded_time).view(batch, frames, 6, -1)[:, frame_index]
        head_embedding = embedded_time.view(batch, frames, -1)[:, frame_index]
        query_positions, key_positions = cache.attention_positions(positions)
        rope = self._rope(query_positions, hidden), self._rope(key_positions, hidden)
        embedded_context = self.text_embedding(context)
        new_self = []
        new_cross = []
        for index, block in enumerate(self.blocks):
            hidden, self_values, cross_values = block(
                hidden,
                time_projection,
                rope,
                embedded_context,
                context_lengths,
                cache.history(index),
                cache.cross(index),
            )
            new_self.append(self_values)
            new_cross.append(cross_values)
        cache.update(new_self, new_cross, positions, block_id, cache_mode, action)
        output = self.head(hidden, head_embedding)
        return torch.stack([self._unpatchify(output[index], grid_sizes[index]) for index in range(batch)])

    def _unpatchify(self, tokens, grid):
        return self.unpatchify(tokens.unsqueeze(0), grid.unsqueeze(0))[0]
