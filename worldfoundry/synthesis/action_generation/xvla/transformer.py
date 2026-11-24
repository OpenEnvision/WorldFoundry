"""X-VLA soft-prompted action transformer.

Adapted from ``models/transformer.py`` in 2toINF/X-VLA at revision
``6bc2513f5f1cbec715cc668b414392a6cae5c671``.  Parameter names remain
checkpoint-compatible while shared MLP, domain projection, sinusoidal
embedding, and GPU-aware attention math come from :mod:`worldfoundry.core`.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from worldfoundry.core.nn import (
    DomainAwareLinear,
    Mlp,
    QKNormRopeSelfAttention,
    sinusoidal_embedding_1d,
)


def _initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TransformerBlock(nn.Module):
    """Checkpoint-compatible pre-normalized X-VLA transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = QKNormRopeSelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.0,
            qk_norm=False,
            fused_attn=True,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=0.1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class SoftPromptedTransformer(nn.Module):
    """Multimodal action transformer with per-embodiment projections."""

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dim_action = int(dim_action)
        self.dim_time = int(dim_time)
        self.len_soft_prompts = int(len_soft_prompts)
        self.use_hetero_proj = bool(use_hetero_proj)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        if use_hetero_proj:
            self.vlm_proj = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj = DomainAwareLinear(
                multi_modal_input_size,
                hidden_size,
                num_domains=num_domains,
            )
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio,
            hidden_size,
            num_domains=num_domains,
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)
        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
        self.apply(_initialize_linear)

    def forward(
        self,
        *,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_actions = action_with_noise.shape[:2]
        time = sinusoidal_embedding_1d(self.dim_time, t, max_period=100.0)
        time = time.unsqueeze(1).expand(batch, num_actions, self.dim_time)
        proprio = proprio.unsqueeze(1).expand(batch, num_actions, proprio.shape[-1])
        action_tokens = torch.cat((action_with_noise, proprio, time), dim=-1)
        hidden_states = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            vlm_features = self.vlm_proj(vlm_features, domain_id)
            aux_visual_inputs = self.aux_visual_proj(aux_visual_inputs, domain_id)
        else:
            vlm_features = self.vlm_proj(vlm_features)
            aux_visual_inputs = self.aux_visual_proj(aux_visual_inputs)
        hidden_states = torch.cat((hidden_states, vlm_features, aux_visual_inputs), dim=1)

        sequence_length = int(hidden_states.shape[1])
        if sequence_length > int(self.pos_emb.shape[1]):
            raise ValueError(f"X-VLA sequence length {sequence_length} exceeds max_len_seq={self.pos_emb.shape[1]}")
        hidden_states = hidden_states + self.pos_emb[:, :sequence_length]
        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(
                batch,
                self.len_soft_prompts,
                self.hidden_size,
            )
            hidden_states = torch.cat((hidden_states, soft_prompts), dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.action_decoder(self.norm(hidden_states[:, :num_actions]), domain_id)


__all__ = ["SoftPromptedTransformer", "TransformerBlock"]
