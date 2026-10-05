"""MultiWorld / It-Takes-Two: multi-agent action on native Wan.

Agent-wise self-attention, action RoPE, and a custom DiT block that
cross-attends pooled agent tokens.  Adds action / multi-agent hooks.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from worldfoundry.core.attention import packed_sequence_attention
from worldfoundry.core.attention.backends.dispatch import torch_sdpa
from worldfoundry.core.nn import RMSNorm, scale_shift

from ..model import DiTBlock, WanModel


class MultiAgentActionRoPE1D(nn.Module):
    """Encode the agent index while leaving each frame independent."""

    def __init__(self, dim: int, max_agents: int = 8, base: float = 2.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("MultiWorld action RoPE requires an even hidden dimension")
        frequencies = 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_agents, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        # Released MultiWorld checkpoints contain trainable parameters only.
        self.register_buffer("freqs_cos_n", angles.cos(), persistent=False)
        self.register_buffer("freqs_sin_n", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        even, odd = value[..., ::2], value[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, agents, frames, _ = value.shape
        if agents > self.freqs_cos_n.shape[0]:
            raise ValueError(
                f"MultiWorld received {agents} agents, but action RoPE supports "
                f"at most {self.freqs_cos_n.shape[0]}"
            )
        flat = rearrange(value, "b n f d -> (b f) n d")
        cos = self.freqs_cos_n[:agents].to(flat).repeat_interleave(2, dim=-1)
        sin = self.freqs_sin_n[:agents].to(flat).repeat_interleave(2, dim=-1)
        flat = flat * cos.unsqueeze(0) + self._rotate_half(flat) * sin.unsqueeze(0)
        return rearrange(flat, "(b f) n d -> b n f d", b=batch, f=frames)


class AgentWiseSelfAttention(nn.Module):
    """Checkpoint-compatible attention over agents at each action frame."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_n: int = 8,
        base_n: float = 2.0,
        pe_type: str = "relative1d",
        qk_norm: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("MultiWorld action hidden dimension must divide num_heads")
        if qk_norm:
            raise ValueError("released ItTakesTwo checkpoints use qk_norm=false")
        if pe_type == "relative1d":
            self.rope: nn.Module = MultiAgentActionRoPE1D(dim, max_agents=max_n, base=base_n)
        elif pe_type == "identity":
            self.rope = nn.Identity()
        else:
            raise ValueError(f"unsupported native MultiWorld action pe_type: {pe_type}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _ = value.shape
        value = self.rope(value)
        query, key, item = rearrange(
            self.qkv(value),
            "b n f (q h d) -> q (b f) h n d",
            q=3,
            h=self.num_heads,
        ).unbind(0)
        output = F.scaled_dot_product_attention(
            query,
            key,
            item,
            scale=self.head_dim**-0.5,
        )
        output = rearrange(output, "(b f) h n d -> b n f (h d)", b=batch, f=frames)
        return self.out_proj(output)


class SoftmaxAgentPooling(nn.Module):
    """Fuse per-agent action tokens into one token for every frame."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        adaptive_agent_pooling: bool = True,
        **attention_config: Any,
    ) -> None:
        super().__init__()
        self.agent_attn = AgentWiseSelfAttention(
            dim=dim,
            num_heads=num_heads,
            **attention_config,
        )
        self.adaptive_agent_pooling = bool(adaptive_agent_pooling)
        if self.adaptive_agent_pooling:
            self.weight_proj = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.agent_attn(value).permute(0, 2, 1, 3)
        if not self.adaptive_agent_pooling:
            return value.sum(dim=2)
        weights = torch.softmax(self.weight_proj(value), dim=2)
        return (value * weights).sum(dim=2)


class ItTakesTwoActionEncoder(nn.Module):
    """Encode the released two-player discrete/continuous action schema."""

    def __init__(
        self,
        discrete_dim: int = 10,
        continuous_dim: int = 2,
        output_dim: int = 3072,
        output_ratio: int = 1,
        adaptive_agent_pooling: bool = True,
        action_pe_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.discrete_dim = int(discrete_dim)
        self.continuous_dim = int(continuous_dim)
        self.output_dim = int(output_dim)
        self.output_ratio = int(output_ratio)
        self.discrete_embedding = nn.Embedding(self.discrete_dim * 2, self.output_dim)
        self.continuous_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.continuous_dim, self.output_dim),
        )
        pooling_config = dict(action_pe_config or {})
        pooling_config.setdefault("dim", self.output_dim)
        self.agent_pooling = SoftmaxAgentPooling(
            **pooling_config,
            adaptive_agent_pooling=adaptive_agent_pooling,
        )
        self.output_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim * self.output_ratio),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name and parameter.ndim >= 2:
                nn.init.kaiming_normal_(parameter, nonlinearity="linear")
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, action: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            discrete = action["discrete_action"]
            continuous = action["continuous_action"]
        except KeyError as error:
            raise KeyError(
                "ItTakesTwo actions require discrete_action and continuous_action"
            ) from error
        if discrete.ndim != 4 or continuous.ndim != 4:
            raise ValueError("ItTakesTwo actions must have shape [B,F,2,D]")
        if discrete.shape[:3] != continuous.shape[:3] or discrete.shape[2] != 2:
            raise ValueError("ItTakesTwo discrete/continuous actions must describe two players")
        if discrete.shape[-1] != self.discrete_dim or continuous.shape[-1] != self.continuous_dim:
            raise ValueError("ItTakesTwo action feature dimensions do not match the checkpoint")

        token_ids = torch.arange(self.discrete_dim, device=discrete.device).view(1, 1, 1, -1)
        token_ids = token_ids * 2 + discrete.to(dtype=torch.long)
        discrete_token = self.discrete_embedding(token_ids).sum(dim=3)
        continuous_token = self.continuous_projection(
            continuous.to(dtype=self.continuous_projection[1].weight.dtype)
        )
        action_token = (discrete_token + continuous_token).permute(0, 2, 1, 3)
        return self.output_projection(self.agent_pooling(action_token))


class MultiWorldCrossAttention(nn.Module):
    """Attend to actions and the VGGT environment through one Wan residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        *,
        has_context_input: bool = True,
        use_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.use_causal_mask = bool(use_causal_mask)
        self.has_context_input = bool(has_context_input)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        if self.has_context_input:
            self.k_ctx = nn.Linear(dim, dim)
            self.v_ctx = nn.Linear(dim, dim)
            self.norm_k_ctx = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action is None or action.ndim != 3:
            raise ValueError("MultiWorld action embeddings must have shape [B,F,D]")
        query = self.norm_q(self.q(x))
        key = self.norm_k(self.k(action))
        value = self.v(action)
        mask = None
        if self.use_causal_mask:
            frames = action.shape[1]
            if x.shape[1] % frames:
                raise ValueError("causal MultiWorld attention requires equal tokens per action frame")
            tokens_per_frame = x.shape[1] // frames
            mask = torch.triu(
                torch.full(
                    (frames, frames),
                    float("-inf"),
                    device=x.device,
                    dtype=query.dtype,
                ),
                diagonal=1,
            )
            mask = (
                mask.view(1, 1, frames, 1, frames)
                .expand(x.shape[0], self.num_heads, frames, tokens_per_frame, frames)
                .reshape(x.shape[0], self.num_heads, x.shape[1], frames)
            )
        output = torch_sdpa(
            query,
            key,
            value,
            q_pattern="b s (n d)",
            k_pattern="b s (n d)",
            v_pattern="b s (n d)",
            out_pattern="b s (n d)",
            dims={"n": self.num_heads},
            attn_mask=mask,
        )
        if self.has_context_input:
            if context is None:
                raise ValueError("MultiWorld environment context is required")
            context_key = self.norm_k_ctx(self.k_ctx(context))
            context_value = self.v_ctx(context)
            output = output + packed_sequence_attention(
                query,
                context_key,
                context_value,
                num_heads=self.num_heads,
            )
        return self.o(output)


class MultiWorldDiTBlock(DiTBlock):
    """Canonical Wan block with MultiWorld cross-attention semantics."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        has_context_input: bool = True,
        use_causal_mask: bool = False,
    ) -> None:
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        self.cross_attn = MultiWorldCrossAttention(
            dim,
            num_heads,
            eps,
            has_context_input=has_context_input,
            use_causal_mask=use_causal_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        action_embeds: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # WanModel forwards its exact-RoPE policy through every block. Keep
        # that policy when MultiWorld replaces the standard Wan cross-attn.
        kwargs.pop("_worldfoundry_inplace_residual", None)
        sequence_timestep = t_mod.ndim == 4
        chunk_dim = 2 if sequence_timestep else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if sequence_timestep:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                item.squeeze(2)
                for item in (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            )
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(scale_shift(self.norm1(x), shift_msa, scale_msa), freqs, **kwargs),
        )
        x = x + self.cross_attn(self.norm3(x), action_embeds, context)
        return self.gate(
            x,
            gate_mlp,
            self.ffn(scale_shift(self.norm2(x), shift_mlp, scale_mlp)),
        )


class MultiWorldWanModel(WanModel):
    """Released MultiWorld DiT expressed as a checkpoint-compatible Wan role."""

    def __init__(
        self,
        *args: Any,
        action_injection: str = "bidi_cross_attention",
        action_encoder_config: Mapping[str, Any] | None = None,
        has_context_input: bool = True,
        **kwargs: Any,
    ) -> None:
        if action_injection not in {"bidi_cross_attention", "causal_cross_attention"}:
            raise ValueError(f"unsupported native MultiWorld action injection: {action_injection}")
        if action_encoder_config is None:
            raise ValueError("MultiWorldWanModel requires action_encoder_config")
        dim = int(kwargs.get("dim", args[0] if args else 0))
        if not dim:
            raise ValueError("MultiWorldWanModel requires the Wan hidden dimension")
        kwargs["per_token_timestep"] = True
        super().__init__(
            *args,
            block_class=MultiWorldDiTBlock,
            block_kwargs={
                "has_context_input": bool(has_context_input),
                "use_causal_mask": action_injection == "causal_cross_attention",
            },
            **kwargs,
        )
        # MultiWorld consumes already projected VGGT tokens; its release does
        # not contain Wan's text projection parameters.
        self.text_embedding = nn.Identity()
        config = dict(action_encoder_config)
        config.setdefault("output_dim", dim)
        self.action_encoder = ItTakesTwoActionEncoder(**config)
        self.action_injection = action_injection
        self.has_context_input = bool(has_context_input)

    def prepare_condition_context(
        self,
        context: torch.Tensor | None,
        *,
        env_context: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del context, kwargs
        if env_context is None:
            raise ValueError("MultiWorld inference requires VGGT environment context")
        if env_context.shape[-1] != self.dim:
            raise ValueError(
                f"MultiWorld environment context width must be {self.dim}; "
                f"got {env_context.shape[-1]}"
            )
        return env_context

    def block_forward_kwargs(
        self,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del grid_size
        action_embeds = kwargs.get("action_embeds")
        if action_embeds is None:
            raise ValueError("MultiWorld inference requires encoded actions")
        return {"action_embeds": action_embeds}


__all__ = [
    "AgentWiseSelfAttention",
    "ItTakesTwoActionEncoder",
    "MultiAgentActionRoPE1D",
    "MultiWorldCrossAttention",
    "MultiWorldDiTBlock",
    "MultiWorldWanModel",
    "SoftmaxAgentPooling",
]
