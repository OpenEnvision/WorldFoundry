import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.nn.transformer.dit import modulate_sequence as _modulate

from ..config import BaseConfig, D


class ActionExpertAttention(nn.Module):
    """Multi-head attention with optional cross-attention source."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden size must be divisible by num heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input queries.
            kv: (B, M, C) key/value source. Defaults to self-attention.
            attn_mask: optional mask broadcastable to (B, num_heads, N, M).
        """
        if kv is None:
            kv = x

        bsz, tgt_len, _ = x.shape
        src_len = kv.shape[1]

        q = self.q_proj(x)
        kv = self.kv_proj(kv)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = kv.view(bsz, src_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # PyTorch selects FlashAttention, memory-efficient attention, or its
        # portable math fallback for the active GPU. This covers Ampere,
        # Ada, Hopper and newer architectures without a compiled extension.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ActionExpertMLP(nn.Module):
    """Feed-forward block used inside the ActionExpert transformer."""

    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ActionExpertBlock(nn.Module):
    """A single DiT-style block with AdaLN-Zero modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = ActionExpertAttention(hidden_size, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = ActionExpertAttention(hidden_size, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = ActionExpertMLP(hidden_size, mlp_ratio, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
        cross_context: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(timestep_embed).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn1(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mca.unsqueeze(1) * self.attn2(
            _modulate(self.norm2(x), shift_mca, scale_mca),
            kv=cross_context,
            attn_mask=attn_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class ActionExpertFinalLayer(nn.Module):
    """Final projection with AdaLN modulation."""

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, timestep_embed: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(timestep_embed).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for continuous diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=timesteps.dtype) * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class ActionExpertConfig(BaseConfig):
    """Configuration for the action expert DiT module."""

    max_horizon: int = 32
    """Maximum sequence length for the action tokens."""

    action_dim: int = 14
    """Dimensionality of each action vector."""

    hidden_size: int = 1024
    """Transformer hidden size."""

    num_layers: int = 32
    """Number of transformer blocks."""

    num_heads: int = 16
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """Width multiplier for the feed-forward layer."""

    timestep_embed_dim: int = 256
    """Size of the sinusoidal timestep embedding."""

    dropout: float = 0.0
    """Dropout rate inside the attention and MLP projections."""

    attn_dropout: float = 0.0
    """Dropout applied to attention weights."""

    context_layer_norm: bool = True
    """Whether to normalize projected context embeddings."""

    def build(self, llm_dim: int, device=None) -> "ActionExpert":
        return ActionExpert(self, llm_dim=llm_dim, device=device)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        return config


class ActionExpert(nn.Module):
    """DiT-style transformer that predicts action trajectories conditioned on VLM features."""

    def __init__(self, config: ActionExpertConfig, llm_dim: int, device=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.action_embed = nn.Linear(config.action_dim, config.hidden_size, device=device)
        self.action_pos_embed = nn.Parameter(torch.zeros(1, config.max_horizon, config.hidden_size, device=device))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
        )
        self.context_proj = nn.Linear(llm_dim, config.hidden_size, bias=False, device=device)
        self.context_norm = (
            nn.LayerNorm(config.hidden_size, elementwise_affine=False) if config.context_layer_norm else nn.Identity()
        )
        self.state_encoder = nn.Linear(config.hidden_size, config.hidden_size, device=device)
        self.state_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.action_dim)

    def _encode_states(self, states: torch.Tensor | None) -> torch.Tensor | None:
        if states is None:
            return None
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if states.shape[-1] != self.hidden_size:
            feat_dim = states.shape[-1]
            if feat_dim < self.hidden_size:
                pad = self.hidden_size - feat_dim
                states = F.pad(states, (0, pad))
            else:
                states = states[..., : self.hidden_size]
        encoded = self.state_encoder(states)
        return self.state_norm(encoded)

    def _prepare_context(
        self,
        encoder_hidden_states: Sequence[torch.Tensor],
        encoded_states: torch.Tensor | None,
    ) -> Sequence[torch.Tensor]:
        contexts = []
        for hidden in encoder_hidden_states:
            ctx = self.context_proj(hidden)
            ctx = self.context_norm(ctx)
            if encoded_states is not None:
                ctx = torch.cat([ctx, encoded_states], dim=1)
            contexts.append(ctx)
        return contexts

    def _build_cross_attention_mask(
        self,
        encoder_attention_mask: torch.Tensor | None,
        encoded_states: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Build a mask for cross-attention operations.

        This function creates an attention mask that determines which tokens in the context
        (encoder outputs and optional state embeddings) each token in the sequence can attend to.

        Args:
            encoder_attention_mask: Optional mask for encoder tokens. If provided, should be
                a tensor of shape (B, L) or (B, 1, 1, L) where L is the encoder sequence length.
            encoded_states: Optional tensor containing encoded state information. If provided,
                the mask will be extended to include these states.
            batch_size: The batch size for creating masks.
            dtype: The data type for the output mask, typically matching the model's compute dtype.

        Returns:
            A tensor of shape (B, 1, 1, L+S) where L is the encoder sequence length and S is the
            state sequence length. Values are 0 for tokens that can be attended to and a large
            negative number (close to -inf) for tokens that should be ignored. Returns None if
            encoder_attention_mask is None.
        """
        # Determine the sequence length of encoded states (if any)
        state_seq_len = 0 if encoded_states is None else encoded_states.shape[1]

        # If no encoder mask is provided, no masking is needed
        if encoder_attention_mask is None:
            return None

        # Reshape 2D mask (B, L) to 4D (B, 1, 1, L) for compatibility with attention operations
        if encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        else:
            # If mask is already in the correct shape, just convert to the right dtype
            mask = encoder_attention_mask.to(dtype=dtype)

        # If we have encoded states, extend the mask to include them
        # We set all state tokens as visible (1s in the mask) to allow attending to all state information
        if state_seq_len > 0:
            ones = torch.ones(
                batch_size,
                1,
                1,
                state_seq_len,
                device=mask.device,
                dtype=mask.dtype,
            )
            # Concatenate the encoder mask with the all-ones state mask
            mask = torch.cat([mask, ones], dim=-1)

        # Convert from attention mask (1=attend, 0=ignore) to additive attention mask
        # (0=attend, -inf=ignore) by inverting and scaling to a large negative number
        return (1.0 - mask) * torch.finfo(dtype).min

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Sequence[torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
        states_mode: str = "cross_attn",
    ) -> torch.Tensor:
        """Forward pass of the ActionExpert model.

        This function processes action sequences through a transformer-based architecture
        to predict denoised actions in a diffusion model setup.

        Args:
            actions: (B, T, action_dim) noisy / sample actions.
            timesteps: (B,) or (B, 1, 1) diffusion timesteps in [0, 1].
            encoder_hidden_states: sequence of tensors with length == num_layers.
            encoder_attention_mask: optional boolean mask over encoder tokens.
            state_embeddings: optional tensor containing state information to condition the model.
        Returns:
            (B, T, action_dim) prediction of denoised actions.
        """

        # Extract batch size and sequence length from input actions
        bsz, seq_len, _ = actions.shape

        # Validate sequence length against configured maximum horizon
        if seq_len > self.config.max_horizon:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds configured max_horizon={self.config.max_horizon}"
            )

        # Ensure the number of encoder hidden states matches the number of transformer blocks
        if len(encoder_hidden_states) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} encoder hidden states, got {len(encoder_hidden_states)}")

        # Embed diffusion timesteps to condition the model
        timestep_embed = self.time_embed(timesteps)

        # Embed input actions
        x = self.action_embed(actions)

        # Process optional state embeddings if provided
        encoded_states = self._encode_states(state_embeddings)

        # Handle different states modes
        if states_mode == "self_attn":
            assert encoded_states is not None, "State embeddings must be provided when states_mode is 'self_attn'"

            # For self_attn mode, prepend encoded states to the action sequence
            x = torch.cat([encoded_states, x], dim=1)

            # Apply positional embeddings: states get 0th position, actions get positions 1, 2, 3, ...
            state_seq_len = encoded_states.shape[1]
            total_seq_len = state_seq_len + seq_len

            # Validate that total sequence length doesn't exceed max_horizon
            if total_seq_len > self.config.max_horizon:
                raise ValueError(
                    f"Total sequence length {total_seq_len} (states: {state_seq_len} + actions: {seq_len}) "
                    f"exceeds configured max_horizon={self.config.max_horizon} in self_attn mode"
                )

            pos = self.action_pos_embed[:, :total_seq_len, :]
            x = x + pos

            # Don't add states to cross-attention context
            contexts = self._prepare_context(encoder_hidden_states, None)

            # For self_attn mode, don't include encoded_states in cross-attention mask
            cross_mask = self._build_cross_attention_mask(
                encoder_attention_mask,
                None,
                bsz,
                x.dtype,
            )
        else:
            # For other modes, add positional embeddings normally and add states to cross-attention context
            pos = self.action_pos_embed[:, :seq_len, :]
            x = x + pos
            contexts = self._prepare_context(encoder_hidden_states, encoded_states)

            # For other modes, include encoded_states in cross-attention mask
            cross_mask = self._build_cross_attention_mask(
                encoder_attention_mask,
                encoded_states,
                bsz,
                x.dtype,
            )

        # Process the embedded actions through each transformer block
        # Each block uses the corresponding context tensor for cross-attention
        for block, context in zip(self.blocks, contexts, strict=False):
            x = block(x, timestep_embed, context, attn_mask=cross_mask)

        # Apply final projection to get the denoised action prediction
        output = self.final_layer(x, timestep_embed)

        # For self_attn mode, only return the action predictions (skip state tokens)
        if states_mode == "self_attn":
            state_seq_len = encoded_states.shape[1]
            output = output[:, state_seq_len:, :]

        return output
