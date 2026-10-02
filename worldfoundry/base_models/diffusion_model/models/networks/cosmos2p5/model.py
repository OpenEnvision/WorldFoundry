"""Inference-only Cosmos Predict 2.5 diffusion transformer.

Checkpoint-shaped DiT math for Predict 2.5 and Transfer 2.5 (VACE control).
Sampling, loading, text encoding, and tokenization live in
``denoisers/cosmos2p5.py`` + the recipe.

Architecture
------------
Latents enter as ``B C T H W``.  After optional condition-mask / padding-mask
channel concat, :class:`Cosmos25PatchEmbed` folds each ``(pt, ph, pw)`` cube
into one token, yielding ``B (T' H' W') C``.  3D RoPE is factorized over
T/H/W (with optional FPS warping).  Each :class:`Cosmos25TransformerBlock`
is AdaLN-Zero gated self-attn → text cross-attn → FFN.  Transfer 2.5 runs a
sparse control branch and adds hints into the base stream.

This module must stay numerically compatible with the pinned Hub checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange, repeat

from worldfoundry.core.attention import apply_rotary_embedding, scaled_dot_product_attention
from worldfoundry.core.nn.diffusion.timestep import Timesteps


class Cosmos25PatchEmbed(nn.Module):
    """Linear 3D patch embed: ``B C T H W`` → ``B T' H' W' C`` tokens."""

    def __init__(self, in_channels: int, out_channels: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj = nn.Linear(in_channels * volume, out_channels, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reshape each patch cube onto the last dim, then project to ``out_channels``."""
        batch, channels, frames, height, width = value.shape
        pt, ph, pw = self.patch_size
        value = value.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        return self.proj(value.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7))


class Cosmos25TimestepEmbedding(nn.Module):
    """SiLU MLP that expands sinusoidal timestep features to ``3 * hidden`` AdaLN bias."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size * 3, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(value)))


class Cosmos25TimeEmbed(nn.Module):
    """Sinusoidal timestep → ``(projected, temb)`` pair used by every AdaLN block."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = Cosmos25TimestepEmbedding(hidden_size)

    def forward(self, reference: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast the Fourier features to ``reference.dtype`` so AdaLN stays in the compute dtype."""
        projected = self.time_proj(timestep).type_as(reference)
        return projected, self.t_embedder(projected)


class Cosmos25AdaLayerNormZero(nn.Module):
    """AdaLN-Zero: LayerNorm plus LoRA-bottleneck ``(shift, scale, gate)`` from timestep."""

    def __init__(self, hidden_size: int, lora_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()
        self.linear_1 = nn.Linear(hidden_size, lora_dim, bias=False)
        self.linear_2 = nn.Linear(lora_dim, hidden_size * 3, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(norm(x) * (1+scale) + shift, gate)``; ``temb`` is the residual AdaLN bias."""
        modulation = self.linear_2(self.linear_1(self.activation(embedded_timestep))) + temb
        shift, scale, gate = modulation.chunk(3, dim=-1)
        return self.norm(hidden_states) * (1 + scale) + shift, gate


class Cosmos25AdaLayerNorm(nn.Module):
    """Final-layer AdaLN (shift/scale only; no residual gate)."""

    def __init__(self, hidden_size: int, lora_dim: int) -> None:
        super().__init__()
        self.embedding_dim = hidden_size
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(hidden_size, lora_dim, bias=False)
        self.linear_2 = nn.Linear(lora_dim, hidden_size * 2, bias=False)

    def forward(self, value: torch.Tensor, embedded_timestep: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply ``norm(x) * (1+scale) + shift``; reuse the first ``2*C`` of ``temb`` as residual."""
        modulation = self.linear_2(self.linear_1(self.activation(embedded_timestep)))
        modulation = modulation + temb[..., : self.embedding_dim * 2]
        shift, scale = modulation.chunk(2, dim=-1)
        return self.norm(value) * (1 + scale) + shift


class Cosmos25Attention(nn.Module):
    """QK-RMSNorm multi-head attention; self-attn when ``context_dim`` is ``None``."""

    def __init__(self, query_dim: int, context_dim: int | None, heads: int, head_dim: int) -> None:
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.ModuleList((nn.Linear(inner_dim, query_dim, bias=False), nn.Dropout(0.0)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attend ``B S C`` queries to context; RoPE is applied in fp32 then cast back."""
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        key = self.to_k(context).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        value = self.to_v(context).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos[None, None]
            sin = sin[None, None]
            query = apply_rotary_embedding(query.float(), cos, sin).type_as(query)
            key = apply_rotary_embedding(key.float(), cos, sin).type_as(key)
        value = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        value = value.transpose(1, 2).flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](value))


class _GELUProjection(nn.Module):
    """Bias-free Linear + GELU, matching the checkpoint's first FFN slice."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.gelu(self.proj(value))


class Cosmos25FeedForward(nn.Module):
    """GELU MLP with a no-op Dropout slot so state-dict keys stay ``net.0/1/2``."""

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = int(hidden_size * mlp_ratio)
        self.net = nn.ModuleList(
            (
                _GELUProjection(hidden_size, intermediate),
                nn.Dropout(0.0),
                nn.Linear(intermediate, hidden_size, bias=False),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            value = layer(value)
        return value


class Cosmos25TransformerBlock(nn.Module):
    """One Predict 2.5 DiT block: gated self-attn, text cross-attn, then FFN."""

    def __init__(
        self, hidden_size: int, heads: int, head_dim: int, text_dim: int, mlp_ratio: float, lora_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = Cosmos25AdaLayerNormZero(hidden_size, lora_dim)
        self.attn1 = Cosmos25Attention(hidden_size, None, heads, head_dim)
        self.norm2 = Cosmos25AdaLayerNormZero(hidden_size, lora_dim)
        self.attn2 = Cosmos25Attention(hidden_size, text_dim, heads, head_dim)
        self.norm3 = Cosmos25AdaLayerNormZero(hidden_size, lora_dim)
        self.ff = Cosmos25FeedForward(hidden_size, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the three AdaLN-gated residual branches on ``B S C`` tokens."""
        normalized, gate = self.norm1(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn1(normalized, rotary_emb=rotary_emb)
        normalized, gate = self.norm2(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.attn2(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        normalized, gate = self.norm3(hidden_states, embedded_timestep, temb)
        return hidden_states + gate * self.ff(normalized)


class Cosmos25RotaryPosEmbed(nn.Module):
    """Factorized 3D RoPE over the post-patch ``(T', H', W')`` lattice.

    Head dim is split as ``(T remainder, H, W)`` with NTK-aware axis scales.
    Temporal positions can be warped by ``fps / base_fps`` so motion speed
    stays consistent across frame rates.
    """

    def __init__(
        self,
        head_dim: int,
        max_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        rope_scale: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.max_size = tuple(size // patch for size, patch in zip(max_size, patch_size, strict=True))
        self.patch_size = patch_size
        self.base_fps = 24.0
        dim_h = head_dim // 6 * 2
        dim_w = head_dim // 6 * 2
        self.axis_dims = (head_dim - dim_h - dim_w, dim_h, dim_w)
        self.ntk_factors = tuple(
            scale ** (dim / (dim - 2)) for scale, dim in zip(rope_scale, self.axis_dims, strict=True)
        )

    def forward(self, value: torch.Tensor, fps: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[T'*H'*W', head_dim]`` from a ``B C T H W`` dummy."""
        _, _, frames, height, width = value.shape
        sizes = (frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2])
        sequence = torch.arange(max(self.max_size), device=value.device, dtype=torch.float32)
        embeddings = []
        for axis, (size, dim, factor) in enumerate(zip(sizes, self.axis_dims, self.ntk_factors, strict=True)):
            positions = sequence[:size]
            if axis == 0 and fps is not None:
                positions = positions / float(fps) * self.base_fps
            dims = torch.arange(0, dim, 2, device=value.device, dtype=torch.float32) / dim
            embeddings.append(torch.outer(positions, 1.0 / ((10000.0 * factor) ** dims)))
        temporal, vertical, horizontal = embeddings
        temporal = temporal[:, None, None].expand(-1, sizes[1], sizes[2], -1)
        vertical = vertical[None, :, None].expand(sizes[0], -1, sizes[2], -1)
        horizontal = horizontal[None, None, :].expand(sizes[0], sizes[1], -1, -1)
        frequencies = torch.cat((temporal, vertical, horizontal) * 2, dim=-1).flatten(0, 2)
        return frequencies.cos(), frequencies.sin()


class Cosmos25Transformer3DModel(nn.Module):
    """Checkpoint-compatible base Predict 2.5 DiT without Diffusers mixins."""

    def __init__(
        self,
        in_channels: int = 17,
        out_channels: int = 16,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_in_channels: int = 100352,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0),
        rope_enable_fps_modulation: bool = False,
        use_crossattn_projection: bool = True,
        concat_padding_mask: bool = True,
        **unused_config: object,
    ) -> None:
        """Build the Predict 2.5 stem; extra checkpoint keys are accepted and ignored."""
        super().__init__()
        del unused_config
        hidden_size = num_attention_heads * attention_head_dim
        self.config = SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            concat_padding_mask=concat_padding_mask,
        )
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        patch_channels = in_channels + int(concat_padding_mask)
        self.patch_embed = Cosmos25PatchEmbed(patch_channels, hidden_size, patch_size)
        self.rope = Cosmos25RotaryPosEmbed(attention_head_dim, max_size, patch_size, rope_scale)
        self.text_embed = (
            nn.Sequential(nn.Linear(text_in_channels, text_embed_dim), nn.GELU())
            if use_crossattn_projection
            else nn.Identity()
        )
        self.time_embed = Cosmos25TimeEmbed(hidden_size)
        self.time_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            Cosmos25TransformerBlock(
                hidden_size, num_attention_heads, attention_head_dim, text_embed_dim, mlp_ratio, adaln_lora_dim
            )
            for _ in range(num_layers)
        )
        self.norm_out = Cosmos25AdaLayerNorm(hidden_size, adaln_lora_dim)
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(hidden_size, patch_volume * out_channels, bias=False)

    def _prepare_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        embedder: Cosmos25PatchEmbed,
        fps: float | None,
        condition_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int, int]]:
        """Optional channel concat + patch embed → ``B (T'H'W') C`` tokens and RoPE."""
        batch, _, frames, height, width = hidden_states.shape
        if condition_mask is not None:
            hidden_states = torch.cat((hidden_states, condition_mask.type_as(hidden_states)), dim=1)
        if self.config.concat_padding_mask:
            if padding_mask is None:
                padding_mask = hidden_states.new_zeros((batch, 1, height, width))
            padding_mask = functional.interpolate(padding_mask, (height, width), mode="nearest")
            padding_mask = padding_mask.unsqueeze(2).expand(-1, -1, frames, -1, -1)
            hidden_states = torch.cat((hidden_states, padding_mask.type_as(hidden_states)), dim=1)
        pt, ph, pw = self.config.patch_size
        grid = (frames // pt, height // ph, width // pw)
        return embedder(hidden_states).flatten(1, 3), self.rope(hidden_states, fps), grid

    def _prepare_conditioning(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project text and broadcast timestep AdaLN to every spatial token.

        ``timestep`` may be scalar per sample or one value per latent frame;
        both are expanded to ``B (T' H' W') C`` to match the token stream.
        """
        batch = hidden_states.shape[0]
        token_frames, token_height, token_width = grid
        encoder_hidden_states = self.text_embed(encoder_hidden_states)
        timestep = timestep.reshape(batch, -1)
        if timestep.shape[1] not in {1, token_frames}:
            raise ValueError("Cosmos2.5 timestep must be scalar per sample or one value per latent frame")
        projected, temb = self.time_embed(hidden_states, timestep.flatten())
        projected = self.time_norm(projected.reshape(batch, -1, projected.shape[-1]))
        temb = temb.reshape(batch, -1, temb.shape[-1])
        temb, projected = (
            repeat(
                value,
                "b t c -> b (t r h w) c",
                r=token_frames if value.shape[1] == 1 else 1,
                h=token_height,
                w=token_width,
            )
            for value in (temb, projected)
        )
        return encoder_hidden_states, projected, temb

    def _unpatchify(self, hidden_states: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """Fold ``B (T'H'W') (pt·ph·pw·C)`` tokens back to ``B C T H W`` video latents."""
        token_frames, token_height, token_width = grid
        pt, ph, pw = self.config.patch_size
        return rearrange(
            hidden_states,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=token_frames,
            h=token_height,
            w=token_width,
            pt=pt,
            ph=ph,
            pw=pw,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        fps: float | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        control_hidden_states: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Denoise ``B C T H W`` latents; optional per-block ``control_hidden_states`` are added in."""
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None].to(dtype=torch.bool)
        hidden_states, rotary_emb, grid = self._prepare_tokens(
            hidden_states,
            embedder=self.patch_embed,
            fps=fps if self.rope_enable_fps_modulation else None,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
        )
        encoder_hidden_states, projected, temb = self._prepare_conditioning(
            hidden_states,
            timestep,
            encoder_hidden_states,
            grid,
        )
        for index, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, projected, temb, rotary_emb, attention_mask)
            if control_hidden_states is not None and str(index) in control_hidden_states:
                hidden_states = hidden_states + control_hidden_states[str(index)]
        hidden_states = self.proj_out(self.norm_out(hidden_states, projected, temb))
        return self._unpatchify(hidden_states, grid)


class Cosmos25ControlTransformerBlock(Cosmos25TransformerBlock):
    """VACE control branch block with checkpoint-compatible zero projections."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        head_dim: int,
        text_dim: int,
        mlp_ratio: float,
        lora_dim: int,
        *,
        block_index: int,
    ) -> None:
        super().__init__(hidden_size, heads, head_dim, text_dim, mlp_ratio, lora_dim)
        if block_index == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
        self.after_proj = nn.Linear(hidden_size, hidden_size)

    def forward_control(
        self,
        hidden_states: torch.Tensor,
        base_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the control branch; layer 0 mixes in base tokens via ``before_proj``."""
        before_proj = getattr(self, "before_proj", None)
        if before_proj is not None:
            hidden_states = before_proj(hidden_states) + base_hidden_states
        hidden_states = super().forward(
            hidden_states,
            encoder_hidden_states,
            embedded_timestep,
            temb,
            rotary_emb,
            attention_mask,
        )
        return hidden_states, self.after_proj(hidden_states)


class Cosmos25Transfer3DModel(Cosmos25Transformer3DModel):
    """Cosmos Transfer 2.5 VACE DiT using the native Predict backbone."""

    def __init__(
        self,
        in_channels: int = 17,
        out_channels: int = 16,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_in_channels: int = 100352,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] = (128, 240, 240),
        patch_size: tuple[int, int, int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0),
        concat_padding_mask: bool = True,
        num_max_modalities: int = 8,
        control_block_every_n: int = 7,
        rope_enable_fps_modulation: bool = False,
        **unused_config: object,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_in_channels=text_in_channels,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
            rope_enable_fps_modulation=rope_enable_fps_modulation,
            concat_padding_mask=concat_padding_mask,
            **unused_config,
        )
        hidden_size = num_attention_heads * attention_head_dim
        latent_channels = in_channels - 1
        self.num_control_channels = latent_channels * num_max_modalities
        control_channels = self.num_control_channels + 1 + int(concat_padding_mask)
        self.control_embedder = Cosmos25PatchEmbed(control_channels, hidden_size, patch_size)
        self.control_layers = tuple(range(0, num_layers, control_block_every_n))
        self.control_blocks = nn.ModuleList(
            Cosmos25ControlTransformerBlock(
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                text_embed_dim,
                mlp_ratio,
                adaln_lora_dim,
                block_index=index,
            )
            for index in range(len(self.control_layers))
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        latent_control_input: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        fps: float | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        control_context_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Denoise with a sparse VACE control branch; pad unused control channels to the max modality count."""
        if condition_mask is None:
            raise ValueError("Cosmos Transfer2.5 requires a condition mask")
        if latent_control_input.shape[1] > self.num_control_channels:
            raise ValueError(
                f"Cosmos Transfer2.5 accepts at most {self.num_control_channels} control latent channels, "
                f"got {latent_control_input.shape[1]}"
            )
        if latent_control_input.shape[1] < self.num_control_channels:
            padding = latent_control_input.new_zeros(
                latent_control_input.shape[0],
                self.num_control_channels - latent_control_input.shape[1],
                *latent_control_input.shape[2:],
            )
            latent_control_input = torch.cat((latent_control_input, padding), dim=1)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None].to(dtype=torch.bool)

        hidden_states, rotary_emb, grid = self._prepare_tokens(
            hidden_states,
            embedder=self.patch_embed,
            fps=fps if self.rope_enable_fps_modulation else None,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
        )
        control_states, _, control_grid = self._prepare_tokens(
            latent_control_input,
            embedder=self.control_embedder,
            fps=fps if self.rope_enable_fps_modulation else None,
            condition_mask=condition_mask,
            padding_mask=padding_mask,
        )
        if control_grid != grid:
            raise ValueError(f"control latent grid {control_grid} does not match diffusion grid {grid}")
        encoder_hidden_states, projected, temb = self._prepare_conditioning(
            hidden_states,
            timestep,
            encoder_hidden_states,
            grid,
        )

        hints: dict[int, torch.Tensor] = {}
        for layer, block in zip(self.control_layers, self.control_blocks, strict=True):
            control_states, hint = block.forward_control(
                control_states,
                hidden_states,
                encoder_hidden_states,
                projected,
                temb,
                rotary_emb,
                attention_mask,
            )
            hints[layer] = hint

        if isinstance(control_context_scale, torch.Tensor) and control_context_scale.numel() != 1:
            raise ValueError("native Cosmos Transfer2.5 currently expects a scalar control_context_scale")
        for index, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                projected,
                temb,
                rotary_emb,
                attention_mask,
            )
            if index in hints:
                hidden_states = hidden_states + hints[index] * control_context_scale
        hidden_states = self.proj_out(self.norm_out(hidden_states, projected, temb))
        return self._unpatchify(hidden_states, grid)


__all__ = [
    "Cosmos25ControlTransformerBlock",
    "Cosmos25Transfer3DModel",
    "Cosmos25Transformer3DModel",
    "Cosmos25TransformerBlock",
]
