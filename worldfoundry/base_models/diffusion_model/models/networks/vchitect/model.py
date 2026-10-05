"""Pure-PyTorch Vchitect-2 diffusion transformer.

Architecture
------------
Latents enter as ``B F C H W``.  :class:`PatchEmbed` is per-frame Conv2d
plus a centered crop of a 2D absolute position table → ``(B·F) S C``.
Timestep and pooled text are added into AdaLN embeddings.  Each
:class:`JointTransformerBlock` attends spatially (video+text), temporally
(with complex RoPE over ``F``), and cross to text.  Output unpatchifies
to ``B F C H W``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from einops import rearrange
from torch import nn

from .blocks import AdaLayerNormContinuous, JointTransformerBlock


class PatchEmbed(nn.Module):
    """2D Conv patch embed + cropped absolute pos table: ``B C H W`` → ``B (H'W') C``."""

    def __init__(
        self,
        *,
        sample_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        pos_embed_max_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.pos_embed_max_size = int(pos_embed_max_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.empty(1, pos_embed_max_size**2, embed_dim))

    def _position(self, height: int, width: int) -> torch.Tensor:
        """Center-crop the square pos table to the current ``H'×W'`` patch grid."""
        if height > self.pos_embed_max_size or width > self.pos_embed_max_size:
            raise ValueError("Vchitect latent patch grid exceeds the checkpoint position table")
        table = self.pos_embed.reshape(
            1,
            self.pos_embed_max_size,
            self.pos_embed_max_size,
            self.pos_embed.shape[-1],
        )
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        return table[:, top : top + height, left : left + width].reshape(1, height * width, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify a single-frame (or folded-frame) ``B C H W`` latent."""
        patches = self.proj(images)
        height, width = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2)
        return patches + self._position(height, width).to(patches)


class TimestepEmbedding(nn.Module):
    """Two-layer SiLU MLP used for both timestep Fourier features and pooled text."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``[..., in_dim]`` to ``[..., out_dim]``."""
        return self.linear_2(self.act(self.linear_1(values)))


def timestep_projection(timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """Sinusoidal Fourier features of shape ``[B, dim]`` from scalar timesteps."""
    half = dim // 2
    exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device)) * torch.arange(
        half,
        device=timesteps.device,
        dtype=torch.float32,
    ) / half
    values = timesteps.float().reshape(-1, 1) * exponent.exp().reshape(1, -1)
    return torch.cat((values.cos(), values.sin()), dim=-1)


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Sum of projected timestep Fourier features and pooled caption embedding."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = TimestepEmbedding(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        """Return the AdaLN embedding shared by every joint block."""
        projected = timestep_projection(timestep, 256).to(dtype=pooled_projection.dtype)
        return self.timestep_embedder(projected) + self.text_embedder(pooled_projection)


class VchitectXLTransformerModel(nn.Module):
    """Inference-only checkpoint layout for Vchitect-2.0."""

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 192,
        rope_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.inner_dim = int(num_attention_heads * attention_head_dim)
        if caption_projection_dim != self.inner_dim:
            raise ValueError("caption_projection_dim must equal the Vchitect transformer width")
        self.pos_embed = PatchEmbed(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(self.inner_dim, pooled_projection_dim)
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)
        self.transformer_blocks = nn.ModuleList(
            JointTransformerBlock(
                self.inner_dim,
                num_attention_heads,
                context_pre_only=index == num_layers - 1,
            )
            for index in range(num_layers)
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.rope_scaling_factor = float(rope_scaling_factor)

    def _frequencies(self, frames: int, device: torch.device) -> torch.Tensor:
        """Complex 1D RoPE frequencies over the temporal axis (used inside attention)."""
        dim = self.config.attention_head_dim
        frequency = 1.0 / (
            1e6 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        positions = torch.arange(frames, device=device, dtype=torch.float32) / self.rope_scaling_factor
        angles = torch.outer(positions, frequency)
        return torch.polar(torch.ones_like(angles), angles)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise ``B F C H W`` latents; return residual in the same layout."""
        if hidden_states.ndim != 5:
            raise ValueError("Vchitect expects latents in [B,F,C,H,W] layout")
        batch, frames, _, height, width = hidden_states.shape
        patches = self.pos_embed(rearrange(hidden_states, "b f c h w -> (b f) c h w"))
        embedding = self.time_text_embed(timestep.reshape(-1), pooled_projections)
        if embedding.shape[0] == 1 and batch > 1:
            embedding = embedding.expand(batch, -1)
        if embedding.shape[0] != batch:
            raise ValueError("Vchitect timestep/pooled batch must match latent batch")
        embedding_frames = embedding.repeat_interleave(frames, dim=0)
        context = self.context_embedder(encoder_hidden_states)
        if context.shape[0] == 1 and batch > 1:
            context = context.expand(batch, -1, -1)
        context = context.repeat_interleave(frames, dim=0)
        frequencies = self._frequencies(frames, patches.device)
        for block in self.transformer_blocks:
            context, patches = block(
                patches,
                context,
                embedding_frames,
                frequencies=frequencies,
                frames=frames,
            )
        patches = self.proj_out(self.norm_out(patches, embedding_frames))
        patch = self.patch_size
        latent_height, latent_width = height // patch, width // patch
        patches = patches.reshape(-1, latent_height, latent_width, patch, patch, self.out_channels)
        output = torch.einsum("nhwpqc->nchpwq", patches).reshape(
            batch * frames,
            self.out_channels,
            height,
            width,
        )
        return rearrange(output, "(b f) c h w -> b f c h w", b=batch, f=frames)


__all__ = [
    "CombinedTimestepTextProjEmbeddings",
    "PatchEmbed",
    "TimestepEmbedding",
    "VchitectXLTransformerModel",
    "timestep_projection",
]
