"""BiWM camera-text conditioning on the canonical Wan transformer.

The public upstream stage-1/stage-2 inference uses uncompressed history and
local window RoPE. No backbone, attention kernel, or VAE is duplicated here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...model import CrossAttentionProcessor, WanModel


class FrameTextProcessor(CrossAttentionProcessor):
    """Reuse Wan cross attention independently for each latent frame."""

    def __call__(self, attention, x, context, *, camera_frames=None, **kwargs):
        if camera_frames is None:
            return super().__call__(attention, x, context, **kwargs)
        batch, tokens, dim = x.shape
        x = x.reshape(batch * camera_frames, tokens // camera_frames, dim)
        return super().__call__(attention, x, context, **kwargs).reshape(batch, tokens, dim)


class BiWMWanModel(WanModel):
    """Stage-1 teacher and stage-2 no-HE generator, with native Wan keys."""

    def __init__(self, *, text_len=512, cross_attn_norm=True,
                 action_embedder=False, **kwargs):
        super().__init__(has_image_input=False, require_vae_embedding=False,
                         require_clip_embedding=False, per_token_timestep=True, **kwargs)
        if self.patch_size[0] != 1 or self.in_dim != self.out_dim:
            raise ValueError("BiWM requires temporal patch size 1 and equal input/output channels")
        self.text_len = text_len
        self.register_buffer("camera_text", None, persistent=False)
        self.set_attn_processor(FrameTextProcessor())
        if not cross_attn_norm:
            for block in self.blocks:
                block.norm3 = nn.Identity()
        if action_embedder:
            # Retain every checkpoint key. Upstream Wan does not call this MLP.
            self.hycam_action_embedder = nn.Module()
            self.hycam_action_embedder.mlp = nn.Sequential(
                nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
        self.set_attention_compatibility_mode(True)
        self.set_rms_norm_precision("fp32")

    def set_camera_text(self, embeddings):
        if len(embeddings) != 81 or any(e.ndim != 2 for e in embeddings):
            raise ValueError("BiWM requires 81 variable-length camera text embeddings")
        self.camera_text = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    def prepare_condition_context(self, context, *, action_labels=None, **kwargs):
        if context is None or context.ndim != 3:
            raise ValueError("caption context must have shape [B,S,D]")
        if action_labels is not None:
            if self.camera_text is None:
                raise ValueError("Camera text embeddings have not been encoded")
            camera = self.camera_text[action_labels]
            batch, frames = action_labels.shape
            caption = context[:, None].expand(batch, frames, *context.shape[1:])
            context = torch.cat([camera, caption], dim=2).flatten(0, 1)
        context = context[:, :self.text_len]
        context = F.pad(context, (0, 0, 0, self.text_len - context.shape[1]))
        return self.text_embedding(context)

    def block_forward_kwargs(self, grid_size, *, action_labels=None, **kwargs):
        return {"camera_frames": grid_size[0] if action_labels is not None else None}

    def forward(self, x, timestep, context, *, action_labels=None, cond_latent_frames=0):
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError("The public BiWM inference contract supports one video at a time")
        batch, _, frames, height, width = x.shape
        if height % self.patch_size[1] or width % self.patch_size[2]:
            raise ValueError("Latent spatial dimensions must be divisible by Wan patch size")
        if not 0 <= cond_latent_frames < frames:
            raise ValueError("History must leave at least one target frame")
        if action_labels is not None:
            if (action_labels.shape != (batch, frames) or action_labels.dtype != torch.long
                    or torch.any((action_labels < 0) | (action_labels > 80))):
                raise ValueError("action_labels must be int64 [1,T] with values in [0,80]")
        tokens_per_frame = (height // self.patch_size[1]) * (width // self.patch_size[2])
        time = torch.as_tensor(timestep, device=x.device, dtype=torch.float32)
        if time.numel() != 1:
            raise ValueError("BiWM requires a scalar diffusion timestep")
        time = time.reshape(1, 1).expand(batch, frames * tokens_per_frame).clone()
        time[:, :cond_latent_frames * tokens_per_frame] = 0
        return super().forward(x, time, context, action_labels=action_labels).float()
