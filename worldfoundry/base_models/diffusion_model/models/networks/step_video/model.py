# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
"""StepVideo DiT: per-frame 2D patch embed + AdaLN-single transformer.

Architecture
------------
Input latents are ``B F C H W``.  :meth:`patchfy` rearranges to
``(B·F) C H W``, Conv2d-patches to ``(B·F) (H'W') C``, then the stack
sees ``B (F·H'·W') C`` with 3D RoPE over ``(F, H', W')``.  Text is
PixArt-projected (optional CLIP concat).  The head AdaLN-modulates
per-frame tokens and unpatchifies to ``B F C H W``.
"""

from types import SimpleNamespace
from typing import Dict, Optional
import torch
from torch import nn
from einops import rearrange, repeat
from .blocks import PatchEmbed, StepVideoTransformerBlock
from .normalization import AdaLayerNormSingle, PixArtAlphaTextProjection



class StepVideoModel(nn.Module):
    """Checkpoint-compatible StepVideo transformer (AdaLN-single, 3D RoPE)."""

    _no_split_modules = ["StepVideoTransformerBlock", "PatchEmbed"]

    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        dropout: float = 0.0,
        patch_size: int = 1,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[int]|list|tuple = [6144, 1024],
        attention_type: Optional[str] = "torch",
    ):
        super().__init__()

        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            caption_channels=caption_channels,
        )

        # Set some common variables used across the board.
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels

        self.use_additional_conditions = use_additional_conditions

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                StepVideoTransformerBlock(
                    dim=self.inner_dim,
                    attention_head_dim=self.config.attention_head_dim,
                    attention_type=attention_type
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.patch_size = patch_size

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )

        if isinstance(self.config.caption_channels, int):
            caption_channel = self.config.caption_channels
        else:
            caption_channel, clip_channel = self.config.caption_channels
            self.clip_projection = nn.Linear(clip_channel, self.inner_dim) 

        self.caption_norm = nn.LayerNorm(caption_channel,  eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channel, hidden_size=self.inner_dim
        )
        
        self.parallel = attention_type=='parallel'

    def patchfy(self, hidden_states):
        """Fold ``B F C H W`` into per-frame 2D patches ``(B·F) (H'W') C``."""
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        """Build a boolean ``[B, Q, max_kv]`` mask and crop padded text tokens."""
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.zeros([len(kv_seqlens), q_seqlen, max(kv_seqlens)], dtype=torch.bool, device=encoder_attention_mask.device)
        encoder_hidden_states = encoder_hidden_states[:,: max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask
        
        
    def block_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        rope_positions=None,
        attn_mask=None,
        parallel=True
    ):
        """Run the AdaLN transformer stack on ``B (F·S) C`` video tokens."""

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                attn_mask=attn_mask,
                rope_positions=rope_positions
            )

        return hidden_states
        

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor=None,
        return_dict: bool = True,
    ):
        """Denoise ``B F C H W`` latents; return the same layout (or ``{'x': ...}``)."""
        assert hidden_states.ndim==5; "hidden_states's shape should be (bsz, f, ch, h ,w)"

        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
                
        hidden_states = self.patchfy(hidden_states) 
        len_frame = hidden_states.shape[1]
                
        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor([(height, width)]*bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "nframe": torch.tensor([frame]*bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "fps": fps
            }    
        else:
            added_cond_kwargs = {}
        
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs
        )

        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))
        
        if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
            clip_embedding = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

        hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()
        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask, encoder_hidden_states, q_seqlen=frame*len_frame)
        
        hidden_states = self.block_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            rope_positions=[frame, height, width],
            attn_mask=attn_mask,
            parallel=self.parallel
        )
        
        hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)
        
        embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()
        
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        
        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        
        hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)

        if return_dict:
            return {'x': output}
        return output
    
    
