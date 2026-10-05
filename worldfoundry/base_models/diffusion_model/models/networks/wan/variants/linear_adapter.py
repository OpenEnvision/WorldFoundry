"""Sana / Sana-Video aliases on the linear-attention Wan 2.1 graph.

Historical Sana checkpoints use ``SanaVideoMSBlock`` / ``SanaWanModel``
class names.  The math is identical to :mod:`.linear`: flash self-attn
or linear (ReLU / power) attention plus the official Wan packing.

This adapter only rebuilds ``self.blocks`` with the Sana type names so
state-dict keys and ``_no_split_modules`` stay compatible.  No action,
camera, VACE, causal, or TeaCache hooks are added here.
"""

import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.linear import WanAttentionBlock, WanLinearAttentionModel, WanModel


class SanaVideoMSBlock(WanAttentionBlock):
    """Checkpoint alias of :class:`WanAttentionBlock` for Sana-Video weights."""
    pass


class SanaWanModel(WanModel):
    """Wan 2.1 T2V/I2V stem with Sana-named flash-attention blocks."""

    def __init__(self, *args, **kwargs):
        """Rebuild the block stack after the official Wan constructor."""
        super().__init__(*args, **kwargs)
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                SanaVideoMSBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )


class SanaWanLinearAttentionModel(WanLinearAttentionModel):
    """Sana-named wrapper around :class:`WanLinearAttentionModel`."""

    def __init__(self, *args, **kwargs):
        """Rebuild blocks as flash self-attn + MLP FFN with Sana class names."""
        super().__init__(*args, **kwargs)
        cross_attn_type = "t2v_cross_attn" if self.model_type == "t2v" else "i2v_cross_attn"
        self_attn_types = ["flash"] * self.num_layers
        ffn_types = ["mlp"] * self.num_layers

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSBlock(
                    cross_attn_type,
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    self_attn_types[i],
                    self.rope_after,
                    self.power,
                    ffn_types[i],
                )
                for i in range(self.num_layers)
            ]
        )
