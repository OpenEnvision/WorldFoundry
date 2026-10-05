"""UniDepth feature contract over the shared DINOv2 implementation."""

import torch
from torch import nn

from worldfoundry.base_models.perception_core.general_perception.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)


class UniDepthEncoder(DinoVisionTransformer):
    """Return every block's spatial features and class token without changing weight keys."""

    def __init__(self, config, *, embed_dim, depth, num_heads, output_idx):
        super().__init__(
            img_size=518,
            patch_size=14,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            init_values=1.0,
            block_chunks=0,
            drop_path_uniform=True,
            num_register_tokens=config.get("num_register_tokens", 0),
            interpolate_offset=config.get("interpolate_offset", 0.0),
            interpolate_antialias=config.get("interpolate_antialias", False),
        )
        # UniDepth checkpoints retain one unused register even with no register tokens.
        if self.register_tokens is None:
            self.register_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.depths = config.get("output_idx", output_idx)
        self.embed_dims = [embed_dim] * depth
        self.use_norm = config.get("use_norm", False)

    def prepare_tokens_with_masks(self, x, masks=None):
        tokens = super().prepare_tokens_with_masks(x, masks)
        if self.num_register_tokens == 0:
            # The shared backbone inserts any non-None register parameter;
            # UniDepth retains a checkpoint parameter but does not insert it.
            tokens = torch.cat((tokens[:, :1], tokens[:, 2:]), dim=1)
        return tokens

    def forward(self, x):
        layers = self.get_intermediate_layers(
            x,
            n=self.n_blocks,
            reshape=True,
            return_class_token=True,
            norm=self.use_norm,
        )
        return ([feature.permute(0, 2, 3, 1) for feature, _ in layers], [token[:, None] for _, token in layers])


def dinov2_vits14(config):
    return UniDepthEncoder(config, embed_dim=384, depth=12, num_heads=6, output_idx=[3, 6, 9, 12])


def dinov2_vitb14(config):
    return UniDepthEncoder(config, embed_dim=768, depth=12, num_heads=12, output_idx=[3, 6, 9, 12])


def dinov2_vitl14(config):
    return UniDepthEncoder(config, embed_dim=1024, depth=24, num_heads=16, output_idx=[5, 12, 18, 24])
