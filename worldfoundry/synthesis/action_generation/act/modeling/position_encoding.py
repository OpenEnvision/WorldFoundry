# Inference-only ACT source retained in-tree.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

import math
from typing import Any


def build_position_encoding(args: Any):
    """Create the positional encoding module used by the ACT visual backbone.

    Args:
        args: Namespace-like object with hidden_dim and position_embedding.
    """
    import torch
    from torch import nn

    class PositionEmbeddingSine(nn.Module):
        def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False) -> None:
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = 2 * math.pi

        def forward(self, tensor):
            x = tensor
            not_mask = torch.ones_like(x[0, [0]])
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    class PositionEmbeddingLearned(nn.Module):
        def __init__(self, num_pos_feats: int = 256) -> None:
            super().__init__()
            self.row_embed = nn.Embedding(50, num_pos_feats)
            self.col_embed = nn.Embedding(50, num_pos_feats)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.row_embed.weight)
            nn.init.uniform_(self.col_embed.weight)

        def forward(self, tensor):
            h, w = tensor.shape[-2:]
            i = torch.arange(w, device=tensor.device)
            j = torch.arange(h, device=tensor.device)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            return (
                torch.cat(
                    [
                        x_emb.unsqueeze(0).repeat(h, 1, 1),
                        y_emb.unsqueeze(1).repeat(1, w, 1),
                    ],
                    dim=-1,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .repeat(tensor.shape[0], 1, 1, 1)
            )

    n_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        return PositionEmbeddingSine(n_steps, normalize=True)
    if args.position_embedding in ("v3", "learned"):
        return PositionEmbeddingLearned(n_steps)
    raise ValueError(f"Unsupported ACT position embedding: {args.position_embedding}")
