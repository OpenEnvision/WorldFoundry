from typing import Dict

import torch
import torch.nn as nn
from worldfoundry.synthesis.visual_generation.adaworld.lam.modules.blocks import (
    SpatioTemporalTransformer,
    patchify,
)
from torch import Tensor


class LatentActionModel(nn.Module):
    """
    Latent action VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        del dec_blocks

    def encode(self, videos: Tensor) -> Dict:
        # AdaWorld supplies two source frames and one target frame. The target
        # is for reconstruction during LAM training; its latent action is
        # encoded from the two source frames, as in the official LAM runtime.
        B, T = videos.shape[:2]
        if T == 2:
            patches = patchify(videos, self.patch_size)
        elif T == 3:
            T = 2
            patches = patchify(videos[:, :2], self.patch_size)
        else:
            raise ValueError(f"LAM expects two source frames or two source frames plus target; got {T}")
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, 1, E)

        # VAE
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu = torch.chunk(moments, 2, dim=1)[0]
        return {"z_rep": z_mu.reshape(B, T - 1, 1, self.latent_dim)}

    def forward(self, batch: Dict) -> Dict:
        return self.encode(batch["videos"])
