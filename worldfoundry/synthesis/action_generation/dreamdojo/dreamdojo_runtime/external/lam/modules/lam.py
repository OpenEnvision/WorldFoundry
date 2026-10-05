from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from external.lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
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
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

    def encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        assert T == 2
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, 1, E)

        # VAE
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        # Reparameterization
        z_rep = z_mu.reshape(B, T - 1, 1, self.latent_dim)
        return {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VAE
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        return outputs
