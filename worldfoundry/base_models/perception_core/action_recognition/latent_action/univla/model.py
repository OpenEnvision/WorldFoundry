"""UniVLA controllable DINO encoder used by LARYBench."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torchvision.transforms import Normalize

from ..backbones import get_dinov2_vitb14_reg_tokenizer
from .blocks import SpatioTemporalTransformer, VectorQuantizer

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ControllableDINOLatentActionModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        del in_dim, patch_size, dec_blocks

        self.latent_dim = latent_dim
        self.num_codes = 4
        self.dino_transform = Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        self.dino_encoder = get_dinov2_vitb14_reg_tokenizer()
        dino_dim = 768

        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq_action = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
        )

    def vq_encode(self, videos: Tensor, **_: Any) -> dict[str, Tensor]:
        if videos.ndim != 5 or videos.shape[1] != 2:
            raise ValueError(f"UniVLA expects [B, 2, C, H, W] frame pairs, got {videos.shape}")

        batch_size = videos.shape[0]
        images = rearrange(videos, "b t c h w -> (b t) c h w")
        images = self.dino_transform(images)
        features = self.dino_encoder.forward_features(images)["x_norm_patchtokens"]
        features = rearrange(features, "(b t) l d -> b t l d", t=2)

        uncontrolled_prompt = self.action_latent.expand(batch_size, 2, -1, -1)
        controllable_prompt = self.action_latent_controllable.expand(batch_size, 2, -1, -1)
        encoded = self.encoder(
            torch.cat(
                (controllable_prompt, uncontrolled_prompt, features),
                dim=2,
            )
        )

        actions = self.to_codebook(encoded[:, 1:, : self.num_codes]).reshape(
            batch_size, self.num_codes, self.latent_dim
        )
        quantized, indices = self.vq_action(actions)
        return {
            "z_q": quantized.reshape(batch_size, 1, self.num_codes, self.latent_dim),
            "indices": indices,
        }
