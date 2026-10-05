from os import path
from typing import Union

import torch
from torch import nn

from external.lam.modules import LatentActionModel


class LAM(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        # Latent action autoencoder
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        ckpt_path: Union[None, str] = None
    ) -> None:
        super().__init__()
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout
        )

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.reload_ckpt(ckpt_path)

    def reload_ckpt(self, ckpt_path: str) -> None:
        if path.exists(ckpt_path):
            lam = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(lam, assign=True)
            print(f"Restored LAM from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing LAM keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected LAM keys: {unexpected}")
        else:
            print(f"LAM checkpoint {ckpt_path} does not exist")
