from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from worldfoundry.synthesis.visual_generation.adaworld.lam.modules import LatentActionModel


class LAM(nn.Module):
    """Inference-only wrapper around AdaWorld's latent action model."""

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        ckpt_path: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
        )
        if ckpt_path:
            self.reload_ckpt(ckpt_path)

    def reload_ckpt(self, ckpt_path: str) -> None:
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            return
        payload = torch.load(path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys and not any(key.startswith("lam.") for key in state_dict):
            self.lam.load_state_dict(state_dict, strict=False)

    def encode(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.lam.encode(videos)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.encode(batch["videos"])
