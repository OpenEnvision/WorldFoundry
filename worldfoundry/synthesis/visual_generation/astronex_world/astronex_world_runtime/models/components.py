# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Astronex-Robotics/Astronex-World (Apache-2.0) at 27584bf1, models/wan22_components.py.
"""The frozen components Astronex samples with: UMT5-xxl and the Wan2.2 VAE.

Both load from the released weights directory (``text_encoder/`` and ``tokenizer/`` in
Transformers layout, ``vae/`` in Diffusers layout) onto the shared native Wan modules.
Interfaces: ``AstronexTextEncoder(text_prompts) -> {"prompt_embeds"}``,
``encode_to_latent(pixel[B, C, F, H, W]) -> [B, F, z, h, w]`` and
``decode_to_pixel(latent[B, F, z, h, w]) -> [B, F, C, H, W]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    convert_diffusers_wan22_vae_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import VideoVAE38_
from worldfoundry.base_models.diffusion_model.models.encoders.wan.component import (
    convert_diffusers_umt5_encoder_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer, WanTextEncoder
from worldfoundry.core.model_loading import load_model, load_state_dict


class AstronexTextEncoder(nn.Module):
    """UMT5-xxl prompt encoder; padding positions of every embedding are exactly zero."""

    def __init__(self, root: str | Path, text_len: int = 512):
        super().__init__()
        root = Path(root)
        config = json.loads((root / "text_encoder" / "config.json").read_text())
        self.tokenizer = HuggingfaceTokenizer(name=str(root / "tokenizer"), seq_len=text_len)
        self.text_encoder = load_model(
            WanTextEncoder,
            str(root / "text_encoder"),
            config={
                "vocab": config["vocab_size"],
                "dim": config["d_model"],
                "dim_attn": config["num_heads"] * config["d_kv"],
                "dim_ffn": config["d_ff"],
                "num_heads": config["num_heads"],
                "num_layers": config["num_layers"],
                "num_buckets": config["relative_attention_num_buckets"],
                "shared_pos": False,
            },
            torch_dtype=torch.bfloat16,
            state_dict_converter=convert_diffusers_umt5_encoder_state_dict,
        ).requires_grad_(False)
        # Token ids go here; it differs from the parameter device when the encoder swaps in per module.
        self.execution_device = torch.device("cpu")

    def forward(self, text_prompts: list[str]) -> dict[str, torch.Tensor]:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.execution_device)
        mask = mask.to(self.execution_device)
        context = self.text_encoder(ids, mask).clone()
        for embedding, length in zip(context, mask.gt(0).sum(dim=1).long(), strict=False):
            embedding[length:] = 0.0
        return {"prompt_embeds": context}


class AstronexVAE(nn.Module):
    """Wan2.2 48-channel VAE; latents are normalised by the checkpoint's ``latents_mean``/``latents_std``."""

    def __init__(self, root: str | Path):
        super().__init__()
        vae_dir = Path(root) / "vae"
        config = json.loads((vae_dir / "config.json").read_text())
        self.model = VideoVAE38_(
            dim=config["base_dim"],
            z_dim=config["z_dim"],
            dec_dim=config["decoder_base_dim"],
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            attn_scales=config["attn_scales"],
            temperal_downsample=config["temperal_downsample"],
            dropout=config["dropout"],
        )
        state = convert_diffusers_wan22_vae_state_dict(
            load_state_dict(str(vae_dir / "diffusion_pytorch_model.safetensors"))
        )
        self.model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
        self.model.eval().requires_grad_(False)
        z_dim = int(config["z_dim"])
        # (1, C, 1, 1, 1) so they broadcast over [B, C, F, H, W].
        self.register_buffer(
            "latents_mean",
            torch.tensor(config["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(config["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        # The native codec's own (mean, 1/std) scaling is held at the identity; normalisation happens here.
        self._identity_scale = [torch.zeros(z_dim), torch.ones(z_dim)]

    @property
    def dtype(self) -> torch.dtype:
        return self.model.conv1.weight.dtype

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """``[B, C, F, H, W]`` pixels in ``[-1, 1]`` -> ``[B, F, z, H/16, W/16]`` latents."""
        z = self.model.encode(pixel.to(self.dtype), self._identity_scale).float()
        z = (z - self.latents_mean.to(z.device)) / self.latents_std.to(z.device)
        return z.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, F, z, h, w]`` latents -> ``[B, F, C, H, W]`` pixels in ``[-1, 1]``."""
        z = latent.permute(0, 2, 1, 3, 4).float()
        z = z * self.latents_std.to(z.device) + self.latents_mean.to(z.device)
        out = self.model.decode(z.to(self.dtype), self._identity_scale).float().clamp_(-1, 1)
        return out.permute(0, 2, 1, 3, 4)
