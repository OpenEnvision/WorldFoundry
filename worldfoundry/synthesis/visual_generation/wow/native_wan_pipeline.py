"""WoW assembly on the shared native Wan roles and staged runner."""

from __future__ import annotations

from pathlib import Path

import torch

from worldfoundry.base_models.diffusion_model.loaders import (
    WanInferenceComponents,
    load_wan_conditioning_components,
    load_wan_transformer_checkpoint,
)
from worldfoundry.base_models.diffusion_model.runners.wan_staged import WanStagedPipeline


def load_wow_wan_pipeline(
    *,
    model_root: str | Path,
    transformer_checkpoint: str | Path,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> WanStagedPipeline:
    """Bind a WoW checkpoint to canonical Wan text/VAE/DiT roles.

    The checkpoint tensor graph decides whether the variant owns a CLIP image
    branch (released 14B models) or only VAE first-frame conditioning (1.3B).
    """

    root = Path(model_root).expanduser().resolve()
    checkpoint = Path(transformer_checkpoint).expanduser().resolve()
    dit = load_wan_transformer_checkpoint(
        checkpoint,
        torch_dtype=torch_dtype,
        device="cpu",
        state_dict_prefixes=("pipe.dit.", "dit."),
        strict=True,
    )
    conditioning = load_wan_conditioning_components(
        root,
        image_conditioned=bool(dit.has_image_input and dit.require_clip_embedding),
        torch_dtype=torch_dtype,
        device="cpu",
    )
    components = WanInferenceComponents(
        dit=dit,
        text_encoder=conditioning.text_encoder,
        vae=conditioning.vae,
        tokenizer_path=conditioning.tokenizer_path,
        image_encoder=conditioning.image_encoder,
    )
    return WanStagedPipeline.from_components(
        components,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["load_wow_wan_pipeline"]
