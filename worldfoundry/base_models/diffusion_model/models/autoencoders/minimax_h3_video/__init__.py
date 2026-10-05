# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 video VAE (causal CNN encoder + ViT decoder).

Package entry for the H3 video codec, ported from SGLang
(Apache-2.0).  :class:`~.model.MiniMaxH3VideoVAE` wraps
:class:`~.klvae.AutoencoderKLLegacy`.

Pixels are BCTHW RGB.  Encoder is a causal 3-D CNN
(:mod:`.vae_cnn`); decoder is ViT3D (:mod:`.vae_vit`).  Latents
are 24-channel with published stats validated by
:func:`~..minimax_h3_common.validate_minimax_h3_vae_latent_stats`.

Sibling of :mod:`..minimax_h3_audio`.
"""

from ..minimax_h3_common import (
    MiniMaxH3VAEContractError,
    validate_minimax_h3_vae_latent_stats,
)
from .config import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
from .klvae import AutoencoderKLLegacy
from .model import MiniMaxH3VideoVAE

__all__ = [
    "AutoencoderKLLegacy",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3VideoVAEArchConfig",
    "MiniMaxH3VideoVAEConfig",
    "MiniMaxH3VAEContractError",
    "validate_minimax_h3_vae_latent_stats",
]
