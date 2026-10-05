# SPDX-License-Identifier: Apache-2.0
"""Dataclasses for MiniMax H3 audio VAE / vocoder architecture.

:class:`MiniMaxH3AudioVAEArchConfig` and
:class:`MiniMaxH3AudioVAEConfig` describe encoder rates,
latent width, sample rate (32 kHz), and vocoder choice.

No tensor math lives here.  :class:`~.minimax_h3.MiniMaxH3AudioVAE`
consumes these at construction.

Sibling of :mod:`~..minimax_h3_video.config`.
"""

from dataclasses import dataclass, field

from ..minimax_h3_common import validate_minimax_h3_vae_latent_stats


@dataclass
class MiniMaxH3AudioVAEArchConfig:
    """Architecture config for the MiniMax H3 audio VAE.

    Ported from SGLang; the original inherited ``VAEArchConfig`` (which pulled in
    tiling / distributed / CLI machinery). Only the audio-relevant fields are
    kept for single-GPU inference. ``latents_mean`` / ``latents_std`` come from
    the model's ``config.json`` and are validated by the shared contract.
    """

    sample_rate: int = 32000
    latent_channels: int = 32
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    output_channel: int = 2


@dataclass
class MiniMaxH3AudioVAEConfig:
    """Model config for the MiniMax H3 audio VAE (single-GPU port)."""

    arch_config: MiniMaxH3AudioVAEArchConfig = field(
        default_factory=MiniMaxH3AudioVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True

    def post_init(self) -> None:
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="audio_vae",
            expected_channels=32,
        )


__all__ = ["MiniMaxH3AudioVAEArchConfig", "MiniMaxH3AudioVAEConfig"]
