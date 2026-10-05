# SPDX-License-Identifier: Apache-2.0
"""Top-level MiniMax H3 audio VAE wrapper.

:class:`MiniMaxH3AudioVAE` subclasses :class:`~.audio_vae.DacAudioVAE`
with the published H3 architecture (32 kHz, 32 latent channels,
BigVGAN decoder).  ``encode`` / ``decode`` plus latent stats
are the public surface.

``layer_names`` documents which submodules SGLang used for
layerwise offload; they are informational only here.

Package entry re-exports this type from :mod:`..minimax_h3_audio`.
"""

from .audio_vae import DacAudioVAE
from .config import MiniMaxH3AudioVAEConfig


class MiniMaxH3AudioVAE(DacAudioVAE):
    # Ported from SGLang. The original also inherited
    # ``LayerwiseOffloadableModuleMixin`` for multi-GPU layerwise offloading;
    # that mixin is SGLang distributed machinery and is dropped for single-GPU
    # inference. The ``layer_names`` below document which submodules ran the
    # forward hooks (BigVGAN keeps each executable upsampler in a one-element
    # ModuleList), retained here only as informational metadata.
    """Encode/decode MiniMax H3 audio with published latent statistics."""
    layer_names = [
        "encoder.block",
        *(f"decoder.ups.{index}" for index in range(7)),
        "decoder.resblocks",
    ]

    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__(
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 5, 5],
            latent_dim=2048,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            sample_rate=32000,
            vae_latent_channels=32,
            attn_proj=True,
            decoder_type="bigvgan",
        )
        self.config = config


__all__ = ["MiniMaxH3AudioVAE"]
