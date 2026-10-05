# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 audio VAE (DAC + BigVGAN), ported from SGLang (Apache-2.0).

Package entry for the H3 speech codec.  :class:`~.minimax_h3.MiniMaxH3AudioVAE`
is the public encode/decode wrapper around :class:`~.audio_vae.DacAudioVAE`
plus the BigVGAN vocoder.

Waveforms are 32 kHz mono.  Latents are 32-channel / 2048-dim
DAC features with published mean/std.  Architecture dataclasses
live in :mod:`.config`.

Sibling of :mod:`..minimax_h3_video`.  Shared contract helpers
are in :mod:`..minimax_h3_common`.
"""

from .audio_vae import DacAudioVAE
from .config import MiniMaxH3AudioVAEArchConfig, MiniMaxH3AudioVAEConfig
from .minimax_h3 import MiniMaxH3AudioVAE

__all__ = [
    "DacAudioVAE",
    "MiniMaxH3AudioVAE",
    "MiniMaxH3AudioVAEArchConfig",
    "MiniMaxH3AudioVAEConfig",
]
