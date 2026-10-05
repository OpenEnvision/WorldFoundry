# Ported from SandAI MAGI-2-preview (Apache-2.0): single-GPU VAE package.
"""MAGI-2-preview VAEs ported into WorldFoundry as single-GPU PyTorch.

Exposes:
    - Wan2_2_VAE / Magi2VideoVAE   : Wan2.2 causal-3D video encode VAE (z_dim=48).
    - TurboVAED / Magi2TurboDecoder: decode-only student (sliding-window decode).
    - AudioAutoencoder / Magi2AudioVAE: Stable-Audio-Open Oobleck stereo VAE.
"""

from .audio_vae import AudioAutoencoder, create_model_from_config
from .audio_vae import AudioAutoencoder as Magi2AudioVAE
from .config import (
    AudioVAEConfig,
    TurboDecoderConfig,
    VideoVAEConfig,
    WAN22_LATENTS_MEAN,
    WAN22_LATENTS_STD,
)
from .turbo_decoder import TurboVAED, get_turbo_vaed
from .turbo_decoder import TurboVAED as Magi2TurboDecoder
from .video_vae import Wan2_2_VAE, get_vae2_2
from .video_vae import Wan2_2_VAE as Magi2VideoVAE

__all__ = [
    # video encode VAE
    "Wan2_2_VAE",
    "Magi2VideoVAE",
    "get_vae2_2",
    # turbo decode-only student
    "TurboVAED",
    "Magi2TurboDecoder",
    "get_turbo_vaed",
    # audio VAE
    "AudioAutoencoder",
    "Magi2AudioVAE",
    "create_model_from_config",
    # configs / stats
    "VideoVAEConfig",
    "TurboDecoderConfig",
    "AudioVAEConfig",
    "WAN22_LATENTS_MEAN",
    "WAN22_LATENTS_STD",
]
