"""Cosmos3 multimodal decoder: Wan2.2 VAE38 video + AVAE sound.

Package entry for the joint Cosmos3 media codec.
:class:`~.component.Cosmos3MediaDecoder` implements
:class:`~...runners.MultiModalLatentDecoder`.

Video latents are 48-channel Wan 2.2 VAE38 (BCTHW).  Audio uses
:class:`~.audio.Cosmos3AVAEAudioDecoder`.  Loaders remap official
Diffusers Wan2.2 keys onto the native VAE38.

Recipes bind :func:`build_cosmos3_media_decoder`.  Video CNN math
is shared with :mod:`..wan`; sound is Cosmos3-specific.
"""

from .audio import Cosmos3AVAEAudioDecoder
from .component import (
    Cosmos3MediaDecoder,
    build_cosmos3_media_decoder,
    load_cosmos3_audio_decoder,
    load_cosmos3_video_vae,
)

__all__ = [
    "Cosmos3AVAEAudioDecoder",
    "Cosmos3MediaDecoder",
    "build_cosmos3_media_decoder",
    "load_cosmos3_audio_decoder",
    "load_cosmos3_video_vae",
]
