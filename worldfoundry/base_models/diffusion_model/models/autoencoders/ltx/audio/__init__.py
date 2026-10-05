"""LTX audio VAE decoder, vocoder, and causal 2-D conv blocks.

Subpackage entry for LTX spectrogram-like audio latents.
:class:`~.audio_vae.AudioEncoder` / :class:`AudioDecoder` plus
:func:`encode_audio` / :func:`decode_audio` are the public
surfaces.  :class:`~.vocoder.Vocoder` turns latents into PCM.

Convolutions are causal 2-D (no future-frame leak) along
:class:`~.causality_axis.CausalityAxis`.  Configurators rebuild
graphs from checkpoint JSON.

Used by :class:`~..component.LTXMediaDecoder` after video decode.
"""

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import (
    AudioDecoder,
    AudioEncoder,
    decode_audio,
    encode_audio,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.model_configurator import (
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import Vocoder, VocoderWithBWE

__all__ = [
    "AudioDecoder",
    "AudioDecoderConfigurator",
    "AudioEncoder",
    "AudioEncoderConfigurator",
    "Vocoder",
    "VocoderConfigurator",
    "VocoderWithBWE",
    "decode_audio",
    "encode_audio",
]
