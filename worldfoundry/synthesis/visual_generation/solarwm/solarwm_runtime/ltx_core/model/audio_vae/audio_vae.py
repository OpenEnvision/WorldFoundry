"""Solar's disposal lifecycle over the native LTX audio codec graphs."""

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import (
    AudioDecoder as NativeAudioDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.audio_vae import (
    AudioEncoder as NativeAudioEncoder,
)

from ..disposable import Disposable


class AudioEncoder(NativeAudioEncoder, Disposable):
    """Native causal audio encoder with persistent-tensor disposal."""


class AudioDecoder(NativeAudioDecoder, Disposable):
    """Native causal audio decoder with persistent-tensor disposal."""
