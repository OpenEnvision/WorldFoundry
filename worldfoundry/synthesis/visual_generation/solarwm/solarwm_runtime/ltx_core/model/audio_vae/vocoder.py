"""Solar's disposal lifecycle over the native LTX vocoder graphs."""

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import (
    Vocoder as NativeVocoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.vocoder import (
    VocoderWithBWE as NativeVocoderWithBWE,
)

from ..disposable import Disposable


class Vocoder(NativeVocoder, Disposable):
    """Native mel-to-waveform graph with Solar's persistent-tensor disposal."""


class VocoderWithBWE(NativeVocoderWithBWE, Disposable):
    """Native FP32 bandwidth extension with Solar's disposal lifecycle."""
