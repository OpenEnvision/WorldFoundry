"""Solar's disposal lifecycle over the native LTX latent upsampler."""

from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import (
    LatentUpsampler as NativeLatentUpsampler,
)

from ..disposable import Disposable


class LatentUpsampler(NativeLatentUpsampler, Disposable):
    """Native spatial/temporal latent upsampling with persistent-tensor disposal."""
