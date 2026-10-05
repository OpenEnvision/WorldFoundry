"""LTX video VAE CNN, tiling, and encoder/decoder configurators.

Subpackage entry for the LTX 3-D residual UNet.
:class:`~.video_vae.VideoEncoder` / :class:`VideoDecoder` are the
public encode/decode surfaces.  Official compression is 8×
temporal / 32× spatial with 128 latent channels (BCTHW).

:class:`~.tiling.TilingConfig` enables spatial / temporal tiles
for long clips.  :class:`~.model_configurator.VideoEncoderConfigurator`
rebuilds the CNN from checkpoint metadata.

Consumed by :class:`~..component.LTXMediaDecoder` and tensor codecs.
"""

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.model_configurator import (
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import (
    VideoDecoder,
    VideoEncoder,
    get_video_chunks_number,
)

__all__ = [
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "TilingConfig",
    "VideoDecoder",
    "VideoDecoderConfigurator",
    "VideoEncoder",
    "VideoEncoderConfigurator",
    "get_video_chunks_number",
]
