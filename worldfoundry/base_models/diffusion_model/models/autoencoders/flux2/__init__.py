"""Native FLUX.2 autoencoder factory and encode helpers.

Package entry for the FLUX.2 2-D AE used as a still / reference
codec.  :func:`~.component.build_flux2_autoencoder` loads
``ae.safetensors`` from ``black-forest-labs/FLUX.2-dev``.

The network is :class:`~.model.Flux2Autoencoder` (BCHW RGB,
32 latent channels).  :func:`~.component.encode_video_batch_refs`
folds ``[B, T, H, W, C]`` frames through that 2-D AE.

This is an image-family codec in the autoencoder tree, not a
causal video VAE like Wan / LTX / Hunyuan.
"""

from .component import (
    FLUX2_AUTOENCODER_FILENAME,
    FLUX2_REPO_ID,
    build_flux2_autoencoder,
    default_flux2_autoencoder_checkpoint,
    encode_video_batch_refs,
    load_flux2_autoencoder,
)
from .model import Flux2Autoencoder, Flux2AutoencoderConfig

__all__ = [
    "FLUX2_AUTOENCODER_FILENAME",
    "FLUX2_REPO_ID",
    "Flux2Autoencoder",
    "Flux2AutoencoderConfig",
    "build_flux2_autoencoder",
    "default_flux2_autoencoder_checkpoint",
    "encode_video_batch_refs",
    "load_flux2_autoencoder",
]
