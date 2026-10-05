"""LTX spatial latent upsampler and :class:`~...contracts.LatentProcessor`.

:func:`build_ltx_spatial_latent_processor` is the recipe factory.  It
unpatchifies stage-1 video, runs the checkpointed Conv3d upsampler, and
forwards audio unchanged.  :class:`LatentUpsampler` is the inner CNN.
"""

from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.component import (
    LTXSpatialLatentProcessor,
    build_ltx_spatial_latent_processor,
)
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import LatentUpsampler, upsample_video
from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model_configurator import (
    LatentUpsamplerConfigurator,
)

__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "LTXSpatialLatentProcessor",
    "build_ltx_spatial_latent_processor",
    "upsample_video",
]
