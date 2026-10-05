"""Build :class:`LatentUpsampler` from serialized LTX checkpoint metadata.

Official LTX spatial-upsampler safetensors store channel counts,
2-D vs 3-D convs, and spatial / temporal upsample flags in JSON
metadata.  :meth:`LatentUpsamplerConfigurator.from_config` reads
those keys (with the same defaults as the released 2× spatial CNN)
and returns the graph used between multi-stage LTX passes.

This is construction only; encode/decode tiling and latent
normalization live on the VAE / processor side.
"""

from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.model import LatentUpsampler


class LatentUpsamplerConfigurator:
    """Build :class:`LatentUpsampler` from official safetensors JSON metadata."""

    @classmethod
    def from_config(cls: type[LatentUpsampler], config: dict) -> LatentUpsampler:
        """Read channel / upsample flags and return the CNN."""
        in_channels = config.get("in_channels", 128)
        mid_channels = config.get("mid_channels", 512)
        num_blocks_per_stage = config.get("num_blocks_per_stage", 4)
        dims = config.get("dims", 3)
        spatial_upsample = config.get("spatial_upsample", True)
        temporal_upsample = config.get("temporal_upsample", False)
        spatial_scale = config.get("spatial_scale", 2.0)
        rational_resampler = config.get("rational_resampler", False)
        return LatentUpsampler(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            dims=dims,
            spatial_upsample=spatial_upsample,
            temporal_upsample=temporal_upsample,
            spatial_scale=spatial_scale,
            rational_resampler=rational_resampler,
        )
