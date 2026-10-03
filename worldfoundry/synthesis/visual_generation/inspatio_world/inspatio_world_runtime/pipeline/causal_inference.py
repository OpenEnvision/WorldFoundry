"""Legacy checkpoint loading around the shared InSpatio causal sampler."""

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from worldfoundry.base_models.diffusion_model.runners.inspatio_world import (
    InspatioCausalRollout,
    denoise_block,
)


class CausalInferencePipeline(InspatioCausalRollout):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        folder = getattr(args, "wan_model_folder", None)
        super().__init__(
            args,
            generator=generator if generator is not None else WanDiffusionWrapper(**getattr(args, "generator", {}), is_causal=True),
            text_encoder=text_encoder if text_encoder is not None else WanTextEncoder(model_folder=folder),
            vae=vae if vae is not None else WanVAEWrapper(model_folder=folder),
        )


__all__ = ["CausalInferencePipeline", "denoise_block"]
