"""Wan :class:`~...contracts.LatentInitializer` roles (T2V, I2V, TI2V, VACE, reference).

Re-exports factories from :mod:`.component`.  T2V draws ``[B,C,T,H/8,W/8]``
noise with 4n+1 frame geometry.  I2V / TI2V / VACE / reference variants
implement :class:`~...contracts.EncodedLatentInitializer` and encode
``DiffusionRequest.inputs`` media through the recipe-bound Wan VAE.
"""

from .component import (
    WAN_DENOISE_MASK_IS_ALL_ONES,
    WanCausalFirstFrameLatentInitializer,
    WanImageToVideoLatentInitializer,
    WanReferenceLatentInitializer,
    WanTextToVideoLatentInitializer,
    WanTextImageToVideoLatentInitializer,
    WanVaceLatentInitializer,
    build_wan_causal_i2v_latent_initializer,
    build_wan_i2v_latent_initializer,
    build_wan_reference_latent_initializer,
    build_wan_t2v_latent_initializer,
    build_wan_ti2v_latent_initializer,
    build_wan_vace_latent_initializer,
)

__all__ = [
    "WAN_DENOISE_MASK_IS_ALL_ONES",
    "WanCausalFirstFrameLatentInitializer",
    "WanImageToVideoLatentInitializer",
    "WanReferenceLatentInitializer",
    "WanTextToVideoLatentInitializer",
    "WanTextImageToVideoLatentInitializer",
    "WanVaceLatentInitializer",
    "build_wan_causal_i2v_latent_initializer",
    "build_wan_i2v_latent_initializer",
    "build_wan_reference_latent_initializer",
    "build_wan_t2v_latent_initializer",
    "build_wan_ti2v_latent_initializer",
    "build_wan_vace_latent_initializer",
]
