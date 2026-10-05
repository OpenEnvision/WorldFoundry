"""Wan 2.1 / 2.2 :class:`~...contracts.LatentEncoder` / :class:`~...contracts.LatentDecoder`.

Package entry for the default Wan video codec.
:class:`~.component.WanVideoDecoder` is the runner-facing adapter
(16-ch 8×/4× or 48-ch VAE38).  :mod:`.model` holds the causal
3-D CNN.  Pixels and latents are BCTHW with official mean/std.

Factories remap official / Diffusers checkpoints
(:func:`convert_wan21_vae_state_dict` and Diffusers converters).
Variants (camera, action, linear, TAE, geometry) live under
:mod:`.variants`.  Cosmos 2.5 / Gamma-World reuse this family.
"""

from .component import (
    WanTAEPreviewDecoder,
    WanVideoDecoder,
    build_diffusers_wan_video_codec,
    build_wan_video_decoder,
    build_wan_video_vae38_decoder,
    convert_diffusers_wan22_vae_state_dict,
    convert_diffusers_wan_vae_state_dict,
    convert_wan21_vae_state_dict,
    load_wan_video_codec,
)
from .model import (
    WanVideoVAE,
    WanVideoVAE38,
    WanVideoVAEStateDictConverter,
)

__all__ = [
    "WanVideoVAE",
    "WanVideoVAE38",
    "WanVideoVAEStateDictConverter",
    "WanTAEPreviewDecoder",
    "WanVideoDecoder",
    "build_diffusers_wan_video_codec",
    "build_wan_video_decoder",
    "build_wan_video_vae38_decoder",
    "convert_diffusers_wan22_vae_state_dict",
    "convert_diffusers_wan_vae_state_dict",
    "convert_wan21_vae_state_dict",
    "load_wan_video_codec",
]
