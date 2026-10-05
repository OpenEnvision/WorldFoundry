"""Gamma-World multi-view Wan VAE :class:`~...contracts.LatentDecoder`.

Package entry for the multi-player world codec.
:class:`~.component.GammaWorldVideoCodec` reuses
:class:`~..wan.component.WanVideoDecoder`.

Encode is a Wan passthrough (BCTHW).  Decode splits packed time
``(V*T)`` into per-player clips, decodes each, then tiles views
along width as ``B C T H (V W)``.  ``n_players`` comes from
``DiffusionRequest.inputs``.

This is a Wan-family layout adapter, not a new CNN.
"""

from .component import GammaWorldVideoCodec, build_gamma_world_video_codec

__all__ = ["GammaWorldVideoCodec", "build_gamma_world_video_codec"]
