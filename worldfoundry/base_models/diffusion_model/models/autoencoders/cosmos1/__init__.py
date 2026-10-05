"""Cosmos Predict1 Tokenize1 :class:`~...contracts.LatentEncoder` / decoder.

Package entry for the official Tokenize1 TorchScript codec.  The
runner-facing type is :class:`~.component.Cosmos1VideoCodec`.

Pixels are BCTHW RGB in ``[-1, 1]``.  Encode applies per-chunk
mean/std then ``latent_scale`` (σ_data = 0.5) to 16-channel latents
at 8× spatial / 8× temporal compression.  Only 1-frame stills or
121-pixel / 16-latent video chunks are supported.

Recipes bind :func:`build_cosmos1_video_codec`.  Inner CNN graphs
live in the TorchScript artifacts, not in this package.
"""

from .component import Cosmos1VideoCodec, build_cosmos1_video_codec, load_cosmos1_video_codec

__all__ = ["Cosmos1VideoCodec", "build_cosmos1_video_codec", "load_cosmos1_video_codec"]
