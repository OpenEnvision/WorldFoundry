"""Cosmos Predict 2.5 Wan-VAE :class:`~...contracts.LatentEncoder` / decoder.

Package entry for Predict 2.5 video latents.  :class:`~.component.Cosmos25VideoCodec`
wraps the shared Wan 2.1 :class:`~..wan.model.WanVideoVAE` after the
official key remap.

Pixels are BCTHW RGB in ``[-1, 1]`` (or Canny edge frames for
Transfer-2.5).  Encode may go through
:func:`~...initializers.video_conditioning.prepare_video_conditioning_pixels`.
Decode honors ``return_latent``.

Recipes bind :func:`build_cosmos25_video_codec`.  This is the 2.5
sibling of Tokenize1 (:mod:`..cosmos1`) and Cosmos3 (:mod:`..cosmos3`).
"""

from .component import Cosmos25VideoCodec, build_cosmos25_video_codec

__all__ = ["Cosmos25VideoCodec", "build_cosmos25_video_codec"]
