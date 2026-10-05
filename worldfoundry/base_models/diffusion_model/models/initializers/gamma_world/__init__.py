"""Gamma-World multi-view :class:`~...contracts.EncodedLatentInitializer`.

Re-exports the Wan-geometry initializer that encodes one first frame per
player and returns shared multi-view noise plus condition tensors.
"""

from .component import GammaWorldLatentInitializer, build_gamma_world_latent_initializer

__all__ = ["GammaWorldLatentInitializer", "build_gamma_world_latent_initializer"]
