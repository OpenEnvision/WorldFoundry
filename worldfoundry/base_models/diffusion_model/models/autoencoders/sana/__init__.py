"""Sana DC-AE :class:`~...contracts.LatentEncoder` / decoder.

Package entry for Sana image latents.  :class:`~.component.SanaDCAutoencoder`
wraps the f32c32 Deep-Compression Autoencoder in :mod:`.dc_ae`.

Image-only: BCHW RGB in, 32-channel latents at 32× spatial
compression, BCHW out.  ``encode`` multiplies by
``scaling_factor`` (~0.414); ``decode`` inverts that scale and
clamps pixels to ``[-1, 1]``.

Recipes bind :func:`build_sana_dc_autoencoder`.  Temporal DC-AE
graphs live under :mod:`.dc_ae.efficientvit` for video forks.
"""

from .component import SanaDCAutoencoder, build_sana_dc_autoencoder

__all__ = ["SanaDCAutoencoder", "build_sana_dc_autoencoder"]
