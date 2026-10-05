"""Checkpoint-compatible DC-AE graph used by Sana image models.

Re-exports :class:`~.efficientvit.dc_ae.DCAE`, :class:`DCAEConfig`,
and the f32c32 factory :func:`dc_ae_f32c32`.  The runner adapter
:class:`~..component.SanaDCAutoencoder` wraps this graph.

Image latents are BCHW, 32 channels, 32× spatial compression.
Temporal / streaming variants live in
:mod:`.efficientvit.dc_ae_with_temporal`.

This is the Sana image-codec family, not Wan / LTX / Hunyuan.
"""

from .efficientvit.dc_ae import DCAE, DCAEConfig, dc_ae_f32c32

__all__ = ["DCAE", "DCAEConfig", "dc_ae_f32c32"]
