"""Checkpoint-shaped Wan codec variants (camera, action, TAE, geometry).

Namespace for Wan VAE graphs that keep the same BCTHW latent
contract as :mod:`~..model` but match a specific checkpoint
layout or extra channel map.

Included: camera-21, action-21, linear, light-22, TAEW2.2,
TAEHV preview, geometry-bridge, Fun-Control / Video-X, and
DreamX-World VAE38.

Recipes should prefer :class:`~..component.WanVideoDecoder`
unless the checkpoint requires one of these shapes.
"""

