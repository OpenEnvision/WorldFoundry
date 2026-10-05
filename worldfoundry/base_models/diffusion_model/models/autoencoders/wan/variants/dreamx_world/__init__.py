"""DreamX-World Wan 2.2 48-channel VAE adapter.

Subpackage entry for the DreamX-World world-model codec.
:class:`~.vae.AutoencoderKLWan3_8` wraps the canonical Wan 2.2
VAE38 CNN via :class:`~...adapter.WanAutoencoderAdapterMixin`.

Latents are 48-channel BCTHW with official Wan 2.2 mean/std,
4× temporal / 8× spatial compression.  This is a loading /
naming adapter, not a new residual graph.

Sibling of :mod:`~...reference_22` and the runner
:class:`~...component.WanVideoDecoder` VAE38 path.
"""

