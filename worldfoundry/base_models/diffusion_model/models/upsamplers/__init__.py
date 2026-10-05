"""Stage-boundary :class:`~...contracts.LatentProcessor` and spatial upsamplers.

LTX-2 / LTX-Video use :class:`~.ltx.component.LTXSpatialLatentProcessor`
between resolution stages.  HunyuanVideo 1.5 ships a spatial upsampler
used by its own super-resolution pass.  Processors implement
``process(states, request) -> dense latents``; they are not denoisers.
"""

__all__: list[str] = []
