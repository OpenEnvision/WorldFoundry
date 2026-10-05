"""LVDM / T2V-Turbo OpenCLIP condition encoders.

Package documentation surface for VideoCrafter-style CLIP
towers.  Recipes import :mod:`.condition` /
:mod:`~..t2v_turbo` factories rather than this ``__init__``.

``FrozenOpenCLIPEmbedder`` emits ``context`` tokens for the
LVDM UNet.  IP-Adapter resamplers are :mod:`.resampler`.

This is the OpenCLIP family, not T5 / Gemma / UMT5.
"""

__all__: list[str] = []
