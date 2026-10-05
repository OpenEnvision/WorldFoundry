"""Native StepVideo DiT (PixArt-style AdaLN-single + 3D RoPE).

Re-exports :class:`StepVideoModel`.  Latents enter as ``B F C H W``;
per-frame 2D patch embed yields ``(B·F) S C``, then tokens are folded to
``B (F·S) C`` for the transformer.  Output is unpatchified back to
``B F C H W``.
"""

from .model import StepVideoModel

__all__ = ["StepVideoModel"]
