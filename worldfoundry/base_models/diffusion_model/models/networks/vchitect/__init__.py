"""Native Vchitect-2 joint transformer (spatial + temporal + text).

Re-exports :class:`VchitectXLTransformerModel`.  Latents are ``B F C H W``;
each frame is 2D-patched to ``(B·F) (H'W') C``.  Blocks run spatial
joint attention, temporal attention with 1D RoPE over frames, and a
cross term against pooled text.
"""

from .model import VchitectXLTransformerModel

__all__ = ["VchitectXLTransformerModel"]
