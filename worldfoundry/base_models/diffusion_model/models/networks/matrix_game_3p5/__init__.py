"""Checkpoint-compatible Matrix Game 3.5 Wan-style DiT.

Re-exports :class:`WanModel`.  Latents are ``B C F H W``; Conv3d patch
embed yields ``B C F' H' W'`` then ``B (F'H'W') C``.  Optional PRoPE
uses camera matrices instead of (or in addition to) temporal RoPE.
"""

from .model import WanModel

__all__ = ["WanModel"]
