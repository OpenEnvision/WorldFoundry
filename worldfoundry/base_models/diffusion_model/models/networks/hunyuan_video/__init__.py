"""Checkpoint-compatible HunyuanVideo DiT family.

Re-exports :class:`HunyuanVideoDiT` (original HYVideo MM-DiT).  Latents
are ``B C T H W`` (or i2v packed channels); after 3D patch embed tokens
are ``B S C``.  Double-stream blocks keep video and text separate until
joint attention; later single-stream blocks concatenate them.
"""

from .original import HunyuanVideoDiT

__all__ = ["HunyuanVideoDiT"]
