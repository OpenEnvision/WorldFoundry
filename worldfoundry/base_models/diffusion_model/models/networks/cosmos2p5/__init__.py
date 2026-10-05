"""Native Cosmos Predict 2.5 / Transfer 2.5 diffusion transformers.

Re-exports the checkpoint-shaped DiT (:class:`Cosmos25Transformer3DModel`)
and the VACE control variant (:class:`Cosmos25Transfer3DModel`).  Latents
are ``B C T H W``; tokens inside the stack are ``B (T'H'W') C``.
"""

from .model import Cosmos25Transfer3DModel, Cosmos25Transformer3DModel

__all__ = [
    "Cosmos25Transfer3DModel",
    "Cosmos25Transformer3DModel",
]
