"""Native Cosmos Predict1 / GEN3C diffusion transformer.

Re-exports :class:`Cosmos1Transformer3DModel`, the checkpoint-shaped 7B DiT
that consumes packed ``B C T H W`` video+camera latents (``in_channels=81``)
and produces video residual latents.  Block primitives live in
``cosmos2p5.model``; this package only adds the factorized learnable T/H/W
position table.
"""

from .model import Cosmos1Transformer3DModel

__all__ = ["Cosmos1Transformer3DModel"]
