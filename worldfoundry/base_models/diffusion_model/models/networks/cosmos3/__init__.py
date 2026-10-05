"""Native Cosmos3 omni Mixture-of-Tokens transformer.

Re-exports :class:`Cosmos3OmniTransformer`, the joint packed-sequence DiT
for text + vision (+ optional sound/action).  Vision cubes are ``C T H W``;
the backbone itself operates on a 1-D ``[S, hidden]`` stream split into
causal understanding and full-attention generation prefixes.
"""

from .model import Cosmos3OmniTransformer

__all__ = ["Cosmos3OmniTransformer"]
