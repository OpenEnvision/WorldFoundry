"""Family :class:`~...contracts.LatentEncoder` / :class:`~...contracts.LatentDecoder` adapters.

Recipes bind ``build_*`` factories via :class:`~...components.ComponentSpec`.
Each codec implements ``encode(pixels)`` and/or
``decode(latents, request)``.  ``request.inputs[\"return_latent\"]`` skips
pixel decode.  Inner CNN/ViT graphs live in sibling packages (Wan, LTX,
HunyuanVideo, MAGI-2, MiniMax H3, Sana DC-AE, …).
"""

__all__: list[str] = []
