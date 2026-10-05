"""Checkpoint-compatible LVDM 3D UNets (not DiT).

Latents stay ``B C T H W`` (OpenAI UNet convention: often ``B C F H W``).
Down/up residual blocks mix spatial convs with temporal conv or
spatial/temporal transformers.  Short trainer, vid2world, and next
variants live in sibling modules.
"""

__all__: list[str] = []
