"""Instance-local memory, control, and research extensions.

Hooks installed on one :class:`~..runners.base.NativeDiffusionRunner`.  They
observe or transform ``Conditioning`` / ``DenoiserInput`` / latents without
mutating class definitions or process-wide registries.  Approximate-attention
step driving lives in :mod:`.approximate_attention_step` and is imported by
the runner strategy, not re-exported here.
"""

from .base import DiffusionExtension, DiffusionRunContext
from .frozen_context import FrozenContextSuffixExtension
from .frozen_mask import FrozenLatentMaskExtension

__all__ = [
    "DiffusionExtension",
    "DiffusionRunContext",
    "FrozenContextSuffixExtension",
    "FrozenLatentMaskExtension",
]
