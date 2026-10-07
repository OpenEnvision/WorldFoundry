from __future__ import annotations

from torch import Tensor


def apply_cfg(cond: Tensor, uncond: Tensor, guidance_weight: float) -> Tensor:
    """Classifier-free guidance: uncond + w * (cond - uncond)."""
    return uncond + guidance_weight * (cond - uncond)
