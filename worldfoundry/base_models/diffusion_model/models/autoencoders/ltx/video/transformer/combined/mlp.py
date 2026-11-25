"""Combined pathway MLP: out-of-place AdaLN SwiGLU residual."""

from __future__ import annotations

import torch
from torch import nn

from ..layers import modulate
from ..swiglu import SwiGLUTileSpec, swiglu_tiled
from ..triton_diffvae_ops import diffvae_rmsnorm_modulate


def residual_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tile: SwiGLUTileSpec,
) -> torch.Tensor:
    """Combined*: ``x + swiglu_tiled(modulate(norm(x), scale, shift))``."""
    if isinstance(norm, nn.RMSNorm) and not norm._forward_hooks and not norm._forward_pre_hooks:
        eps = float(norm.eps) if norm.eps is not None else float(torch.finfo(x.dtype).eps)
        y = diffvae_rmsnorm_modulate(x, norm.weight, scale, shift, eps=eps)
    else:
        y = modulate(norm(x), scale, shift)
    if y.numel() == 0:
        return x
    return x + swiglu_tiled(y, mlp.w_gate.weight, mlp.w_up.weight, mlp.w_down.weight, tile)
