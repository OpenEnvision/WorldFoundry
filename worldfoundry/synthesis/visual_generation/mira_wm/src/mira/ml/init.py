"""Weight initialisation for the codec and world-model modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AdaptiveLayerNorm


def init_weights(module: nn.Module) -> None:
    """Initialise a single module in place, dispatched on its type.

    Intended to be passed to `nn.Module.apply` so it runs over every submodule. Linear and
    convolutional weights are drawn from `N(0, 0.02)` with zero bias; embeddings from `N(0, 0.02)`;
    normalisation layers are set to unit weight and zero bias; and `AdaptiveLayerNorm`'s conditioning
    projection is zeroed so it starts as an identity modulation.

    Args:
        module: The submodule to initialise.
    """
    if isinstance(module, AdaptiveLayerNorm):
        nn.init.zeros_(module.gamma_beta.weight)
        nn.init.zeros_(module.gamma_beta.bias)
    elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            nn.init.zeros_(module.bias)
