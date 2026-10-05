"""State-neutral helpers shared by official-layout Wan transformers.

Wan 2.1 / 2.2 reference models and several research forks (action, camera,
causal) keep a list-of-videos packing layout.  This mixin supplies the
matching ``unpatchify`` reconstruction and the official Xavier / normal
weight-init recipe so those forks do not copy the same methods.

It does not add action, camera, VACE, causal attention, linear attention,
or TeaCache hooks; those stay on the concrete variant classes.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class WanTransformerMethodsMixin:
    """Shared ``unpatchify`` and ``init_weights`` for official Wan graphs."""

    def unpatchify(self, values, grid_sizes, channels=None):
        """Fold packed ``[L, C_out * prod(patch)]`` tokens back to ``[C, F, H, W]``.

        Each sample may have its own ``(F, H, W)`` patch grid.  Tokens past
        ``prod(grid)`` are padding and are discarded.
        """
        channels = self.out_dim if channels is None else channels
        output = []
        for value, grid in zip(values, grid_sizes.tolist()):
            value = value[: math.prod(grid)].view(
                *grid,
                *self.patch_size,
                channels,
            )
            value = torch.einsum("fhwpqrc->cfphqwr", value)
            value = value.reshape(
                channels,
                *[
                    grid_size * patch_size
                    for grid_size, patch_size in zip(grid, self.patch_size)
                ],
            )
            output.append(value)
        return output

    def init_weights(self):
        """Apply the official Wan init: Xavier linears, N(0, 0.02) embeddings, zero head."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


__all__ = ["WanTransformerMethodsMixin"]
