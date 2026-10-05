"""LongVie dual dense/sparse control branch for native Wan 2.x models.

This module owns only the extra ControlNet-style parameters.  Dense and
sparse control videos each run through a half-width stack of canonical
:class:`DiTBlock` layers; per-layer combine linears map hints back to
full hidden size for the host :class:`WanModel`.

No VACE, action, camera Plücker, causal attention, linear attention, or
TeaCache hooks live here.
"""

from __future__ import annotations

import torch
from torch import nn

from ..model import DiTBlock


class WanDualControlModel(nn.Module):
    """Run dense and sparse controls through paired half-width Wan blocks.

    The module intentionally owns only LongVie's additional parameters.  The
    base diffusion transformer remains :class:`WanModel`, so checkpoint
    loading and execution continue to use the shared native Wan stack.
    """

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        eps: float,
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        control_layers: int = 12,
    ) -> None:
        super().__init__()
        if dim % 2 or ffn_dim % 2 or num_heads % 2:
            raise ValueError("Wan dual control requires even model, FFN, and head dimensions")
        if not 0 < control_layers <= num_layers:
            raise ValueError("control_layers must be within the base Wan transformer depth")

        self.control_layers = int(control_layers)
        block_args = (has_image_input, dim // 2, num_heads // 2, ffn_dim // 2, eps)
        self.control_blocks_dense = nn.ModuleList(
            [DiTBlock(*block_args) for _ in range(self.control_layers)]
        )
        self.control_blocks_sparse = nn.ModuleList(
            [DiTBlock(*block_args) for _ in range(self.control_layers)]
        )

        self.control_initial_combine_linear_dense = nn.Linear(dim, dim // 2)
        self.control_initial_combine_linear_sparse = nn.Linear(dim, dim // 2)
        self.control_text_linear = nn.Linear(dim, dim // 2)
        self.control_t_mod = nn.Linear(dim, dim // 2)
        self.control_combine_linears = nn.ModuleList(
            [nn.Linear(dim // 2, dim) for _ in range(self.control_layers)]
        )

        nn.init.zeros_(self.control_initial_combine_linear_dense.weight)
        nn.init.zeros_(self.control_initial_combine_linear_dense.bias)
        nn.init.zeros_(self.control_initial_combine_linear_sparse.weight)
        nn.init.zeros_(self.control_initial_combine_linear_sparse.bias)
        for projection in self.control_combine_linears:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)


# Released LongVie checkpoints use the historical symbol in their prefixes.
WanModelDualControl = WanDualControlModel


__all__ = ["WanDualControlModel", "WanModelDualControl"]
