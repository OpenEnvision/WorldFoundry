"""Tanh-approximate GELU used as the LTX transformer FFN expansion.

LTX DiT feed-forwards project ``[..., dim_in]`` to ``[..., dim_out]``
then apply ``gelu(..., approximate='tanh')``.  The tanh form matches
the official checkpoint (exact GELU would drift).  This module is the
expansion half of the FFN; the contraction linear lives in the parent
block.

Used only by the native LTX video DiT, not by the VAE or upsampler.
"""

import torch


class GELUApprox(torch.nn.Module):
    """Linear projection followed by ``gelu(..., approximate='tanh')``."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand ``[..., dim_in]`` to ``[..., dim_out]`` with tanh-GELU."""
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")
