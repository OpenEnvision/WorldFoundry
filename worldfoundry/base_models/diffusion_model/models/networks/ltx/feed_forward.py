"""LTX DiT feed-forward: GELU-tanh expand then linear contract.

Operates on packed ``B T D`` tokens.  The Identity slot preserves the
checkpoint's three-module ``net`` key layout.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    """GELU-approx MLP used after self/cross-attn in each LTX block."""

    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[..., dim]`` tokens to ``[..., dim_out]``."""
        return self.net(x)
