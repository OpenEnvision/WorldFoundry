"""PixArt-style AdaLN-single for LTX timestep modulation.

Produces a packed coefficient vector of length 6 (self-attn + FFN +
output scale/shift) or 9 when text cross-attn also uses AdaLN.  Tokens
remain ``B T D``; the linear maps sinusoidal timestep features to those
coefficients.
"""

from typing import Optional, Tuple

import torch

from worldfoundry.core.nn.diffusion.timestep import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)

# Number of AdaLN modulation parameters per transformer block.
# Base: 2 params (shift + scale) x 3 norms (self-attn, feed-forward, output).
ADALN_NUM_BASE_PARAMS = 6
# Cross-attention AdaLN adds 3 more (scale, shift, gate) for the CA norm.
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    """Total number of AdaLN parameters per block."""
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0)


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(linear(silu(emb(t))), emb(t))`` for block AdaLN and the final head."""
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
