"""2D rotary position embeddings for patch-token attention.

Height/width only (no time axis). Used by image ViTs and 2D DiT
heads that tokenize a single frame. :class:`PositionGetter` caches
the ``(h, w)`` patch grid so a repeated forward does not rebuild
coordinates. :class:`RotaryPositionEmbedding2D` applies the axial
cos/sin tables.

Not this module:
    Video tokens that also have a time index must use :mod:`.rope`
    (3D) or :mod:`.rope_nd`. Sequence-parallel freq slicing lives in
    :mod:`.sequence_parallel_rope`. This module does not own attention.

Public surface:

- :class:`PositionGetter` — cached ``(y, x)`` patch coordinates.
- :class:`RotaryPositionEmbedding2D` — axial cos/sin apply.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ──────────────────────────────────────────────────────────────────────────
# Patch grid cache — rebuild only when (H, W) or device changes
# ──────────────────────────────────────────────────────────────────────────


class PositionGetter:
    """Cache 2D patch-grid positions by grid shape."""

    def __init__(self) -> None:
        """Start with an empty ``(height, width) → coordinates`` cache."""

        self.position_cache: Dict[tuple[int, int], Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        """Return ``[B, H*W, 2]`` integer ``(y, x)`` coords, cloned per caller.

        The clone is required because some ViT blocks write into the
        returned tensor; sharing the cached storage would corrupt the
        next forward. Device mismatch rebuilds the table rather than
        calling ``.to`` on a potentially large cache entry.
        """

        key = (int(height), int(width))
        cached = self.position_cache.get(key)
        if cached is None or cached.device != device:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            cached = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[key] = cached
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


# ──────────────────────────────────────────────────────────────────────────
# Axial 2D RoPE — half the channels rotate Y, half rotate X
# ──────────────────────────────────────────────────────────────────────────


class RotaryPositionEmbedding2D(nn.Module):
    """Apply rotary embeddings to tokens with ``(y, x)`` patch coordinates."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0) -> None:
        """Cache cos/sin tables keyed by ``(dim, seq_len, device, dtype)``."""

        super().__init__()
        self.base_frequency = float(frequency)
        self.scaling_factor = float(scaling_factor)
        self.frequency_cache: Dict[tuple[int, int, torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """Apply Y then X RoPE to even-width features.

        Failure: odd last-dim or a positions tensor that is not
        ``(B, N, 2)`` — those layouts belong to 1-D / 3-D helpers.
        """

        if tokens.size(-1) % 2:
            raise ValueError("Feature dimension must be even.")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("Positions must have shape (batch_size, n_tokens, 2).")

        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.abs().max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    def _compute_frequency_components(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Build or reuse duplicated-angle ``(cos, sin)`` tables for one axis."""

        cache_key = (int(dim), int(seq_len), device, dtype)
        cached = self.frequency_cache.get(cache_key)
        if cached is not None:
            return cached

        exponents = torch.arange(0, dim, 2, device=device).float() / dim
        inv_freq = 1.0 / (self.base_frequency**exponents)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) * self.scaling_factor
        angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cached = (angles.cos().to(dtype), angles.sin().to(dtype))
        self.frequency_cache[cache_key] = cached
        return cached

    @staticmethod
    def _rotate_features(value: Tensor) -> Tensor:
        """Half-split rotate ``[-x2, x1]`` used by the non-interleaved pair layout."""

        feature_dim = value.shape[-1]
        first, second = value[..., : feature_dim // 2], value[..., feature_dim // 2 :]
        return torch.cat((-second, first), dim=-1)

    def _apply_1d_rope(self, tokens: Tensor, positions: Tensor, cos_comp: Tensor, sin_comp: Tensor) -> Tensor:
        """Gather per-token angles via embedding so positions need not be dense."""

        absolute_positions = positions.abs()
        cos = F.embedding(absolute_positions, cos_comp)[:, None, :, :]
        sin = F.embedding(absolute_positions, sin_comp)[:, None, :, :]
        # Special tokens may use negative coordinates (for example CUT3R poses).
        sin = sin * positions.sign()[:, None, :, None]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)


__all__ = ["PositionGetter", "RotaryPositionEmbedding2D"]
