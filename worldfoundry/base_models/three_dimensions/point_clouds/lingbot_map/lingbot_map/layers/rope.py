# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.attention.rotary.rope_nd import get_1d_rotary_pos_embed_linear as get_1d_rotary_pos_embed


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components.
            # .detach().clone() ensures the cached tensors are plain CUDA tensors
            # (not CUDA-graph-owned memory), so they can safely be reused as inputs
            # to subsequent torch.compile / CUDA graph captures.
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype).detach().clone()
            sin_components = angles.sin().to(dtype).detach().clone()
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components.
        # Use positions.shape[1] (token count) as the frequency table size instead of
        # int(positions.max()) + 1.  Both are valid upper bounds, but shape[1] is a
        # static integer known at trace time, so it is CUDA-graph-compatible.
        # (positions.max() requires a device→host sync / aten._local_scalar_dense,
        # which prevents CUDA graph capture in torch.compile.)
        max_position = positions.shape[1]
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)


class WanRotaryPosEmbed(nn.Module):
    """Encode frame, height and width positions with independent rotary factors.

    Split attention_head_dim into even frame/height/width feature dimensions,
    precompute complex factors for each axis, then concatenate them. Explicit
    fhw_dim values must sum to attention_head_dim. With fhw_dim=None, height
    and width each use 2 * (attention_head_dim // 6) dimensions and time uses
    the remainder (24, 20, 20 for a head dimension of 64).
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: tuple[int, int, int] | None = [20, 22, 22],
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if fhw_dim is not None:
            assert attention_head_dim == sum(fhw_dim), (
                f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            )
            t_dim, h_dim, w_dim = fhw_dim
        else:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim

        self.fhw_dim = (t_dim, h_dim, w_dim)

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(
        self, ppf, pph, ppw, patch_start_idx, device: torch.device, f_start: int = 0, f_end: int | None = None
    ) -> torch.Tensor:
        """Return complex rotary factors in frame-major, special-token-then-patch order.

        Args:
            ppf: Frame count when f_end is None (frames start at zero).
            pph: Number of patch rows per frame.
            ppw: Number of patch columns per frame.
            patch_start_idx: Number of special tokens preceding each frame's patches.
            device: Destination for the cached frequency tensor, updated in place.
            f_start: First frame index when f_end is specified.
            f_end: Exclusive last frame index; overrides ppf with f_end - f_start.

        Returns:
            Complex factors [1, 1, frames * (patch_start_idx + pph * ppw), head_dim/2].
            Within each frame, special token i uses position (f, i, i); patches
            use (f, patch_start_idx + h, patch_start_idx + w). Without special
            tokens, patches use (f, h, w).
        """

        self.freqs = self.freqs.to(device)
        if hasattr(self, "fhw_dim") and self.fhw_dim is not None:
            t_dim, h_dim, w_dim = self.fhw_dim
        else:
            h_dim = w_dim = 2 * (self.attention_head_dim // 6)
            t_dim = self.attention_head_dim - h_dim - w_dim

        freqs = self.freqs.split_with_sizes(
            [
                t_dim // 2,
                h_dim // 2,
                w_dim // 2,
            ],
            dim=1,
        )

        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            frame_slice = slice(0, ppf)

        ## For other tokens
        # Special tokens occupy diagonal spatial positions; patches start after them.
        if patch_start_idx > 0:
            # camera: (f, 0, 0), register_0: (f, 1, 1), ..., scale: (f, 5, 5)
            # Shape: (ppf, patch_start_idx, dim)
            freqs_special_f = (
                freqs[0][frame_slice].reshape(ppf, 1, -1).expand(ppf, patch_start_idx, -1)
            )  # (ppf, patch_start_idx, dim_f)
            freqs_special_h = (
                freqs[1][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)
            )  # (ppf, patch_start_idx, dim_h)
            freqs_special_w = (
                freqs[2][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)
            )  # (ppf, patch_start_idx, dim_w)
            freqs_special = torch.cat(
                [freqs_special_f, freqs_special_h, freqs_special_w], dim=-1
            )  # (ppf, patch_start_idx, dim)
            freqs_special = freqs_special.reshape(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim)

            # Shape: (ppf, pph, ppw, dim)
            freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f)
            freqs_h = (
                freqs[1][patch_start_idx : patch_start_idx + pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
            )  # (ppf, pph, ppw, dim_h)
            freqs_w = (
                freqs[2][patch_start_idx : patch_start_idx + ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
            )  # (ppf, pph, ppw, dim_w)
            freqs_patches = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)  # (ppf, pph, ppw, dim)
            freqs_patches = freqs_patches.reshape(ppf, pph * ppw, -1)  # (ppf, pph * ppw, dim)

            # Concatenate special tokens and patches for each frame along the second dimension
            # Shape: (ppf, patch_start_idx + pph * ppw, dim)
            freqs = torch.cat([freqs_special, freqs_patches], dim=1)  # (ppf, patch_start_idx + pph * ppw, dim)

            # Flatten to get final shape: (ppf * (patch_start_idx + pph * ppw), dim)
            freqs = freqs.reshape(ppf * (patch_start_idx + pph * ppw), -1)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, ppf * (patch_start_idx + pph * ppw)
            return freqs

        freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f)
        freqs_h = freqs[1][:pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_h)
        freqs_w = freqs[2][:ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_w)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )  # (1, 1, ppf * pph * ppw, dim)
        return freqs


def apply_rotary_emb(x, freqs):
    """Apply 3D rotary position embedding using real arithmetic (torch.compile-safe).

    Equivalent to complex multiplication but avoids torch.view_as_complex /
    view_as_real, which are not supported by torchinductor and break CUDA graphs.

    Args:
        x: [B, H, N, D] real tensor (bfloat16 or float32).
        freqs: [1, 1, N, D//2] complex tensor (cos + i*sin per frequency).

    Returns:
        [B, H, N, D] tensor of same dtype as x.
    """
    # Real-arithmetic implementation: equivalent to (x1+i*x2)*(cos+i*sin) but avoids
    # torch.view_as_complex / view_as_real which break torch.compile CUDA graphs.
    cos = freqs.real.to(x.dtype)  # [1, 1, N, D//2]
    sin = freqs.imag.to(x.dtype)  # [1, 1, N, D//2]

    # Interleaved pairs: even indices = "real", odd indices = "imag"
    x1 = x[..., 0::2]  # [B, H, N, D//2]
    x2 = x[..., 1::2]  # [B, H, N, D//2]

    # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.stack([out1, out2], dim=-1).reshape(x.shape)
