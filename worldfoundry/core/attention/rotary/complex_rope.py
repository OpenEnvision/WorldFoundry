"""Complex-valued rotary tables used by video diffusion transformers.

Some Wan checkpoints store RoPE as complex cis tables (``cis = exp(iθ)``)
rather than separate cos/sin. This module builds those tables and
applies them with a complex multiply so the checkpoint math stays
bit-compatible. Real cos/sin 3D RoPE — including the CP-aware and
KV-cache-relative variants — lives in
:mod:`worldfoundry.core.attention.rotary.rope`.

:func:`complex_rotary_frequencies` is 1-D; :func:`complex_rotary_frequencies_3d`
packs time/height/width into one table for video tokens.
"""

from __future__ import annotations

import torch
from einops import rearrange


def complex_rotary_frequencies(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
    *,
    subdivisions: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a complex RoPE table, optionally at fractional position steps."""

    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")
    if end < 0 or subdivisions <= 0:
        raise ValueError("end must be non-negative and subdivisions must be positive")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    indices = torch.arange(0, dim, 2, dtype=torch.float64, device=resolved_device)
    inverse_frequencies = 1.0 / (float(theta) ** (indices / dim))
    positions = torch.arange(end * subdivisions, dtype=torch.float64, device=resolved_device) / subdivisions
    angles = torch.outer(positions, inverse_frequencies)
    return torch.polar(torch.ones_like(angles), angles)


def complex_rotary_frequencies_3d(
    dim: int,
    end: int = 1024,
    theta: float = 10_000.0,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build temporal, height, and width complex RoPE tables."""

    spatial_dim = dim // 3
    temporal_dim = dim - 2 * spatial_dim
    return (
        complex_rotary_frequencies(temporal_dim, end, theta, device=device),
        complex_rotary_frequencies(spatial_dim, end, theta, device=device),
        complex_rotary_frequencies(spatial_dim, end, theta, device=device),
    )


def assemble_complex_rotary_frequencies_3d(
    frequency_tables: tuple[torch.Tensor, ...],
    grid_size: tuple[int, int, int],
    *,
    start_frame: int = 0,
) -> torch.Tensor:
    """Expand existing T/H/W tables into ``[T*H*W,1,D/2]`` token order.

    Temporal positions are absolute frame positions, including after KV
    eviction. The input tables' complex precision and multiply convention
    are preserved; no trigonometry or precision conversion is performed.
    """
    frames, height, width = grid_size
    temporal, vertical, horizontal = frequency_tables
    return torch.cat(
        [
            temporal[start_frame : start_frame + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            vertical[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            horizontal[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1)


def apply_complex_rotary_embedding(
    value: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
    *,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Apply a broadcast complex RoPE table to ``[B,S,H*D]`` tokens.

    ``compute_dtype`` is explicit because casting only the frequency table does
    not select a lower-precision complex multiply: a float64 activation would
    promote a complex64 table back to complex128.  Keeping the activation and
    table on the same real/complex precision makes fp32 and fp64 genuine,
    independently testable execution paths.
    """

    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("complex RoPE compute_dtype must be float32 or float64")

    value = rearrange(value, "b s (h d) -> b s h d", h=int(num_heads))
    complex_value = torch.view_as_complex(value.to(compute_dtype).reshape(*value.shape[:-1], -1, 2).contiguous())
    complex_dtype = torch.complex64 if compute_dtype is torch.float32 else torch.complex128
    if frequencies.device.type == "npu":
        complex_dtype = torch.complex64
    frequencies = frequencies.to(complex_dtype)
    output = torch.view_as_real(complex_value * frequencies).flatten(2)
    return output.to(value.dtype)


__all__ = [
    "apply_complex_rotary_embedding",
    "assemble_complex_rotary_frequencies_3d",
    "complex_rotary_frequencies",
    "complex_rotary_frequencies_3d",
]
