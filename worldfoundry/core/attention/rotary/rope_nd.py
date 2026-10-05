"""Generic n-D rotary position embedding utilities.

Some architectures (n-D DiTs, multi-axis world models) rotate tokens
along more than three axes. This module builds a meshgrid of integer
positions, looks up a 1-D frequency table per axis, and composes them
through :func:`worldfoundry.core.attention.rotary.rope.apply_rotary_embedding`.

Prefer :mod:`worldfoundry.core.attention.rotary.rope` / :mod:`.rope_2d` when
the layout is known — those modules match released checkpoint
frequency splits (time vs height vs width head-dim slices). This
generic path is for adapters that cannot hard-code the axis count.

Not this module:
    Not a drop-in for Wan 3D RoPE (2:2:2 split). Not sequence-parallel.
    Complex cis tables live in :mod:`.complex_rope`.

Public surface:

- :func:`get_meshgrid_nd` / :func:`get_nd_rotary_pos_embed` /
  :func:`get_1d_rotary_pos_embed` — table builders.
- :func:`reshape_rotary_for_broadcast` / :func:`apply_nd_rotary_embedding`
  — apply with optional KV-prefix ``start_offset``.
"""

from __future__ import annotations

import torch

from worldfoundry.core.attention.rotary.rope import apply_rotary_embedding


# ──────────────────────────────────────────────────────────────────────────
# Axis helpers — accept a scalar broadcast or an explicit per-axis tuple
# ──────────────────────────────────────────────────────────────────────────


def _to_tuple(value: int | tuple[int, ...] | list[int], *, dim: int) -> tuple[int, ...]:
    """Broadcast a scalar to ``dim`` axes, or validate an explicit tuple."""

    if isinstance(value, int):
        return (value,) * dim
    if len(value) == dim:
        return tuple(int(item) for item in value)
    raise ValueError(f"Expected length {dim} or int, but got {value}")


def _default_device() -> torch.device:
    """Prefer the current CUDA device so tables land next to activations."""

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
# Grid + apply — linspace(endpoint=False) so integer cells stay unshifted
# ──────────────────────────────────────────────────────────────────────────


def get_meshgrid_nd(
    start: int | tuple[int, ...] | list[int],
    *args: int | tuple[int, ...] | list[int],
    dim: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build an n-D meshgrid with PyTorch ``linspace(endpoint=False)`` semantics."""

    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start_tuple = (0,) * dim
        stop_tuple = num
    elif len(args) == 1:
        start_tuple = _to_tuple(start, dim=dim)
        stop_tuple = _to_tuple(args[0], dim=dim)
        num = tuple(stop_tuple[index] - start_tuple[index] for index in range(dim))
    elif len(args) == 2:
        start_tuple = _to_tuple(start, dim=dim)
        stop_tuple = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    resolved_device = torch.device(device) if device is not None else _default_device()
    axis_grid = []
    for index in range(dim):
        axis_start, axis_stop, axis_num = start_tuple[index], stop_tuple[index], num[index]
        grid = torch.linspace(
            axis_start,
            axis_stop,
            axis_num + 1,
            dtype=torch.float32,
            device=resolved_device,
        )[:axis_num]
        axis_grid.append(grid)
    return torch.stack(torch.meshgrid(*axis_grid, indexing="ij"), dim=0)


def reshape_rotary_for_broadcast(
    freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    value: torch.Tensor,
    *,
    head_first: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reshape RoPE frequencies for ``[B, S, H, D]`` or ``[B, H, S, D]`` tensors."""

    ndim = value.ndim
    if ndim < 3:
        raise ValueError(f"value must have at least 3 dimensions, got {value.shape}")

    seq_dim = -2 if head_first else 1
    expected_shape = (value.shape[seq_dim], value.shape[-1])
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        if cos.shape[-1] != expected_shape[-1] or cos.shape[0] < expected_shape[0]:
            raise ValueError(f"cos shape {cos.shape} cannot cover expected {expected_shape}")
        if sin.shape[-1] != expected_shape[-1] or sin.shape[0] < expected_shape[0]:
            raise ValueError(f"sin shape {sin.shape} cannot cover expected {expected_shape}")
        shape = [1] * ndim
        shape[seq_dim] = cos.shape[0]
        shape[-1] = value.shape[-1]
        return cos.view(*shape), sin.view(*shape)

    if freqs_cis.shape[-1] != expected_shape[-1] or freqs_cis.shape[0] < expected_shape[0]:
        raise ValueError(f"freqs_cis shape {freqs_cis.shape} cannot cover expected {expected_shape}")
    shape = [1] * ndim
    shape[seq_dim] = freqs_cis.shape[0]
    shape[-1] = value.shape[-1]
    return freqs_cis.view(*shape)


def apply_nd_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    head_first: bool = False,
    start_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed n-D rotary frequencies to query and key tensors.

    Args:
        query: Query tensor in ``[B, S, H, D]`` layout by default, or
            ``[B, H, S, D]`` when ``head_first=True``.
        key: Key tensor with the same layout convention and head width.
        freqs_cis: Complex rotary frequencies or an explicit ``(cos, sin)``
            pair covering the requested sequence window.
        head_first: Select ``[B, H, S, D]`` sequence-axis interpretation.
        start_offset: First frequency position, used with an existing KV prefix.

    Returns:
        Rotated ``(query, key)`` tensors with original dtype and shape.

    Raises:
        ValueError: Frequency tensors do not cover the requested shape.

    Notes:
        The final head width must be compatible with complex pairs. Build
        matching structured frequencies with ``get_nd_rotary_pos_embed``.
    """

    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_rotary_for_broadcast(freqs_cis, query, head_first=head_first)
        cos = cos.to(query.device)
        sin = sin.to(query.device)
        query_cos = _slice_freqs_for_value(cos, query, head_first=head_first, start_offset=start_offset)
        query_sin = _slice_freqs_for_value(sin, query, head_first=head_first, start_offset=start_offset)
        query_out = apply_rotary_embedding(query.float(), query_cos, query_sin, interleaved=True).type_as(query)
        key_cos = _slice_freqs_for_value(cos, key, head_first=head_first, start_offset=start_offset)
        key_sin = _slice_freqs_for_value(sin, key, head_first=head_first, start_offset=start_offset)
        key_out = apply_rotary_embedding(key.float(), key_cos, key_sin, interleaved=True).type_as(key)
        return query_out, key_out

    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    freqs = reshape_rotary_for_broadcast(freqs_cis, query_complex, head_first=head_first).to(query.device)
    query_freqs = _slice_freqs_for_value(freqs, query_complex, head_first=head_first, start_offset=start_offset)
    query_out = torch.view_as_real(query_complex * query_freqs).flatten(3).type_as(query)

    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    key_freqs = _slice_freqs_for_value(freqs, key_complex, head_first=head_first, start_offset=start_offset)
    key_out = torch.view_as_real(key_complex * key_freqs).flatten(3).type_as(key)
    return query_out, key_out


def _slice_freqs_for_value(
    freqs: torch.Tensor,
    value: torch.Tensor,
    *,
    head_first: bool,
    start_offset: int,
) -> torch.Tensor:
    """Slice a frequency table to the value's sequence window (KV prefix)."""

    seq_dim = -2 if head_first else 1
    if start_offset == 0 and freqs.shape[seq_dim] == value.shape[seq_dim]:
        return freqs
    slices = [slice(None)] * freqs.ndim
    slices[seq_dim] = slice(start_offset, start_offset + value.shape[seq_dim])
    return freqs[tuple(slices)]


# ──────────────────────────────────────────────────────────────────────────
# Table builders — one 1-D table per axis, concatenated on the head width
# ──────────────────────────────────────────────────────────────────────────


def get_nd_rotary_pos_embed(
    rope_dim_list: list[int],
    start: int | tuple[int, ...] | list[int],
    *args: int | tuple[int, ...] | list[int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float | list[float] = 1.0,
    interpolation_factor: float | list[float] = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build n-D RoPE frequencies for tokens with structured grid coordinates."""

    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list), device=device)
    theta_rescale_factor = _expand_factor(theta_rescale_factor, len(rope_dim_list), "theta_rescale_factor")
    interpolation_factor = _expand_factor(interpolation_factor, len(rope_dim_list), "interpolation_factor")

    embeddings = []
    for index, rope_dim in enumerate(rope_dim_list):
        embeddings.append(
            get_1d_rotary_pos_embed(
                rope_dim,
                grid[index].reshape(-1),
                theta,
                use_real=use_real,
                theta_rescale_factor=theta_rescale_factor[index],
                interpolation_factor=interpolation_factor[index],
            )
        )

    if use_real:
        return (
            torch.cat([embedding[0] for embedding in embeddings], dim=1),
            torch.cat([embedding[1] for embedding in embeddings], dim=1),
        )
    return torch.cat(embeddings, dim=1)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor | int,
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build one-dimensional rotary position frequencies."""

    if isinstance(pos, int):
        resolved_device = _default_device()
        pos = torch.arange(pos, device=resolved_device).float()
    else:
        resolved_device = pos.device

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=resolved_device)[: (dim // 2)].float() / dim))
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        return freqs.cos().repeat_interleave(2, dim=1), freqs.sin().repeat_interleave(2, dim=1)
    return torch.polar(torch.ones_like(freqs), freqs)


def _expand_factor(
    factor: float | list[float],
    length: int,
    name: str,
) -> list[float]:
    """Expand a scalar or length-1 list to one factor per RoPE axis."""

    if isinstance(factor, (int, float)):
        return [float(factor)] * length
    if len(factor) == 1:
        return [float(factor[0])] * length
    if len(factor) != length:
        raise ValueError(f"len({name}) should equal {length}, got {len(factor)}")
    return [float(item) for item in factor]


__all__ = [
    "apply_nd_rotary_embedding",
    "get_1d_rotary_pos_embed",
    "get_meshgrid_nd",
    "get_nd_rotary_pos_embed",
    "reshape_rotary_for_broadcast",
]
