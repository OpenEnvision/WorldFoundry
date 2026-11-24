# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# This file is adapted from the official PRoPE reference implementation:
# - https://www.liruilong.cn/prope/
# - https://github.com/liruilong940607/prope
#
# Modifications for this repo:
# - Support per-sample `image_hw` (different H/W inside the same batch).
# - Expose lightweight helpers that return Q/KV/O transform callables so we can
#   plug PRoPE into existing attention backends (e.g., xFormers).

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.prope import _lift_K
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.prope import (
    _apply_block_diagonal,
    _apply_tiled_projmat,
    _invert_K,
    _invert_SE3,
    _rope_precompute_coeffs,
)


def get_rope_coeffs_2d(
    *,
    patches_x: int,
    patches_y: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    freq_base: float = 100.0,
    freq_scale: float = 1.0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Precompute RoPE coeffs for a single image (row-major patches).

    Returns:
        coeffs_x, coeffs_y
        Each is a tuple (cos, sin) with shape: (1, 1, seqlen_per_image, feat_dim//2)
    """
    assert head_dim % 4 == 0
    feat_dim = head_dim // 4
    # Positions for a single image in row-major order.
    pos_x = torch.tile(torch.arange(patches_x, device=device), (patches_y,))
    pos_y = torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x)
    coeffs_x = _rope_precompute_coeffs(pos_x, freq_base=freq_base, freq_scale=freq_scale, feat_dim=feat_dim)
    coeffs_y = _rope_precompute_coeffs(pos_y, freq_base=freq_base, freq_scale=freq_scale, feat_dim=feat_dim)
    # Cast for runtime dtype (usually fp16/bf16/fp32).
    coeffs_x = (coeffs_x[0].to(dtype=dtype), coeffs_x[1].to(dtype=dtype))
    coeffs_y = (coeffs_y[0].to(dtype=dtype), coeffs_y[1].to(dtype=dtype))
    return coeffs_x, coeffs_y


def prepare_prope_apply_fns(
    *,
    head_dim: int,
    viewmats: torch.Tensor,  # (B, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (B, cameras, 3, 3) or None
    patches_x: int,
    patches_y: int,
    image_hw: torch.Tensor,  # (B, 2) with [H, W] in pixels
    coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
    coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare PRoPE transforms for Q / KV / O.

    The returned callables accept tensors with shape:
      (B, num_heads, seqlen, head_dim)

    Constraints:
    - seqlen must equal cameras * patches_x * patches_y
    - token ordering must be camera-major, then row-major within each camera
    """
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert image_hw.shape[0] == batch and image_hw.shape[-1] == 2
    assert head_dim % 4 == 0

    # Normalize camera intrinsics (per-sample image sizes).
    if Ks is not None:
        # image_hw is [H, W]
        image_h = image_hw[:, 0].to(device=device, dtype=Ks.dtype).unsqueeze(1)  # (B,1)
        image_w = image_hw[:, 1].to(device=device, dtype=Ks.dtype).unsqueeze(1)  # (B,1)

        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_w
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_h
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_w - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_h - 0.5
        Ks_norm[..., 2, 2] = 1.0

        # PRoPE projection matrices.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )
    else:
        # GTA formula.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    # RoPE coefficients (single-image), repeated across cameras at application time.
    if coeffs_x is None or coeffs_y is None:
        # Use torch.float32 to compute, then cast inside get_rope_coeffs_2d
        coeffs_x, coeffs_y = get_rope_coeffs_2d(
            patches_x=patches_x,
            patches_y=patches_y,
            head_dim=head_dim,
            device=device,
            dtype=viewmats.dtype,
        )

    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def reorder_tokens_to_camera_major(
    x: torch.Tensor,  # (B, num_heads, seqlen, head_dim) or (B, seqlen, num_heads, head_dim)
    *,
    cameras: int,
    patches_y: int,
    patches_x_total: int,
    is_bnhd: bool,
) -> torch.Tensor:
    """Reorder tokens from 'merged-width row-major' to 'camera-major row-major'.

    Input corresponds to patch tokens for a single image of width `patches_x_total`,
    where that width is actually `cameras * patches_x_per_cam`.

    This matches the existing PixArtWorldFM tri-condition behavior (concatenating on width),
    and converts it into PRoPE's expected ordering.
    """
    assert patches_x_total % cameras == 0
    patches_x_per_cam = patches_x_total // cameras

    if is_bnhd:
        # (B, num_heads, seqlen, head_dim)
        B, Hh, N, D = x.shape
        assert patches_y * patches_x_total == N
        x = x.view(B, Hh, patches_y, patches_x_total, D)
        x = x.view(B, Hh, patches_y, cameras, patches_x_per_cam, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, heads, cam, y, x, D
        return x.view(B, Hh, cameras * patches_y * patches_x_per_cam, D)
    else:
        # (B, seqlen, num_heads, head_dim)
        B, N, Hh, D = x.shape
        assert patches_y * patches_x_total == N
        x = x.view(B, patches_y, patches_x_total, Hh, D)
        x = x.view(B, patches_y, cameras, patches_x_per_cam, Hh, D)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()  # B, cam, y, x, heads, D
        return x.view(B, cameras * patches_y * patches_x_per_cam, Hh, D)


def reorder_tokens_from_camera_major(
    x: torch.Tensor,
    *,
    cameras: int,
    patches_y: int,
    patches_x_total: int,
    is_bnhd: bool,
) -> torch.Tensor:
    """Inverse of reorder_tokens_to_camera_major."""
    assert patches_x_total % cameras == 0
    patches_x_per_cam = patches_x_total // cameras

    if is_bnhd:
        B, Hh, N, D = x.shape
        assert cameras * patches_y * patches_x_per_cam == N
        x = x.view(B, Hh, cameras, patches_y, patches_x_per_cam, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, heads, y, cam, x, D
        x = x.view(B, Hh, patches_y, patches_x_total, D)
        return x.view(B, Hh, patches_y * patches_x_total, D)
    else:
        B, N, Hh, D = x.shape
        assert cameras * patches_y * patches_x_per_cam == N
        x = x.view(B, cameras, patches_y, patches_x_per_cam, Hh, D)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()  # B, y, cam, x, heads, D
        x = x.view(B, patches_y, patches_x_total, Hh, D)
        return x.view(B, patches_y * patches_x_total, Hh, D)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (B, num_heads, seqlen, feat_dim)
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = coeffs
    # Allow (cos,sin) for a single image and repeat to match (camera-major) seqlen.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )
