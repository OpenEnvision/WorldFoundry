"""Projective camera transforms composed with rotary attention (PRoPE).

PRoPE lifts camera intrinsics and SE(3) into the rotary frame so a
camera-conditioned video DiT attends in a projective coordinate
system rather than a purely grid-index one. Invert helpers stay here
so geometry / pose code does not depend on a specific transformer.

The attention call itself still goes through
:func:`worldfoundry.core.attention.scaled_dot_product_attention` after
the projective rotate. This is not a replacement for 3D RoPE on
patch indices — models that need both compose them.

Not this module:
    Grid-index 3D / 2D / n-D RoPE lives in :mod:`.rope`, :mod:`.rope_2d`,
    :mod:`.rope_nd`. Pose I/O lives in :mod:`worldfoundry.core.geometry.pose`.
    The old ``prope_diagonal_mask`` / ``attn_mask`` hooks are
    intentionally not part of this interface.

Public surface:

- :func:`prope_dot_product_attention` — wrap SDPA with P/P_T/P_inv.
- :func:`invert_se3` / :func:`invert_k` / :func:`lift_k` — camera
  algebra used by pose adapters.
- :func:`make_rope3_coeffs` — optional 3-axis view-change RoPE on the
  tail of ``head_dim``.
"""

from functools import partial
from typing import Callable, List, Tuple

import torch
from einops import rearrange

from worldfoundry.core.attention import scaled_dot_product_attention

# ──────────────────────────────────────────────────────────────────────────
# Camera tiling — choose a head_dim subspace that does not fight x/y-RoPE
# ──────────────────────────────────────────────────────────────────────────

# Camera tiling layouts (consumed by ``_prepare_apply_fns``).
#
# * ``"full"`` -- legacy behaviour: tile every sub-frame matrix across the
#   WHOLE head_dim via ``_apply_tiled_projmat``. Overlaps x/y-RoPE.
# * ``"sf13"`` -- contiguous 2-camera layout for the ``prope_disable_t_rope``
#   path: subsample the 4 VAE sub-frame cameras to the 2nd and 4th (0-based
#   ``{1, 3}``) and write each as a contiguous ``dims_per_cam``-wide block at
#   the FRONT of head_dim (sub-frame 1 -> ``[0:16)``, sub-frame 3 ->
#   ``[16:32)``), each block = ``dims_per_cam // 4`` four-vectors all getting
#   that matrix. The tail is left identity. Under ``prope_disable_t_rope`` the
#   front of head_dim is rope-free, so the camera lives in a clean subspace
#   disjoint from x/y-RoPE (no conjugation entanglement).
PROPE_CAMERA_LAYOUTS = {
    "full": None,
    "sf13": {"subframes": (1, 3), "dims_per_cam": 16},
}


# ──────────────────────────────────────────────────────────────────────────
# Attention wrap — rotate Q/K/V, call exact SDPA, rotate output back
# ──────────────────────────────────────────────────────────────────────────


def prope_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    viewmats,
    view_change_positions=None,
    rope3_each=None,
    camera_layout="full",
    scale=None,
    **kwargs,
) -> torch.Tensor:
    """Apply PRoPE transforms around torch SDPA.

    `viewmats` is a tuple of `(P, P_T, P_inv)` with shape
    `(batch, cameras, 4, 4, 4)`. The old `prope_diagonal_mask` and
    `attn_mask` hooks are intentionally not part of this interface.
    `camera_layout` selects the head_dim tiling (see ``PROPE_CAMERA_LAYOUTS``).
    """
    batch, num_heads, seqlen, head_dim = q.shape
    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        view_change_positions=view_change_positions,
        rope3_each=rope3_each,
        camera_layout=camera_layout,
    )
    out = scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        scale=scale,
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,
    viewmats,
    view_change_positions=None,
    rope3_each=None,
    camera_layout="full",
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Build Q / KV / output applicators for one camera-layout choice.

    ``full`` tiles every sub-frame matrix across the whole head (legacy,
    overlaps axial RoPE). ``sf13`` writes two cameras into a rope-free
    prefix so conjugation does not entangle pose with x/y frequencies.
    Optional ``view_change_positions`` compose a 3-axis RoPE on the tail;
    the output path inverts that RoPE *before* the projective restore so
    the two frames do not cancel incorrectly.
    """

    P, P_T, P_inv = viewmats
    if camera_layout not in PROPE_CAMERA_LAYOUTS:
        raise ValueError(
            f"unknown prope camera_layout {camera_layout!r}; expected one of {sorted(PROPE_CAMERA_LAYOUTS)}."
        )
    spec = PROPE_CAMERA_LAYOUTS[camera_layout]
    if spec is None:
        transforms_q = [(partial(_apply_tiled_projmat, matrix=P_T), head_dim)]
        transforms_kv = [(partial(_apply_tiled_projmat, matrix=P_inv), head_dim)]
        transforms_o = [(partial(_apply_tiled_projmat, matrix=P), head_dim)]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
        apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    else:
        sub, dpc = spec["subframes"], spec["dims_per_cam"]
        apply_fn_q = partial(_apply_contiguous_projmat, matrix=P_T, subframes=sub, dims_per_cam=dpc)
        apply_fn_kv = partial(_apply_contiguous_projmat, matrix=P_inv, subframes=sub, dims_per_cam=dpc)
        apply_fn_o = partial(_apply_contiguous_projmat, matrix=P, subframes=sub, dims_per_cam=dpc)

    if view_change_positions is not None:
        each = rope3_each if rope3_each is not None else ROPE3_EACH
        coeffs3 = make_rope3_coeffs(view_change_positions, each, device=P.device)
        proj_q, proj_kv, proj_o = apply_fn_q, apply_fn_kv, apply_fn_o

        def apply_fn_q(x: torch.Tensor) -> torch.Tensor:
            """Project then apply view-change RoPE on the query subspace."""

            return _apply_rope3_slice(proj_q(x), coeffs3, each, inverse=False)

        def apply_fn_kv(x: torch.Tensor) -> torch.Tensor:
            """Project then apply view-change RoPE on the key/value subspace."""

            return _apply_rope3_slice(proj_kv(x), coeffs3, each, inverse=False)

        def apply_fn_o(x: torch.Tensor) -> torch.Tensor:
            """Undo view-change RoPE, then restore the camera frame on output."""

            return proj_o(_apply_rope3_slice(x, coeffs3, each, inverse=True))

    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply one 4×4 camera matrix to every four-vector across ``head_dim``.

    The reshape assumes ``head_dim`` factors as ``4*4*4*r`` so the same
    matrix hits every sub-frame channel. That is why ``full`` overlaps
    axial RoPE: there is no reserved prefix.
    """

    feat_pc = rearrange(
        feats,
        "b h (s t) (p q j r) -> b h s t r p q j",
        s=matrix.shape[1],
        p=4,
        q=4,
        j=4,
    )
    out_pc = torch.einsum("bscij,bhstrpcj->bhstrpci", matrix, feat_pc)
    return rearrange(out_pc, "b h s t r p q j -> b h (s t) (p q j r)")


def _apply_contiguous_projmat(
    feats: torch.Tensor,
    matrix: torch.Tensor,
    subframes: Tuple[int, ...],
    dims_per_cam: int = 16,
) -> torch.Tensor:
    """Contiguous per-sub-frame PRoPE tiling (see ``PROPE_CAMERA_LAYOUTS``).

    ``matrix`` is ``(batch, frames, sub_frames, 4, 4)``. Each selected
    sub-frame's matrix is written into its own contiguous block at the front
    of head_dim: sub-frame ``subframes[k]`` -> dims
    ``[k*dims_per_cam : (k+1)*dims_per_cam)``, holding ``dims_per_cam // 4``
    four-vectors that all get that matrix. Tail dims pass through unchanged.
    """
    if dims_per_cam % 4 != 0:
        raise ValueError(f"dims_per_cam must be a multiple of 4, got {dims_per_cam}.")
    n = len(subframes)
    cam_dims = n * dims_per_cam
    head_dim = feats.shape[-1]
    if cam_dims > head_dim:
        raise ValueError(f"contiguous camera needs {cam_dims} dims but head_dim={head_dim}.")
    num_sub = matrix.shape[2]
    if max(subframes) >= num_sub:
        raise ValueError(f"sub-frame index {max(subframes)} out of range; matrix has {num_sub} sub-frames.")
    fourvec = dims_per_cam // 4
    cam, rest = feats[..., :cam_dims], feats[..., cam_dims:]
    cam = rearrange(
        cam,
        "b h (s t) (n f j) -> b h s t n f j",
        s=matrix.shape[1],
        n=n,
        f=fourvec,
        j=4,
    )
    idx = torch.as_tensor(subframes, device=matrix.device, dtype=torch.long)
    m_sel = matrix.index_select(2, idx)  # (b, s, n, 4, 4)
    out = torch.einsum("bsnij,bhstnfj->bhstnfi", m_sel, cam)
    out = rearrange(out, "b h s t n f j -> b h (s t) (n f j)")
    return torch.cat([out, rest], dim=-1)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Split ``head_dim`` into named blocks and apply one transform per block.

    Invariant: the sum of block sizes equals the last dimension; a
    mismatch is an assertion because it means the layout spec and the
    checkpoint head width have drifted.
    """

    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([func(x_block) for func, x_block in zip(funcs, x_blocks)], dim=-1)
    assert out.shape == feats.shape
    return out


# ──────────────────────────────────────────────────────────────────────────
# Camera algebra — invert / lift without importing a geometry package
# ──────────────────────────────────────────────────────────────────────────


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert an SE(3) matrix as ``R^T`` and ``-R^T t`` (last row stays 0 0 0 1).

    A generic ``torch.linalg.inv`` would be slower and can break the
    homogeneous row on half-precision pose tensors.
    """

    assert transforms.shape[-2:] == (4, 4)
    r_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", r_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    """Embed 3×3 intrinsics into 4×4 so they compose with SE(3) viewmats."""

    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros(
        intrinsics.shape[:-2] + (4, 4),
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    out[..., :3, :3] = intrinsics
    out[..., 3, 3] = 1.0
    return out


ROPE3_EACH = 8


# ──────────────────────────────────────────────────────────────────────────
# View-change RoPE — log-scale + two linear axes on the head_dim tail
# ──────────────────────────────────────────────────────────────────────────


def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim, device):
    """Precompute ``(cos, sin)`` tables for one axis of view-change RoPE."""

    if positions.dim() == 1:
        positions = positions[None]
    num_freqs = feat_dim // 2
    freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=device, dtype=positions.dtype) / num_freqs))
    angles = positions[:, None, :, None] * freqs[None, None, None, :]
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(feats, coeffs, inverse=False):
    """Apply or invert a 2-D rotation stored as broadcast ``(cos, sin)``.

    Sequence length may be an integer multiple of the coeff length
    (sub-sampled view changes); a non-divisible pair is a layout bug
    and raises ``ValueError``.
    """

    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        if feats.shape[2] % cos.shape[2] != 0:
            raise ValueError(
                "view_change_positions length must divide feature sequence length, "
                f"got coeffs={cos.shape[2]} feats={feats.shape[2]}."
            )
        n = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n, 1)
        sin = sin.repeat(1, 1, n, 1)
    half = feats.shape[-1] // 2
    x_in, y_in = feats[..., :half], feats[..., half:]
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def make_rope3_coeffs(view_change_positions, rope_each, device, freq_base=100.0):
    """Build three RoPE coeff tables from ``(scale, y, x)`` view deltas.

    Scale is logged so a multiplicative zoom stays additive in the
    rotary plane. Near-zero scale is clamped to avoid ``-inf``.
    """

    p = view_change_positions.to(device)
    pos = torch.stack([torch.log(p[..., 0].clamp_min(1e-6)), p[..., 1], p[..., 2]], dim=-1)
    return [_rope_precompute_coeffs(pos[..., i], freq_base, 1.0, rope_each, device) for i in range(3)]


def _apply_rope3_slice(feats, coeffs3, rope_each, inverse=False):
    """Rotate the last ``3 * rope_each`` channels; leave the prefix untouched.

    The tail is the rope-free camera subspace under ``sf13`` +
    ``prope_disable_t_rope``. A head narrower than the slice is a
    checkpoint mismatch, not a silent truncate.
    """

    if coeffs3 is None:
        return feats
    n = 3 * rope_each
    if feats.shape[-1] < n:
        raise ValueError(f"head_dim={feats.shape[-1]} is too small for 3VAL RoPE slice {n}.")
    head, tail = feats[..., : feats.shape[-1] - n], feats[..., feats.shape[-1] - n :]
    blocks = list(tail.split(rope_each, dim=-1))
    rolled = [_rope_apply_coeffs(blocks[i], coeffs3[i], inverse=inverse) for i in range(3)]
    return torch.cat([head] + rolled, dim=-1)


def invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    """Invert pinhole intrinsics in closed form (fx, fy, cx, cy only).

    A generic inverse would mix skew / unused rows that official
    recipes keep as identity.
    """

    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros_like(intrinsics)
    out[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    out[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    out[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    out[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


invert_camera_intrinsics = invert_k
lift_camera_intrinsics = lift_k


__all__ = [
    "PROPE_CAMERA_LAYOUTS",
    "invert_camera_intrinsics",
    "invert_k",
    "invert_se3",
    "lift_camera_intrinsics",
    "lift_k",
    "make_rope3_coeffs",
    "prope_dot_product_attention",
]
