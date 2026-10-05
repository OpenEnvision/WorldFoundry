# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Multiscale UCPE geometry; local control poses and global memory poses stay separate."""

from __future__ import annotations
from typing import Any
import torch
from einops import rearrange, repeat
from . import geometry as ucpe_cc
from ..ucpe import prope as prope_torch


def rope_axis_positions(
    num_patches: int,
    reference_num_patches: int | None = None,
    scale_invariant: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """PRoPE positions along one axis of a token grid.

    With ``scale_invariant`` set, positions are expressed on the reference grid
    so a physical location keeps the same angle at every pyramid level:

        pos(m) = (m + 0.5) * (W_ref / W_s) - 0.5

    Position ``m`` then sits at the centre of the reference columns it covers.
    At ``W_s == W_ref``, the expression reduces to ``arange(W)``.
    """
    base = torch.arange(num_patches, device=device, dtype=torch.float32)
    if not scale_invariant:
        return base
    scale = (reference_num_patches or num_patches) / num_patches
    return (base + 0.5) * scale - 0.5


def _build_stage_rope_coeffs(
    cameras: int,
    patches_y: int,
    patches_x: int,
    head_dim: int,
    freq_base: float,
    freq_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    reference_patches_y: int | None = None,
    reference_patches_x: int | None = None,
    scale_invariant_positions: bool = True,
):
    """PRoPE cos/sin coefficients for one token grid."""
    base_x = rope_axis_positions(
        patches_x, reference_patches_x, scale_invariant_positions, device
    )
    base_y = rope_axis_positions(
        patches_y, reference_patches_y, scale_invariant_positions, device
    )

    x_positions = torch.tile(base_x, (patches_y * cameras,))
    y_positions = torch.tile(torch.repeat_interleave(base_y, patches_x), (cameras,))

    coeffs_x = prope_torch._rope_precompute_coeffs(
        x_positions,
        freq_base=freq_base,
        freq_scale=freq_scale,
        feat_dim=head_dim // 4,
        dtype=dtype,
    )
    coeffs_y = prope_torch._rope_precompute_coeffs(
        y_positions,
        freq_base=freq_base,
        freq_scale=freq_scale,
        feat_dim=head_dim // 4,
        dtype=dtype,
    )
    return coeffs_x, coeffs_y


def _slice_pose_chunk(
    pose: torch.Tensor,
    chunk_index: int,
    window_num_frames: int,
    restart_each_chunk: bool = True,
):
    if restart_each_chunk:
        start, end = 0, window_num_frames
    else:
        start = chunk_index * window_num_frames
        end = start + window_num_frames
    if pose.shape[1] < end:
        return None
    return pose[:, start:end]


def resolve_pose_chunk(
    camera_trajectory: dict[str, Any],
    num_latent_frames_per_chunk: int,
    chunk_index: int,
    vae_scale_factor_temporal: int = 4,
    restart_each_chunk: bool = True,
    translation_scale: float = 1.0,
    pose_is_chunk_aligned: bool = False,
):
    """Extract the ``[B, T, 3, 4]`` camera-to-world poses for one chunk.

    Returns ``(pose_chunk, x_fov, xi)`` or ``None`` when the trajectory is too
    short for the requested chunk.
    """
    pose = camera_trajectory["pose"]
    x_fov = camera_trajectory["x_fov"]
    xi = camera_trajectory["xi"]

    if pose.ndim == 3:
        pose = pose.unsqueeze(0)
    if torch.is_tensor(x_fov) and x_fov.ndim == 0:
        x_fov = x_fov.unsqueeze(0)
    if torch.is_tensor(xi) and xi.ndim == 0:
        xi = xi.unsqueeze(0)

    if pose_is_chunk_aligned:
        if pose.shape[1] < num_latent_frames_per_chunk:
            return None
        pose_chunk = pose[:, :num_latent_frames_per_chunk]
    else:
        window_num_frames = (
            num_latent_frames_per_chunk - 1
        ) * vae_scale_factor_temporal + 1
        pose_chunk = _slice_pose_chunk(
            pose, chunk_index, window_num_frames, restart_each_chunk=restart_each_chunk
        )
        if pose_chunk is None:
            return None
        pose_chunk = pose_chunk[:, ::vae_scale_factor_temporal]

    if pose_chunk.shape[-2:] == (4, 4):
        pose_chunk = pose_chunk[..., :3, :4]
    elif pose_chunk.shape[-2:] != (3, 4):
        raise ValueError(
            f"pose_chunk is expected to be [B, T, 3, 4] or [B, T, 4, 4], got shape={tuple(pose_chunk.shape)}"
        )

    if translation_scale != 1.0:
        pose_chunk = pose_chunk.clone()
        pose_chunk[..., 3] *= pose_chunk.new_tensor(float(translation_scale))

    return pose_chunk, x_fov, xi


def build_ucpe_camera_input(
    transformer,
    pose_chunk: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    grid_h: int,
    grid_w: int,
    reference_grid_h: int | None = None,
    reference_grid_w: int | None = None,
    pixel_center: bool = False,
    scale_invariant_positions: bool = True,
    force_explicit_coeffs: bool = False,
):
    """Build one ``control_camera_dit_input`` dict for a single token grid."""
    method = transformer.camera_condition
    if "gta" in method or "prope" in method:
        raise NotImplementedError(
            "The UCPE bridge implements relray_absmap only."
        )
    if "relray" not in method:
        raise ValueError(f"Unsupported camera condition: {method}")

    attn = transformer.blocks[0].cam_self_attn
    reference_grid_h = reference_grid_h or attn.patches_y
    reference_grid_w = reference_grid_w or attn.patches_x

    if grid_h * reference_grid_w != grid_w * reference_grid_h:
        raise ValueError(
            "UCPE pyramid levels must share an aspect ratio, otherwise fy=fx with cy=H/2 makes the "
            f"vertical FOV differ per level. Got level grid {grid_h}x{grid_w} against reference "
            f"{reference_grid_h}x{reference_grid_w}."
        )

    c2w = torch.eye(4, device=pose_chunk.device, dtype=pose_chunk.dtype)
    c2w = repeat(
        c2w, "... -> B T ...", B=pose_chunk.shape[0], T=pose_chunk.shape[1]
    ).clone()
    c2w[..., :3, :4] = pose_chunk

    d_cam = ucpe_cc.ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=grid_h,
        width=grid_w,
        device=pose_chunk.device,
        dtype=pose_chunk.dtype,
        pixel_center=pixel_center,
    )
    raymats = ucpe_cc.world_to_ray_mats(d_cam, c2w)
    viewmats = rearrange(raymats, "B T H W ... -> B (T H W) ...")

    camera_input: dict[str, Any] = {
        "viewmats": viewmats,
        "patches_y": grid_h,
        "patches_x": grid_w,
    }

    needs_coeffs = (
        force_explicit_coeffs or grid_h != attn.patches_y or grid_w != attn.patches_x
    )
    if needs_coeffs:
        coeffs_x, coeffs_y = _build_stage_rope_coeffs(
            cameras=pose_chunk.shape[1],
            patches_y=grid_h,
            patches_x=grid_w,
            head_dim=attn.head_dim,
            freq_base=attn.freq_base,
            freq_scale=attn.freq_scale,
            device=pose_chunk.device,
            dtype=pose_chunk.dtype,
            reference_patches_y=reference_grid_h,
            reference_patches_x=reference_grid_w,
            scale_invariant_positions=scale_invariant_positions,
        )
        camera_input["coeffs_x"] = coeffs_x
        camera_input["coeffs_y"] = coeffs_y

    if "absmap" in method:
        up_map, lat_map = ucpe_cc.compute_up_lat_map(
            R=pose_chunk[..., :3, :3],
            x_fov=x_fov,
            xi=xi,
            width=grid_w,
            height=grid_h,
            device=pose_chunk.device,
            pixel_center=pixel_center,
        )
        cam_emb = torch.cat([up_map, lat_map], dim=-1)
        camera_input["cam_emb"] = rearrange(cam_emb, "B T H W C -> B (T H W) C")

    expected_token_count = pose_chunk.shape[1] * grid_h * grid_w
    if camera_input["viewmats"].shape[1] != expected_token_count:
        raise ValueError(
            f"camera viewmats token count mismatch: expected {expected_token_count}, "
            f"got {camera_input['viewmats'].shape[1]}"
        )
    if (
        "cam_emb" in camera_input
        and camera_input["cam_emb"].shape[1] != expected_token_count
    ):
        raise ValueError(
            f"camera cam_emb token count mismatch: expected {expected_token_count}, "
            f"got {camera_input['cam_emb'].shape[1]}"
        )

    return camera_input


def pyramid_token_grids(
    reference_grid_h: int,
    reference_grid_w: int,
    num_stages: int,
    low_to_high: bool = False,
) -> list[tuple[int, int]]:
    """Token grids for each pyramid level, finest first by default.

    Stage-2 halves the spatial latent size per level, which halves the post-patch
    token grid too. Levels that would not halve evenly are rejected, since an
    inconsistent aspect ratio changes the vertical FOV.

    Set ``low_to_high`` for the coarsest-first order used by fast sampling.
    """
    grids = [(reference_grid_h, reference_grid_w)]
    for level in range(1, num_stages):
        prev_h, prev_w = grids[-1]
        if prev_h % 2 or prev_w % 2:
            raise ValueError(
                f"Cannot build {num_stages} pyramid levels from a {reference_grid_h}x{reference_grid_w} token "
                f"grid: level {level} would halve {prev_h}x{prev_w} unevenly, which changes the aspect ratio "
                "and therefore the vertical FOV."
            )
        grids.append((prev_h // 2, prev_w // 2))
    return list(reversed(grids)) if low_to_high else grids


def build_ucpe_attention_kwargs_pyramid(
    transformer,
    camera_trajectory: dict[str, Any] | None,
    num_latent_frames_per_chunk: int,
    chunk_index: int,
    token_grids: list[tuple[int, int]],
    vae_scale_factor_temporal: int = 4,
    restart_each_chunk: bool = True,
    translation_scale: float = 1.0,
    pose_is_chunk_aligned: bool = False,
    pixel_center: bool = True,
    scale_invariant_positions: bool = True,
):
    """Build per-level UCPE inputs in coarsest-first order.

    The returned list follows ``token_grids``. Every level uses the finest grid
    as its common geometric reference. Sampling at pixel centers keeps the
    frustum symmetric, including at the coarsest resolution.
    """
    if (
        camera_trajectory is None
        or getattr(transformer, "camera_condition", "none") == "none"
    ):
        return None

    resolved = resolve_pose_chunk(
        camera_trajectory,
        num_latent_frames_per_chunk=num_latent_frames_per_chunk,
        chunk_index=chunk_index,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        restart_each_chunk=restart_each_chunk,
        translation_scale=translation_scale,
        pose_is_chunk_aligned=pose_is_chunk_aligned,
    )
    if resolved is None:
        return None
    pose_chunk, x_fov, xi = resolved

    reference_grid_h, reference_grid_w = max(
        token_grids, key=lambda grid: grid[0] * grid[1]
    )
    camera_inputs = [
        build_ucpe_camera_input(
            transformer,
            pose_chunk=pose_chunk,
            x_fov=x_fov,
            xi=xi,
            grid_h=grid_h,
            grid_w=grid_w,
            reference_grid_h=reference_grid_h,
            reference_grid_w=reference_grid_w,
            pixel_center=pixel_center,
            scale_invariant_positions=scale_invariant_positions,
            force_explicit_coeffs=True,
        )
        for grid_h, grid_w in token_grids
    ]

    return {"camera_control_ucpe_input_list": camera_inputs}


def build_ucpe_attention_kwargs_sequential_pyramid(
    transformer,
    camera_trajectory: dict[str, Any] | None,
    num_latent_frames_per_chunk: int,
    chunk_index: int,
    token_grids: list[tuple[int, int]],
    **kwargs,
):
    """One single-camera kwargs dict per pyramid level, coarsest first.

    Fast sampling denoises levels sequentially. Each forward receives the
    camera input for its resolution, with the finest grid as the shared
    geometric reference.
    """
    pyramid_kwargs = build_ucpe_attention_kwargs_pyramid(
        transformer,
        camera_trajectory,
        num_latent_frames_per_chunk=num_latent_frames_per_chunk,
        chunk_index=chunk_index,
        token_grids=token_grids,
        **kwargs,
    )
    if pyramid_kwargs is None:
        return None
    return [
        {"camera_control_ucpe_input": camera_input}
        for camera_input in pyramid_kwargs["camera_control_ucpe_input_list"]
    ]
