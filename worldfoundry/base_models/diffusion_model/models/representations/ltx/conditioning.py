# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""LTX image content and generated-keyframe slots on canonical modality states.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Appended slots are removed before spatial upsampling
and media decoding; only the stage-one transformer generates their content.
"""

from __future__ import annotations

import torch

from ....contracts import ModalityState
from .patchifiers import get_pixel_coords
from .tools import VideoLatentTools
from .types import VideoLatentShape


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend LTX latents while retaining the denoised product's input-precision rounding."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def inject_ltx_image(
    state: ModalityState,
    tools: VideoLatentTools,
    encoded: torch.Tensor,
    *,
    frame_index: int,
    strength: float,
) -> ModalityState:
    """Replace the first clean latent frame or append a later image as reference tokens."""
    tokens = tools.patchifier.patchify(encoded)
    if frame_index == 0:
        result = state.clone()
        result.clean_latent[:, : tokens.shape[1]] = tokens
        result.denoise_mask[:, : tokens.shape[1]] = 1.0 - strength
        return result
    coordinates = tools.patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape.from_torch_shape(encoded.shape),
        device=encoded.device,
    )
    positions = get_pixel_coords(coordinates, tools.scale_factors, causal_fix=False)
    positions[:, 0] += frame_index
    positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
    positions = positions.float()
    positions[:, 0] /= tools.fps
    mask = torch.full((*tokens.shape[:2], 1), 1.0 - strength, device=encoded.device, dtype=encoded.dtype)
    return state.with_updates(
        latent=torch.cat([state.latent, torch.zeros_like(tokens)], dim=1),
        clean_latent=torch.cat([state.clean_latent, tokens], dim=1),
        denoise_mask=torch.cat([state.denoise_mask, mask], dim=1),
        positions=torch.cat([state.positions, positions], dim=2),
        keyframes_mask=(
            torch.cat([state.keyframes_mask, torch.zeros_like(mask)], dim=1)
            if state.keyframes_mask is not None
            else None
        ),
    )


def append_ltx_keyframe_slots(
    state: ModalityState, tools: VideoLatentTools, *, count: int, num_frames: int
) -> ModalityState:
    """Append evenly spaced, fully denoised single-frame slots marked for the DiT."""
    if count < 0 or num_frames < count + 2:
        raise ValueError("LTX generated keyframes require a non-negative count and at least count + 2 frames")
    if count == 0:
        return state
    frame_indices = torch.linspace(0, num_frames - 1, count + 2).round().to(torch.int64).tolist()[1:-1]
    frame_shape = tools.target_shape._replace(frames=1)
    coordinates = tools.patchifier.get_patch_grid_bounds(output_shape=frame_shape, device=state.latent.device)
    positions = []
    for frame_index in frame_indices:
        position = get_pixel_coords(coordinates, tools.scale_factors, causal_fix=False).float()
        position[:, 0] = float(frame_index)
        position[:, 0, ..., 1:] = position[:, 0, ..., :1] + 1
        position[:, 0] /= tools.fps
        positions.append(position)
    tokens = state.latent.new_zeros(
        state.latent.shape[0],
        tools.patchifier.get_token_count(frame_shape) * count,
        state.latent.shape[2],
    )
    mask = state.denoise_mask.new_ones((*tokens.shape[:2], 1))
    return state.with_updates(
        latent=torch.cat([state.latent, tokens], dim=1),
        clean_latent=torch.cat([state.clean_latent, torch.zeros_like(tokens)], dim=1),
        denoise_mask=torch.cat([state.denoise_mask, mask], dim=1),
        positions=torch.cat([state.positions, *positions], dim=2),
        keyframes_mask=torch.cat(
            [
                state.keyframes_mask if state.keyframes_mask is not None else torch.zeros_like(state.denoise_mask),
                mask,
            ],
            dim=1,
        ),
    )


def add_ltx_noise(state: ModalityState, *, scale: float, generator: torch.Generator) -> ModalityState:
    """Mix initial latent with Gaussian noise, then project onto the conditioning content."""
    noise = torch.randn(state.latent.shape, generator=generator, device=state.latent.device, dtype=state.latent.dtype)
    latent = torch.lerp(state.latent.float(), noise.float(), scale)
    latent = torch.lerp(state.clean_latent.float(), latent, state.denoise_mask)
    return state.with_updates(latent=latent.to(state.latent.dtype))


def append_ltx_reference_video(
    state: ModalityState,
    tools: VideoLatentTools,
    encoded: torch.Tensor,
    *,
    strength: float,
    downscale_factor: int = 1,
    temporal_scale_factor: int = 1,
) -> ModalityState:
    """Append IC-LoRA video tokens to an unmasked target with aligned space/time coordinates."""
    tokens = tools.patchifier.patchify(encoded)
    coordinates = tools.patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape.from_torch_shape(encoded.shape), device=encoded.device
    )
    positions = get_pixel_coords(coordinates, tools.scale_factors, causal_fix=tools.causal_fix).float()
    positions[:, 0] /= tools.fps / temporal_scale_factor
    if temporal_scale_factor != 1:
        target_start = state.positions[:, 0, 0:1, 1:2].float()
        positions[:, 0] = torch.clamp(positions[:, 0] - (temporal_scale_factor - 1) * target_start, min=0)
    if downscale_factor != 1:
        positions[:, 1] *= downscale_factor
        positions[:, 2] *= downscale_factor
    mask = torch.full((*tokens.shape[:2], 1), 1.0 - strength, device=encoded.device, dtype=encoded.dtype)
    return state.with_updates(
        latent=torch.cat([state.latent, torch.zeros_like(tokens)], dim=1),
        clean_latent=torch.cat([state.clean_latent, tokens], dim=1),
        denoise_mask=torch.cat([state.denoise_mask, mask], dim=1),
        positions=torch.cat([state.positions, positions], dim=2),
        keyframes_mask=(
            torch.cat([state.keyframes_mask, torch.zeros_like(mask)], dim=1)
            if state.keyframes_mask is not None
            else None
        ),
    )
