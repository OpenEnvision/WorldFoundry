"""Projection composition utilities.

Merges multiple event (foreground) projections into a single fg_proj tensor.

The scene_proj and fg_proj are kept separate and passed to UnifiedBackbonePipeline's
_prepare_sp_context(), which handles the channel concatenation:
- scene_proj: [C, T, h, w] (16 channels) -> target_scene_proj
- fg_proj: [C, T, h, w] (16 channels) -> target_fg_proj
- State Adapter context: [2C, T+P, h, w] (32 channels) after internal concatenation
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .io_utils import decode_latent_to_rgb


def merge_event_projections(
    event_projs: list[torch.Tensor],
    vae=None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Merge multiple event projections into a single fg_proj.

    When multiple projections exist, merges in pixel space to avoid
    latent-space artifacts (decode each to RGB, overlay, re-encode).

    Args:
        event_projs: List of event projection latents, each [C, T, h, w].
        vae: VAE wrapper (required when len(event_projs) > 1).
        device: Torch device (required when len(event_projs) > 1).
        dtype: Torch dtype (required when len(event_projs) > 1).

    Returns:
        Merged fg_proj [C, T, h, w], or None if no events.
    """
    if not event_projs:
        return None

    if len(event_projs) == 1:
        return event_projs[0]

    # Multiple projections: merge in pixel space to avoid latent artifacts.
    rgb_layers = [decode_latent_to_rgb(ep, vae, device, dtype) for ep in event_projs]

    # Overlay: later non-black pixels overwrite earlier ones.
    merged_rgb = rgb_layers[0].copy()
    for layer in rgb_layers[1:]:
        fg_mask = (layer > 10).any(axis=-1, keepdims=True)  # [T, H, W, 1]
        merged_rgb = np.where(fg_mask, layer, merged_rgb)

    # Re-encode merged RGB to latent.
    combined_tensor = (
        torch.from_numpy(
            merged_rgb.transpose(0, 3, 1, 2)  # [T, 3, H, W]
        ).float()
        / 127.5
        - 1.0
    )

    with torch.no_grad():
        vae_device = next(vae.model.parameters()).device
        if vae_device != device:
            vae.model.to(device)
            vae.mean = vae.mean.to(device)
            vae.std = vae.std.to(device)
        combined_tensor = combined_tensor.to(device=device, dtype=dtype)
        combined_tensor = combined_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
        latent = vae.encode_to_latent(combined_tensor)  # [1, C, T', h, w]
        latent = latent.squeeze(0).permute(1, 0, 2, 3)  # [C, T', h, w]

    return latent


def overlay_fg_on_scene(
    scene_proj: torch.Tensor,
    fg_proj: torch.Tensor,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, np.ndarray]:
    """Overlay fg projection onto scene projection in pixel space.

    Computes filled-contour mask of fg to black out the scene in that region,
    then pastes fg pixels on top. Returns both the re-encoded latent and the
    pixel-space composite (to avoid decode artifacts when visualizing).

    Args:
        scene_proj: Scene projection latent [C, T, h, w].
        fg_proj: Foreground projection latent [C, T, h, w].
        vae: VAE wrapper with encode/decode methods.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple of:
        - combined latent [C, T, h, w]
        - combined_rgb [T, H, W, 3] uint8 (pixel-space, use for visualization)
    """
    # Decode both to RGB [T, H, W, 3] uint8.
    scene_rgb = decode_latent_to_rgb(scene_proj, vae, device, dtype)
    fg_rgb = decode_latent_to_rgb(fg_proj, vae, device, dtype)

    # Build filled-contour mask: find outermost contours of fg pixels per frame,
    # fill interior (including holes), then black out scene inside this region
    # before pasting fg on top.
    fg_pixel_mask = (fg_rgb > 10).any(axis=-1)  # [T, H, W] bool
    filled_mask = np.zeros_like(fg_pixel_mask, dtype=np.uint8)  # [T, H, W]
    for t in range(fg_pixel_mask.shape[0]):
        binary = fg_pixel_mask[t].astype(np.uint8) * 255
        ret = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = ret[-2]
        cv2.drawContours(filled_mask[t], contours, -1, 1, thickness=cv2.FILLED)
    filled_mask_3ch = filled_mask[..., None]  # [T, H, W, 1]

    # Black out scene where fg contour covers, then paste fg pixels.
    scene_rgb = np.where(filled_mask_3ch, 0, scene_rgb)
    fg_nonblack = (fg_rgb > 10).any(axis=-1, keepdims=True)  # [T, H, W, 1]
    combined_rgb = np.where(fg_nonblack, fg_rgb, scene_rgb)

    # Re-encode to latent.
    combined_tensor = (
        torch.from_numpy(
            combined_rgb.transpose(0, 3, 1, 2)  # [T, 3, H, W]
        ).float()
        / 127.5
        - 1.0
    )

    with torch.no_grad():
        vae_device = next(vae.model.parameters()).device
        if vae_device != device:
            vae.model.to(device)
            vae.mean = vae.mean.to(device)
            vae.std = vae.std.to(device)
        combined_tensor = combined_tensor.to(device=device, dtype=dtype)
        combined_tensor = combined_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
        latent = vae.encode_to_latent(combined_tensor)  # [1, C, T', h, w]
        latent = latent.squeeze(0).permute(1, 0, 2, 3)  # [C, T', h, w]

    return latent, combined_rgb


class ProjectionCompositor:
    """Compose scene and event projections for State Adapter input.

    Returns scene_proj and fg_proj separately. The UnifiedBackbonePipeline's
    _prepare_sp_context() handles the channel concatenation.
    """

    def __init__(self) -> None:
        pass

    def compose(
        self,
        scene_proj: torch.Tensor,
        event_projs: list[torch.Tensor],
        vae=None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compose scene projection with event projections.

        Args:
            scene_proj: Scene projection latent, shape [C, T, h, w].
            event_projs: List of event projection latents, each [C, T, h, w].
            vae: VAE wrapper (needed for pixel-space merge when multiple events).
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            Tuple of (scene_proj, fg_proj):
            - scene_proj: [C, T, h, w] (unchanged)
            - fg_proj: [C, T, h, w] or None if no events
        """
        fg_proj = merge_event_projections(event_projs, vae, device, dtype)
        return scene_proj, fg_proj
