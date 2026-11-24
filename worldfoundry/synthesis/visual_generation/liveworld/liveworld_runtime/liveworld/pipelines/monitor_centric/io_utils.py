"""I/O helpers for event-centric outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.utils import save_video_h264


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON file with deterministic formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_video(
    path: str | Path,
    frames: list[np.ndarray] | np.ndarray,
    fps: float = 16.0,
) -> None:
    """Save frames as an MP4 video.

    Args:
        path: Output video path.
        frames: List of RGB frames [H, W, 3] uint8, or array [T, H, W, 3].
        fps: Frames per second.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, np.ndarray):
        frames = list(frames)
    if not frames:
        return

    save_video_h264(out, np.stack(frames, axis=0), fps=fps)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an RGB uint8 image as PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(out))


def save_pointcloud_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Save point cloud as PLY file.

    Args:
        path: Output PLY path.
        points: (N, 3) float32 XYZ coordinates.
        colors: (N, 3) uint8 RGB colors. If None, defaults to white.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    N = len(points)
    if colors is None:
        colors = np.full((N, 3), 255, dtype=np.uint8)

    with out.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def decode_latent_to_rgb(
    latent: torch.Tensor,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Decode VAE latent to RGB frames.

    Args:
        latent: Latent tensor [C, T, h, w] or [T, C, h, w].
        vae: VAE wrapper with decode method.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        RGB frames array [T, H, W, 3] uint8.
    """
    # Ensure shape is [1, C, T, H, W] for VAE decode.
    if latent.dim() == 4:
        # Assume [C, T, h, w], add batch dim.
        latent = latent.unsqueeze(0)

    latent = latent.to(device=device, dtype=dtype)

    with torch.no_grad():
        # Ensure VAE is on the correct device
        vae_device = next(vae.model.parameters()).device
        if vae_device != device:
            vae.model.to(device)
            vae.mean = vae.mean.to(device)
            vae.std = vae.std.to(device)

        # WanVAEWrapper uses decode_to_pixel, expects [B, T, C, H, W]
        # Our latent is [1, C, T, H, W], need to permute to [1, T, C, H, W]
        if latent.dim() == 5 and latent.shape[1] == 16:  # noqa: SIM108 - Branches carry per-case comments.  # [1, C, T, H, W]
            latent_for_vae = latent.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
        else:
            latent_for_vae = latent

        decoded = vae.decode_to_pixel(latent_for_vae)  # Returns [B, T, C, H, W]

    # Convert to [T, H, W, C] numpy.
    if decoded.dim() == 5:
        # [1, T, C, H, W] -> [T, H, W, C]
        decoded = decoded[0].permute(0, 2, 3, 1)

    frames = decoded.cpu().float().numpy()
    frames = (frames * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    return frames


def visualize_projection_latent(
    latent: torch.Tensor,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Visualize projection latent as RGB video frames.

    For sp_in_dim=32 latents (scene + fg concatenated), decodes both and
    overlays foreground on top of the scene background.

    Args:
        latent: Projection latent [2C, T, h, w] (32 channels) or [C, T, h, w] (16 channels).
        vae: VAE wrapper.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        RGB frames [T, H, W, 3] uint8.
    """
    C = latent.shape[0]

    if C == 32:
        # Split scene and fg.
        scene_latent = latent[:16]  # [16, T, h, w]
        fg_latent = latent[16:]  # [16, T, h, w]

        scene_frames = decode_latent_to_rgb(scene_latent, vae, device, dtype)
        fg_frames = decode_latent_to_rgb(fg_latent, vae, device, dtype)

        # Overlay fg on scene: where fg has non-black pixels, use fg.
        fg_mask = (fg_frames > 10).any(axis=-1, keepdims=True)  # [T, H, W, 1]
        vis_frames = np.where(fg_mask, fg_frames, scene_frames)
        return vis_frames
    else:
        # Single projection.
        return decode_latent_to_rgb(latent, vae, device, dtype)
