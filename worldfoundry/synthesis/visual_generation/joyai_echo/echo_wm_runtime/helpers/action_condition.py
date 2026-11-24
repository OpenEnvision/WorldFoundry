"""Build the private training-space camera condition required by pure UCPE.

The public API exposes only semantic camera controls. The internal calibration
keeps the action trajectory in the same numerical regime as the released model.
"""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.variants.joyai_echo.transformer import (
    ActionBlockConfig,
)

from .action_camera import build_action_pt_from_string

_INTERNAL_TRANSLATION_CALIBRATION = 30.0
_TEMPORAL_COMPRESSION = 8


def action_config(width: int, height: int, num_blocks: int = 48) -> ActionBlockConfig:
    return ActionBlockConfig(
        enabled=True,
        block_indices=list(range(num_blocks)),
        ucpe=True,
        ucpe_attn_dim=1024,
        ucpe_num_heads=8,
        ucpe_patches_x=width // 32,
        ucpe_patches_y=height // 32,
        ucpe_image_width=width,
        ucpe_image_height=height,
        ucpe_freq_base=100.0,
        ucpe_freq_scale=1.0,
    )


def _normalize_trajectory(c2ws: torch.Tensor) -> torch.Tensor:
    anchored = torch.linalg.inv(c2ws[:, 0:1]) @ c2ws
    result = anchored.clone()
    result[..., :3, 3] /= _INTERNAL_TRANSLATION_CALIBRATION
    return result


def _build_action_condition(
    action: str,
    *,
    num_frames: int,
    width: int,
    height: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    fov_deg: float,
    device: torch.device,
    fps: float,
    output_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    pt_data = build_action_pt_from_string(
        action,
        num_frames=num_frames,
        image_width=width,
        image_height=height,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_limit_deg=pitch_limit_deg,
        fov_deg=fov_deg,
        fps=fps,
    )
    c2ws = pt_data["c2ws_raw"].unsqueeze(0).to(device=device, dtype=torch.float32)
    K = pt_data["K_pix"].unsqueeze(0).to(device=device, dtype=torch.float32)
    c2ws = _normalize_trajectory(c2ws)
    latent_frames = (num_frames + 7) // _TEMPORAL_COMPRESSION
    # The released UCPE checkpoint was trained with the stored c2w convention.
    # Keep this internal compatibility choice out of the public configuration.
    viewmats = c2ws[:, ::_TEMPORAL_COMPRESSION][:, :latent_frames]
    Ks = K.unsqueeze(1).expand(-1, latent_frames, -1, -1).contiguous()
    return {
        "ucpe_viewmats": viewmats.to(dtype=output_dtype),
        "ucpe_Ks": Ks.to(dtype=output_dtype),
    }


def build_action_condition(
    action: str,
    *,
    num_frames: int,
    width: int,
    height: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    fov_deg: float,
    device: torch.device,
    fps: float,
) -> dict[str, torch.Tensor]:
    return _build_action_condition(
        action,
        num_frames=num_frames,
        width=width,
        height=height,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_limit_deg=pitch_limit_deg,
        fov_deg=fov_deg,
        device=device,
        fps=fps,
        output_dtype=torch.bfloat16,
    )


def build_causal_action_condition(
    action: str,
    *,
    num_frames: int,
    width: int,
    height: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    fov_deg: float,
    device: torch.device,
    fps: float,
) -> dict[str, torch.Tensor]:
    """Build the FP32 camera path used by bounded anchor translation."""
    return _build_action_condition(
        action,
        num_frames=num_frames,
        width=width,
        height=height,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_limit_deg=pitch_limit_deg,
        fov_deg=fov_deg,
        device=device,
        fps=fps,
        output_dtype=torch.float32,
    )


def build_action_trajectory(
    action: str,
    *,
    num_frames: int,
    translation_speed: float,
    rotation_speed_deg: float,
    pitch_limit_deg: float,
    fps: float,
) -> torch.Tensor:
    """Return the unnormalized camera-to-world path for optional HUD rendering."""
    pt_data = build_action_pt_from_string(
        action,
        num_frames=num_frames,
        image_width=1,
        image_height=1,
        translation_speed=translation_speed,
        rotation_speed_deg=rotation_speed_deg,
        pitch_limit_deg=pitch_limit_deg,
        fov_deg=70.0,
        fps=fps,
    )
    return pt_data["c2ws_raw"]


def validate_action_checkpoint_keys(keys: list[str]) -> None:
    forbidden = ("plucker", "fine_proj", "kbd_", "action_encoder", "cam_")
    leaked = [key for key in keys if any(token in key.lower() for token in forbidden)]
    if leaked:
        raise ValueError(f"Checkpoint contains unsupported action parameters: {leaked[:5]}")
